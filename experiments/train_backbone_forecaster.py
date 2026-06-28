import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset

from run_sdrc_experiments import download_datasets, load_dataset


class SlidingWindowDataset(Dataset):
    def __init__(self, arr, input_len, pred_len, max_windows=None, stride=1):
        self.arr = arr.astype("float32")
        total = len(arr) - input_len - pred_len + 1
        if total <= 0:
            raise ValueError("Sequence too short for requested window.")
        starts = np.arange(0, total, max(1, stride), dtype=np.int64)
        if max_windows and max_windows > 0 and len(starts) > max_windows:
            idx = np.linspace(0, len(starts) - 1, max_windows).astype(np.int64)
            starts = starts[idx]
        self.starts = starts
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        x = self.arr[s : s + self.input_len]
        y = self.arr[s + self.input_len : s + self.input_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


class RevIN(nn.Module):
    def __init__(self, channels, affine=True):
        super().__init__()
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, channels))
            self.bias = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, x, mode, stats=None):
        if mode == "norm":
            mean = x.mean(1, keepdim=True).detach()
            std = x.std(1, keepdim=True).detach().clamp_min(1e-5)
            y = (x - mean) / std
            if self.affine:
                y = y * self.weight + self.bias
            return y, (mean, std)
        mean, std = stats
        y = x
        if self.affine:
            y = (y - self.bias) / self.weight.clamp_min(1e-5)
        return y * std + mean


def moving_average(x, kernel_size):
    pad = (kernel_size - 1) // 2
    front = x[:, :1, :].repeat(1, pad, 1)
    end = x[:, -1:, :].repeat(1, pad, 1)
    y = torch.cat([front, x, end], dim=1)
    return torch.nn.functional.avg_pool1d(y.transpose(1, 2), kernel_size=kernel_size, stride=1).transpose(1, 2)


class NLinear(nn.Module):
    def __init__(self, input_len, pred_len, channels, individual=False):
        super().__init__()
        self.individual = individual
        self.channels = channels
        if individual:
            self.linears = nn.ModuleList([nn.Linear(input_len, pred_len) for _ in range(channels)])
        else:
            self.linear = nn.Linear(input_len, pred_len)

    def forward(self, x):
        last = x[:, -1:, :]
        z = (x - last).transpose(1, 2)
        if self.individual:
            outs = [self.linears[i](z[:, i]) for i in range(self.channels)]
            y = torch.stack(outs, dim=-1)
        else:
            y = self.linear(z).transpose(1, 2)
        return y + last


class DLinear(nn.Module):
    def __init__(self, input_len, pred_len, channels, kernel_size=25, individual=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.individual = individual
        self.channels = channels
        if individual:
            self.trend = nn.ModuleList([nn.Linear(input_len, pred_len) for _ in range(channels)])
            self.seasonal = nn.ModuleList([nn.Linear(input_len, pred_len) for _ in range(channels)])
        else:
            self.trend = nn.Linear(input_len, pred_len)
            self.seasonal = nn.Linear(input_len, pred_len)

    def forward(self, x):
        trend = moving_average(x, self.kernel_size)
        seasonal = x - trend
        t = trend.transpose(1, 2)
        s = seasonal.transpose(1, 2)
        if self.individual:
            outs = [self.trend[i](t[:, i]) + self.seasonal[i](s[:, i]) for i in range(self.channels)]
            return torch.stack(outs, dim=-1)
        return (self.trend(t) + self.seasonal(s)).transpose(1, 2)


class MixerBlock(nn.Module):
    def __init__(self, n_tokens, d_model, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token = nn.Sequential(
            nn.Linear(n_tokens, n_tokens * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_tokens * 2, n_tokens),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.channel = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        x = x + self.token(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(self.norm2(x))
        return x


class PatchMixer(nn.Module):
    def __init__(self, input_len, pred_len, channels, d_model=128, patch_len=16, stride=8, depth=3, dropout=0.1):
        super().__init__()
        self.revin = RevIN(channels)
        self.patch_len = min(patch_len, input_len)
        self.stride = max(1, stride)
        self.n_patches = 1 + max(0, (input_len - self.patch_len) // self.stride)
        self.proj = nn.Linear(self.patch_len, d_model)
        self.blocks = nn.ModuleList([MixerBlock(self.n_patches, d_model, dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(self.n_patches * d_model),
            nn.Linear(self.n_patches * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len),
        )

    def forward(self, x):
        b, _, c = x.shape
        z, stats = self.revin(x, "norm")
        u = z.transpose(1, 2).reshape(b * c, -1)
        patches = u.unfold(1, self.patch_len, self.stride)
        h = self.proj(patches)
        for block in self.blocks:
            h = block(h)
        y = self.head(h.reshape(b * c, -1)).reshape(b, c, -1).transpose(1, 2)
        return self.revin(y, "denorm", stats)


class TSMixer(nn.Module):
    def __init__(self, input_len, pred_len, channels, d_model=128, depth=4, dropout=0.1):
        super().__init__()
        self.revin = RevIN(channels)
        self.temporal = nn.Linear(input_len, pred_len)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(channels),
                        "channel": nn.Sequential(
                            nn.Linear(channels, d_model),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model, channels),
                        ),
                        "norm2": nn.LayerNorm(channels),
                        "temporal": nn.Sequential(
                            nn.Linear(pred_len, pred_len * 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(pred_len * 2, pred_len),
                        ),
                    }
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        z, stats = self.revin(x, "norm")
        y = self.temporal(z.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            y = y + block["channel"](block["norm1"](y))
            y = y + block["temporal"](block["norm2"](y).transpose(1, 2)).transpose(1, 2)
        return self.revin(y, "denorm", stats)


class _ACPatchBranch(nn.Module):
    def __init__(self, input_len, pred_len, d_model, patch_len, stride, depth, dropout):
        super().__init__()
        self.patch_len = min(int(patch_len), input_len)
        self.stride = max(1, int(stride))
        self.n_patches = 1 + max(0, (input_len - self.patch_len) // self.stride)
        self.proj = nn.Linear(self.patch_len, d_model)
        self.blocks = nn.ModuleList([MixerBlock(self.n_patches, d_model, dropout) for _ in range(max(1, depth))])
        self.head = nn.Sequential(
            nn.LayerNorm(self.n_patches * d_model),
            nn.Linear(self.n_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, z):
        b, _, c = z.shape
        u = z.transpose(1, 2).reshape(b * c, -1)
        patches = u.unfold(1, self.patch_len, self.stride)
        h = self.proj(patches)
        for block in self.blocks:
            h = block(h)
        return self.head(h.reshape(b * c, -1)).reshape(b, c, -1).transpose(1, 2)


class ActionCompatibleMixer(nn.Module):
    """Forecast-field backbone designed to leave stable structural actions for RASPF."""

    def __init__(
        self,
        input_len,
        pred_len,
        channels,
        d_model=128,
        depth=3,
        dropout=0.1,
        patch_sizes=(16, 32, 64),
        patch_stride=8,
        kernel_size=25,
        dataset_period=24,
    ):
        super().__init__()
        self.revin = RevIN(channels)
        self.input_len = input_len
        self.pred_len = pred_len
        self.channels = channels
        self.dataset_period = max(1, int(dataset_period))
        self.trend = DLinear(input_len, pred_len, channels, kernel_size=kernel_size, individual=False)
        self.temporal = nn.Linear(input_len, pred_len)
        self.mix_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(channels),
                        "channel": nn.Sequential(
                            nn.Linear(channels, d_model),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model, channels),
                        ),
                        "norm2": nn.LayerNorm(channels),
                        "temporal": nn.Sequential(
                            nn.Linear(pred_len, pred_len * 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(pred_len * 2, pred_len),
                        ),
                    }
                )
                for _ in range(max(1, depth))
            ]
        )
        sizes = tuple(int(p) for p in patch_sizes if int(p) > 1)
        if not sizes:
            sizes = (16, 32, 64)
        self.patch_branches = nn.ModuleList(
            [_ACPatchBranch(input_len, pred_len, d_model, p, patch_stride, max(1, depth // 2), dropout) for p in sizes]
        )
        self.branch_norm = nn.Identity()
        self.gate = nn.Sequential(
            nn.Linear(5, max(16, d_model // 4)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, d_model // 4), 4),
        )
        self.output_blend = nn.Parameter(torch.tensor([-2.0, 3.0, -2.0, -2.0], dtype=torch.float32))

    def _positive_corr_weight(self, z):
        recent = z[:, -min(self.dataset_period, z.shape[1]) :, :]
        centered = recent - recent.mean(dim=1, keepdim=True)
        norm = centered.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-5)
        u = centered / norm
        corr = torch.bmm(u.transpose(1, 2), u).clamp_min(0.0)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        eye = torch.eye(corr.shape[-1], device=corr.device, dtype=corr.dtype).unsqueeze(0)
        corr = corr * (1.0 - eye)
        denom = corr.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        return torch.nan_to_num(corr / denom, nan=0.0, posinf=0.0, neginf=0.0)

    def _descriptors(self, z):
        recent = z[:, -min(self.dataset_period, z.shape[1]) :, :]
        vol = recent.std(dim=(1, 2)).clamp_min(1e-5)
        level = recent.mean(dim=(1, 2)).abs()
        drift = (recent[:, -1, :] - recent[:, 0, :]).abs().mean(dim=1)
        spread = (recent.quantile(0.75, dim=1) - recent.quantile(0.25, dim=1)).abs().mean(dim=1)
        corr_mass = self._positive_corr_weight(z).mean(dim=(1, 2))
        return torch.nan_to_num(torch.stack([vol, level, drift, spread, corr_mass], dim=-1), nan=0.0, posinf=10.0, neginf=-10.0)

    def forward(self, x):
        z, stats = self.revin(x, "norm")
        trend = self.trend(z)
        temporal = self.temporal(z.transpose(1, 2)).transpose(1, 2)
        for block in self.mix_blocks:
            temporal = temporal + block["channel"](block["norm1"](temporal))
            temporal = temporal + block["temporal"](block["norm2"](temporal).transpose(1, 2)).transpose(1, 2)
        patch = torch.stack([branch(z) for branch in self.patch_branches], dim=0).mean(dim=0)
        w = self._positive_corr_weight(z).detach()
        graph = torch.bmm(temporal.reshape(-1, self.channels).unsqueeze(1), w.repeat_interleave(self.pred_len, dim=0).transpose(1, 2))
        graph = graph.squeeze(1).reshape(z.shape[0], self.pred_len, self.channels)
        desc = self._descriptors(z).detach()
        gate = torch.softmax(self.gate(desc) + self.output_blend, dim=-1)
        branches = torch.stack([trend, temporal, patch, graph], dim=1)
        y = (branches * gate[:, :, None, None]).sum(dim=1)
        y = torch.nan_to_num(self.branch_norm(y), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        return self.revin(y, "denorm", stats)


class StructureConditionedMixer(nn.Module):
    """Multi-scale mixer whose branch weights are conditioned on local time-series structure."""

    def __init__(
        self,
        input_len,
        pred_len,
        channels,
        d_model=128,
        depth=3,
        dropout=0.1,
        patch_sizes=(8, 16, 32, 64),
        patch_stride=8,
        kernel_size=25,
        dataset_period=24,
    ):
        super().__init__()
        self.revin = RevIN(channels)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.dataset_period = max(1, int(dataset_period))
        self.kernel_size = max(3, int(kernel_size) | 1)

        self.direct = nn.Linear(input_len, pred_len)
        self.trend = nn.Linear(input_len, pred_len)
        sizes = tuple(int(p) for p in patch_sizes if int(p) > 1)
        self.patch_branches = nn.ModuleList(
            [_ACPatchBranch(input_len, pred_len, d_model, p, patch_stride, max(1, depth), dropout) for p in sizes]
        )
        self.channel_refine = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, channels),
        )
        self.phase_refine = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, channels),
        )
        self.gate = nn.Sequential(
            nn.Linear(7, max(24, d_model // 4)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(24, d_model // 4), 5),
        )
        self.horizon_scale = nn.Parameter(torch.ones(1, pred_len, 1))
        self.horizon_bias = nn.Parameter(torch.zeros(1, pred_len, 1))
        self.register_buffer("gate_prior", torch.tensor([0.15, 0.35, 0.25, 0.15, 0.10]).log())

    def _corr_mass(self, z):
        recent = z[:, -min(self.dataset_period, z.shape[1]) :, :]
        centered = recent - recent.mean(dim=1, keepdim=True)
        norm = centered.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-5)
        u = centered / norm
        corr = torch.bmm(u.transpose(1, 2), u).clamp_min(0.0)
        eye = torch.eye(corr.shape[-1], device=z.device, dtype=z.dtype).unsqueeze(0)
        corr = corr * (1.0 - eye)
        return torch.nan_to_num(corr.mean(dim=(1, 2)), nan=0.0, posinf=0.0, neginf=0.0)

    def _descriptors(self, z, trend, seasonal, phase):
        recent = z[:, -min(self.dataset_period, z.shape[1]) :, :]
        vol = recent.std(dim=(1, 2)).clamp_min(1e-5)
        drift = (recent[:, -1, :] - recent[:, 0, :]).abs().mean(dim=1)
        trend_energy = trend.square().mean(dim=(1, 2)).sqrt()
        seasonal_energy = seasonal.square().mean(dim=(1, 2)).sqrt()
        phase_error = (phase[:, : min(self.pred_len, z.shape[1]), :] - z[:, -min(self.pred_len, z.shape[1]) :, :]).abs()
        phase_instability = phase_error.mean(dim=(1, 2)) if phase_error.numel() else vol
        iqr = (recent.quantile(0.75, dim=1) - recent.quantile(0.25, dim=1)).abs().mean(dim=1)
        return torch.nan_to_num(
            torch.stack([vol, drift, trend_energy, seasonal_energy, phase_instability, iqr, self._corr_mass(z)], dim=-1),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    def forward(self, x):
        z, stats = self.revin(x, "norm")
        trend_hist = moving_average(z, self.kernel_size)
        seasonal = z - trend_hist

        direct = self.direct(z.transpose(1, 2)).transpose(1, 2)
        trend = self.trend(trend_hist.transpose(1, 2)).transpose(1, 2)
        if self.patch_branches:
            patch = torch.stack([branch(seasonal) for branch in self.patch_branches], dim=0).mean(dim=0)
        else:
            patch = direct
        phase = _torch_phase_profile(z, self.pred_len, self.dataset_period, "mean")
        phase = phase + self.phase_refine(phase)
        channel = direct + self.channel_refine(direct)

        desc = self._descriptors(z, trend_hist, seasonal, phase).detach()
        gate = torch.softmax(self.gate(desc) + self.gate_prior.to(desc.device), dim=-1)
        branches = torch.stack([trend, direct, patch, phase, channel], dim=1)
        y = (branches * gate[:, :, None, None]).sum(dim=1)
        y = y * self.horizon_scale + self.horizon_bias
        y = torch.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        return self.revin(y, "denorm", stats)


class SeasonalTrendMixer(nn.Module):
    """A compact decomposed MLP-Mixer: linear trend extrapolation plus residual seasonal mixing."""

    def __init__(self, input_len, pred_len, channels, d_model=128, depth=3, dropout=0.1, kernel_size=25):
        super().__init__()
        self.revin = RevIN(channels)
        self.kernel_size = max(3, int(kernel_size) | 1)
        self.trend_head = nn.Linear(input_len, pred_len)
        self.seasonal_head = nn.Linear(input_len, pred_len)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(channels),
                        "channel": nn.Sequential(
                            nn.Linear(channels, d_model),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model, channels),
                        ),
                        "norm2": nn.LayerNorm(channels),
                        "temporal": nn.Sequential(
                            nn.Linear(pred_len, pred_len * 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(pred_len * 2, pred_len),
                        ),
                    }
                )
                for _ in range(max(1, depth))
            ]
        )
        self.blend = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))

    def forward(self, x):
        z, stats = self.revin(x, "norm")
        trend_hist = moving_average(z, self.kernel_size)
        seasonal = z - trend_hist
        trend = self.trend_head(trend_hist.transpose(1, 2)).transpose(1, 2)
        y = self.seasonal_head(seasonal.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            y = y + block["channel"](block["norm1"](y))
            y = y + block["temporal"](block["norm2"](y).transpose(1, 2)).transpose(1, 2)
        w = torch.softmax(self.blend, dim=0)
        return self.revin(w[0] * trend + w[1] * y, "denorm", stats)


class ProbabilisticDecomposedMixer(nn.Module):
    """Decomposed Mixer with interleaved residual-distribution heads."""

    def __init__(
        self,
        input_len,
        pred_len,
        channels,
        d_model=128,
        depth=3,
        dropout=0.1,
        kernel_size=25,
        dataset_period=24,
        support_size=64,
        support_scale=1.25,
    ):
        super().__init__()
        self.revin = RevIN(channels)
        self.kernel_size = max(3, int(kernel_size) | 1)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.dataset_period = max(1, int(dataset_period))
        self.support_size = int(support_size)
        self.support_scale = float(support_scale)

        self.trend_head = nn.Linear(input_len, pred_len)
        self.seasonal_head = nn.Linear(input_len, pred_len)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(channels),
                        "channel": nn.Sequential(
                            nn.Linear(channels, d_model),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model, channels),
                        ),
                        "norm2": nn.LayerNorm(channels),
                        "temporal": nn.Sequential(
                            nn.Linear(pred_len, pred_len * 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(pred_len * 2, pred_len),
                        ),
                    }
                )
                for _ in range(max(1, depth))
            ]
        )
        self.feature_norm = nn.LayerNorm(5)
        self.logit_head_1 = nn.Sequential(nn.Linear(5, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, support_size))
        self.logit_head_2 = nn.Sequential(nn.Linear(5, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, support_size))
        self.anchor_blend = nn.Parameter(torch.tensor([1.0, 1.0, 0.25], dtype=torch.float32))

        q = torch.linspace(0.001, 0.999, support_size)
        support = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
        support = support * self.support_scale
        shift = 0.5 * torch.median(support[1:] - support[:-1])
        self.register_buffer("support_1", support)
        self.register_buffer("support_2", support + shift)

    def _normalize_target(self, y, stats):
        mean, std = stats
        z = (y - mean) / std
        if self.revin.affine:
            z = z * self.revin.weight + self.revin.bias
        return z

    def _nearest_index(self, residual, support):
        return (residual.unsqueeze(-1) - support.view(1, 1, 1, -1)).abs().argmin(dim=-1)

    def _forward_norm(self, x):
        z, stats = self.revin(x, "norm")
        trend_hist = moving_average(z, self.kernel_size)
        seasonal_hist = z - trend_hist
        trend = self.trend_head(trend_hist.transpose(1, 2)).transpose(1, 2)
        seasonal = self.seasonal_head(seasonal_hist.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            seasonal = seasonal + block["channel"](block["norm1"](seasonal))
            seasonal = seasonal + block["temporal"](block["norm2"](seasonal).transpose(1, 2)).transpose(1, 2)
        phase = _torch_phase_profile(z, self.pred_len, self.dataset_period, "mean")
        w = torch.softmax(self.anchor_blend, dim=0)
        anchor = w[0] * trend + w[1] * seasonal + w[2] * phase
        local_slope = anchor - torch.cat([anchor[:, :1, :], anchor[:, :-1, :]], dim=1)
        features = torch.stack([trend, seasonal, phase, anchor, local_slope], dim=-1)
        features = self.feature_norm(torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0))
        logits_1 = self.logit_head_1(features)
        logits_2 = self.logit_head_2(features)
        prob_1 = torch.softmax(logits_1, dim=-1)
        prob_2 = torch.softmax(logits_2, dim=-1)
        corr_1 = (prob_1 * self.support_1.view(1, 1, 1, -1)).sum(dim=-1)
        corr_2 = (prob_2 * self.support_2.view(1, 1, 1, -1)).sum(dim=-1)
        conf_1 = prob_1.max(dim=-1).values
        conf_2 = prob_2.max(dim=-1).values
        mix = conf_1 / (conf_1 + conf_2).clamp_min(1e-6)
        pred_norm = anchor + mix * corr_1 + (1.0 - mix) * corr_2
        return pred_norm, stats, anchor, logits_1, logits_2

    def forward(self, x):
        pred_norm, stats, *_ = self._forward_norm(x)
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        return self.revin(pred_norm, "denorm", stats)

    def training_step_loss(self, x, y, args):
        pred_norm, stats, anchor, logits_1, logits_2 = self._forward_norm(x)
        target_norm = self._normalize_target(y, stats)
        residual = (target_norm - anchor).detach().clamp(-8.0, 8.0)
        target_1 = self._nearest_index(residual, self.support_1)
        target_2 = self._nearest_index(residual, self.support_2)
        ce_1 = torch.nn.functional.cross_entropy(logits_1.reshape(-1, self.support_size), target_1.reshape(-1))
        ce_2 = torch.nn.functional.cross_entropy(logits_2.reshape(-1, self.support_size), target_2.reshape(-1))
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        exp_loss = loss_fn(pred_norm, target_norm, str(getattr(args, "loss", "huber")))
        anchor_loss = torch.nn.functional.smooth_l1_loss(anchor, target_norm, beta=0.5)
        ce_weight = float(getattr(args, "dist_ce_weight", 0.05))
        anchor_weight = float(getattr(args, "dist_anchor_weight", 0.2))
        loss = exp_loss + ce_weight * 0.5 * (ce_1 + ce_2) + anchor_weight * anchor_loss
        pred = self.revin(pred_norm, "denorm", stats)
        return pred, loss


class DirectDistributionMixer(nn.Module):
    """Direct per-step value distribution mixer with trend/phase logits priors."""

    def __init__(
        self,
        input_len,
        pred_len,
        channels,
        d_model=128,
        depth=3,
        dropout=0.1,
        kernel_size=25,
        dataset_period=24,
        support_size=96,
        support_scale=1.25,
    ):
        super().__init__()
        self.revin = RevIN(channels)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.kernel_size = max(3, int(kernel_size) | 1)
        self.dataset_period = max(1, int(dataset_period))
        self.support_size = int(support_size)

        self.direct_head = nn.Linear(input_len, pred_len)
        self.trend_head = nn.Linear(input_len, pred_len)
        self.seasonal_head = nn.Linear(input_len, pred_len)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(channels),
                        "channel": nn.Sequential(
                            nn.Linear(channels, d_model),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model, channels),
                        ),
                        "norm2": nn.LayerNorm(channels),
                        "temporal": nn.Sequential(
                            nn.Linear(pred_len, pred_len * 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(pred_len * 2, pred_len),
                        ),
                    }
                )
                for _ in range(max(1, depth))
            ]
        )
        self.feature_norm = nn.LayerNorm(7)
        self.logit_head_1 = nn.Sequential(nn.Linear(7, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, support_size))
        self.logit_head_2 = nn.Sequential(nn.Linear(7, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, support_size))
        self.prior_blend = nn.Parameter(torch.tensor([0.35, 0.35, 0.20, 0.10], dtype=torch.float32))

        q = torch.linspace(0.001, 0.999, support_size)
        support = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0) * float(support_scale)
        shift = 0.5 * torch.median(support[1:] - support[:-1])
        self.register_buffer("support_1", support)
        self.register_buffer("support_2", support + shift)

    def _normalize_target(self, y, stats):
        mean, std = stats
        z = (y - mean) / std
        if self.revin.affine:
            z = z * self.revin.weight + self.revin.bias
        return z

    def _nearest_index(self, target, support):
        return (target.unsqueeze(-1) - support.view(1, 1, 1, -1)).abs().argmin(dim=-1)

    def _logit_prior(self, center, support, args):
        prior_weight = float(getattr(args, "dist_prior_weight", 0.35))
        prior_width = max(1e-3, float(getattr(args, "dist_prior_width", 0.75)))
        d2 = (support.view(1, 1, 1, -1) - center.unsqueeze(-1)).square()
        return -prior_weight * d2 / (2.0 * prior_width * prior_width)

    def _forward_norm(self, x, args=None):
        z, stats = self.revin(x, "norm")
        trend_hist = moving_average(z, self.kernel_size)
        seasonal_hist = z - trend_hist
        direct = self.direct_head(z.transpose(1, 2)).transpose(1, 2)
        trend = self.trend_head(trend_hist.transpose(1, 2)).transpose(1, 2)
        seasonal = self.seasonal_head(seasonal_hist.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            seasonal = seasonal + block["channel"](block["norm1"](seasonal))
            seasonal = seasonal + block["temporal"](block["norm2"](seasonal).transpose(1, 2)).transpose(1, 2)
        phase = _torch_phase_profile(z, self.pred_len, self.dataset_period, "mean")
        slope = direct - torch.cat([direct[:, :1, :], direct[:, :-1, :]], dim=1)
        spread = (seasonal - phase).abs()
        prior_w = torch.softmax(self.prior_blend, dim=0)
        prior_center = prior_w[0] * direct + prior_w[1] * trend + prior_w[2] * seasonal + prior_w[3] * phase

        features = torch.stack([direct, trend, seasonal, phase, prior_center, slope, spread], dim=-1)
        features = self.feature_norm(torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0))
        logits_1 = self.logit_head_1(features)
        logits_2 = self.logit_head_2(features)
        if args is not None:
            logits_1 = logits_1 + self._logit_prior(prior_center, self.support_1, args)
            logits_2 = logits_2 + self._logit_prior(prior_center, self.support_2, args)
        prob_1 = torch.softmax(logits_1, dim=-1)
        prob_2 = torch.softmax(logits_2, dim=-1)
        pred_1 = (prob_1 * self.support_1.view(1, 1, 1, -1)).sum(dim=-1)
        pred_2 = (prob_2 * self.support_2.view(1, 1, 1, -1)).sum(dim=-1)
        conf_1 = prob_1.max(dim=-1).values
        conf_2 = prob_2.max(dim=-1).values
        mix = conf_1 / (conf_1 + conf_2).clamp_min(1e-6)
        pred_norm = mix * pred_1 + (1.0 - mix) * pred_2
        return pred_norm, stats, prior_center, logits_1, logits_2, pred_1, pred_2

    def forward(self, x):
        pred_norm, stats, *_ = self._forward_norm(x, None)
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        return self.revin(pred_norm, "denorm", stats)

    def training_step_loss(self, x, y, args):
        pred_norm, stats, _prior_center, logits_1, logits_2, pred_1, pred_2 = self._forward_norm(x, args)
        target_norm = self._normalize_target(y, stats).clamp(-8.0, 8.0)
        target_1 = self._nearest_index(target_norm, self.support_1)
        target_2 = self._nearest_index(target_norm, self.support_2)
        ce_1 = torch.nn.functional.cross_entropy(logits_1.reshape(-1, self.support_size), target_1.reshape(-1))
        ce_2 = torch.nn.functional.cross_entropy(logits_2.reshape(-1, self.support_size), target_2.reshape(-1))
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        exp_loss = loss_fn(pred_norm, target_norm, str(getattr(args, "loss", "huber")))
        consistency = torch.nn.functional.smooth_l1_loss(pred_1, pred_2, beta=0.5)
        ce_weight = float(getattr(args, "dist_ce_weight", 0.10))
        consistency_weight = float(getattr(args, "dist_consistency_weight", 0.05))
        loss = exp_loss + ce_weight * 0.5 * (ce_1 + ce_2) + consistency_weight * consistency
        pred = self.revin(pred_norm, "denorm", stats)
        return pred, loss


class HybridDistributionMixer(DirectDistributionMixer):
    """Deterministic decomposed Mixer with a bounded direct-distribution refinement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_gate_logit = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

    def _hybrid_forward_norm(self, x, args=None):
        dist_norm, stats, prior_center, logits_1, logits_2, pred_1, pred_2 = super()._forward_norm(x, args)
        max_mix = float(getattr(args, "dist_max_mix", 0.35)) if args is not None else 0.35
        mix = torch.sigmoid(self.dist_gate_logit) * max(0.0, min(1.0, max_mix))
        pred_norm = (1.0 - mix) * prior_center + mix * dist_norm
        return pred_norm, stats, prior_center, logits_1, logits_2, pred_1, pred_2, mix

    def forward(self, x):
        pred_norm, stats, *_ = self._hybrid_forward_norm(x, None)
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        return self.revin(pred_norm, "denorm", stats)

    def training_step_loss(self, x, y, args):
        pred_norm, stats, prior_center, logits_1, logits_2, pred_1, pred_2, mix = self._hybrid_forward_norm(x, args)
        target_norm = self._normalize_target(y, stats).clamp(-8.0, 8.0)
        target_1 = self._nearest_index(target_norm, self.support_1)
        target_2 = self._nearest_index(target_norm, self.support_2)
        ce_1 = torch.nn.functional.cross_entropy(logits_1.reshape(-1, self.support_size), target_1.reshape(-1))
        ce_2 = torch.nn.functional.cross_entropy(logits_2.reshape(-1, self.support_size), target_2.reshape(-1))
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        pred_loss = loss_fn(pred_norm, target_norm, str(getattr(args, "loss", "huber")))
        anchor_loss = torch.nn.functional.smooth_l1_loss(prior_center, target_norm, beta=0.5)
        consistency = torch.nn.functional.smooth_l1_loss(pred_1, pred_2, beta=0.5)
        ce_weight = float(getattr(args, "dist_ce_weight", 0.03))
        anchor_weight = float(getattr(args, "dist_anchor_weight", 0.10))
        consistency_weight = float(getattr(args, "dist_consistency_weight", 0.03))
        mix_penalty = float(getattr(args, "dist_mix_penalty", 0.01)) * mix.square()
        loss = pred_loss + anchor_weight * anchor_loss + ce_weight * 0.5 * (ce_1 + ce_2) + consistency_weight * consistency + mix_penalty
        pred = self.revin(pred_norm, "denorm", stats)
        return pred, loss


class _PatchConvBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1, kernel_size=3):
        super().__init__()
        pad = max(1, int(kernel_size) // 2)
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=pad),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        y = self.norm(x).transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        x = x + y[:, : x.shape[1], :]
        return x + self.ffn(self.ffn_norm(x))


class PatchConvResidualMixer(nn.Module):
    """Deterministic EMA-decomposed patch-convolution residual Mixer."""

    def __init__(
        self,
        input_len,
        pred_len,
        channels,
        d_model=128,
        depth=3,
        dropout=0.1,
        patch_len=16,
        stride=8,
        kernel_size=25,
        dataset_period=24,
        conv_kernel=3,
    ):
        super().__init__()
        self.revin = RevIN(channels)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.kernel_size = max(3, int(kernel_size) | 1)
        self.dataset_period = max(1, int(dataset_period))
        self.patch_len = min(int(patch_len), self.input_len)
        self.stride = max(1, int(stride))
        self.n_patches = 1 + max(0, (self.input_len - self.patch_len) // self.stride)

        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.blocks = nn.ModuleList([_PatchConvBlock(d_model, dropout, conv_kernel) for _ in range(max(1, depth))])
        self.seasonal_head = nn.Sequential(
            nn.LayerNorm(self.n_patches * d_model),
            nn.Linear(self.n_patches * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len),
        )
        self.trend_head = nn.Linear(input_len, pred_len)
        self.direct_head = nn.Linear(input_len, pred_len)
        self.phase_refine = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, max(32, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, d_model // 2), channels),
        )
        self.branch_gate = nn.Sequential(
            nn.Linear(5, max(16, d_model // 4)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, d_model // 4), 4),
        )
        self.branch_prior = nn.Parameter(torch.tensor([1.8, 1.8, -0.8, -1.2], dtype=torch.float32))
        self.channel_refine = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, max(32, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, d_model // 2), channels),
        )
        self.refine_scale = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

    def _descriptors(self, z, trend_hist, seasonal_hist):
        recent = z[:, -min(self.dataset_period, z.shape[1]) :, :]
        vol = recent.std(dim=(1, 2)).clamp_min(1e-5)
        drift = (recent[:, -1, :] - recent[:, 0, :]).abs().mean(dim=1)
        trend_energy = trend_hist.square().mean(dim=(1, 2)).sqrt()
        seasonal_energy = seasonal_hist.square().mean(dim=(1, 2)).sqrt()
        spread = (recent.quantile(0.75, dim=1) - recent.quantile(0.25, dim=1)).abs().mean(dim=1)
        return torch.nan_to_num(torch.stack([vol, drift, trend_energy, seasonal_energy, spread], dim=-1), nan=0.0)

    def _seasonal_patch_stream(self, seasonal):
        b, _, c = seasonal.shape
        u = seasonal.transpose(1, 2).reshape(b * c, -1)
        patches = u.unfold(1, self.patch_len, self.stride)
        h = self.patch_proj(patches) + self.pos
        for block in self.blocks:
            h = block(h)
        return self.seasonal_head(h.reshape(b * c, -1)).reshape(b, c, -1).transpose(1, 2)

    def _components_norm(self, z):
        trend_hist = moving_average(z, self.kernel_size)
        seasonal_hist = z - trend_hist
        trend = self.trend_head(trend_hist.transpose(1, 2)).transpose(1, 2)
        seasonal = self._seasonal_patch_stream(seasonal_hist)
        direct = self.direct_head(z.transpose(1, 2)).transpose(1, 2)
        phase = _torch_phase_profile(z, self.pred_len, self.dataset_period, "mean")
        phase = phase + 0.1 * self.phase_refine(phase)
        return trend, seasonal, direct, phase, trend_hist, seasonal_hist

    def forward_norm(self, x):
        z, stats = self.revin(x, "norm")
        trend, seasonal, direct, phase, trend_hist, seasonal_hist = self._components_norm(z)
        desc = self._descriptors(z, trend_hist, seasonal_hist).detach()
        gate = torch.softmax(self.branch_gate(desc) + self.branch_prior, dim=-1)
        branches = torch.stack([trend, seasonal, direct, phase], dim=1)
        y = (branches * gate[:, :, None, None]).sum(dim=1)
        y = y + torch.sigmoid(self.refine_scale) * self.channel_refine(y)
        y = torch.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        return y, stats

    def forward(self, x):
        y, stats = self.forward_norm(x)
        return self.revin(y, "denorm", stats)


class PhaseAnchoredPCRMixer(PatchConvResidualMixer):
    """PCRMixer with the empirically useful phase pull-in made part of the forecast map."""

    def __init__(self, *args, phase_alpha_init=0.4, phase_alpha_max=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_gate = nn.Parameter(torch.tensor([1.2, 1.2, -0.4], dtype=torch.float32))
        self.phase_alpha_max = float(phase_alpha_max)
        init = max(1e-4, min(1 - 1e-4, float(phase_alpha_init) / max(1e-4, self.phase_alpha_max)))
        self.phase_alpha_logit = nn.Parameter(torch.tensor(math.log(init / (1.0 - init)), dtype=torch.float32))
        self.phase_residual_scale = nn.Parameter(torch.tensor(-2.5, dtype=torch.float32))

    def forward_norm(self, x):
        z, stats = self.revin(x, "norm")
        trend, seasonal, direct, phase, _trend_hist, _seasonal_hist = self._components_norm(z)
        gate = torch.softmax(self.base_gate, dim=0)
        base = gate[0] * trend + gate[1] * seasonal + gate[2] * direct
        alpha = self.phase_alpha_max * torch.sigmoid(self.phase_alpha_logit)
        residual = torch.sigmoid(self.phase_residual_scale) * self.channel_refine(base)
        y = (1.0 - alpha) * base + alpha * phase + residual
        y = torch.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        return y, stats


def _coarse_mean(x, factor):
    factor = max(1, int(factor))
    if factor <= 1:
        return x
    usable = (x.shape[1] // factor) * factor
    if usable <= 0:
        return x.mean(dim=1, keepdim=True)
    y = x[:, :usable, :].reshape(x.shape[0], usable // factor, factor, x.shape[2]).mean(dim=2)
    if usable < x.shape[1]:
        tail = x[:, usable:, :].mean(dim=1, keepdim=True)
        y = torch.cat([y, tail], dim=1)
    return y


class CoarseFinePCRMixer(PatchConvResidualMixer):
    """PCRMixer trained with coarse target supervision and fine/coarse consistency."""

    def __init__(self, *args, coarse_factor=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.coarse_factor = max(1, int(coarse_factor))
        self.coarse_len = math.ceil(self.pred_len / self.coarse_factor)
        self.coarse_head = nn.Linear(self.pred_len, self.coarse_len)

    def _normalize_target(self, y, stats):
        mean, std = stats
        z = (y - mean) / std
        if self.revin.affine:
            z = z * self.revin.weight + self.revin.bias
        return z

    def training_step_loss(self, x, y, args):
        pred_norm, stats = self.forward_norm(x)
        target_norm = self._normalize_target(y, stats)
        pred = self.revin(pred_norm, "denorm", stats)
        base = loss_fn(pred, y, str(getattr(args, "loss", "huber")))
        target_coarse = _coarse_mean(target_norm, self.coarse_factor)
        pred_down = _coarse_mean(pred_norm, self.coarse_factor)
        pred_coarse = self.coarse_head(pred_norm.transpose(1, 2)).transpose(1, 2)
        if pred_coarse.shape[1] != target_coarse.shape[1]:
            m = min(pred_coarse.shape[1], target_coarse.shape[1])
            pred_coarse = pred_coarse[:, :m, :]
            target_coarse = target_coarse[:, :m, :]
            pred_down = pred_down[:, :m, :]
        beta = float(getattr(args, "coarse_loss_weight", 0.2))
        gamma = float(getattr(args, "coarse_consistency_weight", 0.1))
        coarse = torch.nn.functional.smooth_l1_loss(pred_coarse, target_coarse, beta=0.5)
        consistency = torch.nn.functional.mse_loss(pred_down, pred_coarse)
        return pred, base + beta * coarse + gamma * consistency


class PhaseAnchoredCoarseFineMixer(PhaseAnchoredPCRMixer):
    """Phase-anchored PCRMixer with coarse/fine consistency training."""

    def __init__(self, *args, coarse_factor=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.coarse_factor = max(1, int(coarse_factor))
        self.coarse_len = math.ceil(self.pred_len / self.coarse_factor)
        self.coarse_head = nn.Linear(self.pred_len, self.coarse_len)

    def _normalize_target(self, y, stats):
        mean, std = stats
        z = (y - mean) / std
        if self.revin.affine:
            z = z * self.revin.weight + self.revin.bias
        return z

    def training_step_loss(self, x, y, args):
        pred_norm, stats = self.forward_norm(x)
        target_norm = self._normalize_target(y, stats)
        pred = self.revin(pred_norm, "denorm", stats)
        base = loss_fn(pred, y, str(getattr(args, "loss", "huber")))
        target_coarse = _coarse_mean(target_norm, self.coarse_factor)
        pred_down = _coarse_mean(pred_norm, self.coarse_factor)
        pred_coarse = self.coarse_head(pred_norm.transpose(1, 2)).transpose(1, 2)
        if pred_coarse.shape[1] != target_coarse.shape[1]:
            m = min(pred_coarse.shape[1], target_coarse.shape[1])
            pred_coarse = pred_coarse[:, :m, :]
            target_coarse = target_coarse[:, :m, :]
            pred_down = pred_down[:, :m, :]
        beta = float(getattr(args, "coarse_loss_weight", 0.2))
        gamma = float(getattr(args, "coarse_consistency_weight", 0.1))
        coarse = torch.nn.functional.smooth_l1_loss(pred_coarse, target_coarse, beta=0.5)
        consistency = torch.nn.functional.mse_loss(pred_down, pred_coarse)
        return pred, base + beta * coarse + gamma * consistency


class PhaseAnchoredHybridDistributionMixer(HybridDistributionMixer):
    """Bounded distribution refinement around a phase-anchored deterministic prior."""

    def __init__(self, *args, phase_alpha_init=0.4, phase_alpha_max=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_alpha_max = float(phase_alpha_max)
        init = max(1e-4, min(1 - 1e-4, float(phase_alpha_init) / max(1e-4, self.phase_alpha_max)))
        self.phase_alpha_logit = nn.Parameter(torch.tensor(math.log(init / (1.0 - init)), dtype=torch.float32))

    def _hybrid_forward_norm(self, x, args=None):
        dist_norm, stats, prior_center, logits_1, logits_2, pred_1, pred_2 = DirectDistributionMixer._forward_norm(self, x, args)
        z, _ = self.revin(x, "norm")
        phase = _torch_phase_profile(z, self.pred_len, self.dataset_period, "mean")
        alpha = self.phase_alpha_max * torch.sigmoid(self.phase_alpha_logit)
        phase_prior = (1.0 - alpha) * prior_center + alpha * phase
        max_mix = float(getattr(args, "dist_max_mix", 0.25)) if args is not None else 0.25
        mix = torch.sigmoid(self.dist_gate_logit) * max(0.0, min(1.0, max_mix))
        pred_norm = (1.0 - mix) * phase_prior + mix * dist_norm
        return pred_norm, stats, phase_prior, logits_1, logits_2, pred_1, pred_2, mix


class SeasonalAnchorLowRankMixer(nn.Module):
    """Patch backbone with explicit seasonal anchors and low-rank cross-variable coupling."""

    def __init__(
        self,
        input_len,
        pred_len,
        channels,
        d_model=128,
        depth=3,
        dropout=0.1,
        patch_len=16,
        stride=8,
        kernel_size=25,
        dataset_period=24,
        conv_kernel=3,
        low_rank=4,
    ):
        super().__init__()
        self.revin = RevIN(channels)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.kernel_size = max(3, int(kernel_size) | 1)
        self.dataset_period = max(1, int(dataset_period))
        self.patch_len = min(int(patch_len), self.input_len)
        self.stride = max(1, int(stride))
        self.n_patches = 1 + max(0, (self.input_len - self.patch_len) // self.stride)
        self.low_rank = max(1, min(int(low_rank), self.channels))

        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.patch_pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.patch_blocks = nn.ModuleList([_PatchConvBlock(d_model, dropout, conv_kernel) for _ in range(max(1, depth))])
        self.patch_head = nn.Sequential(
            nn.LayerNorm(self.n_patches * d_model),
            nn.Linear(self.n_patches * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len),
        )

        self.trend_head = nn.Linear(input_len, pred_len)
        self.direct_encoder = nn.Sequential(
            nn.LayerNorm(input_len),
            nn.Linear(input_len, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.horizon_emb = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.direct_decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.branch_gate = nn.Sequential(
            nn.Linear(7, max(32, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, d_model // 2), 5),
        )
        self.branch_prior = nn.Parameter(torch.tensor([1.2, 1.4, 1.6, 0.6, 0.2], dtype=torch.float32))
        self.mix_left = nn.Parameter(torch.randn(channels, self.low_rank) * 0.02)
        self.mix_right = nn.Parameter(torch.randn(channels, self.low_rank) * 0.02)
        self.mix_scale = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.channel_refine = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, max(32, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, d_model // 2), channels),
        )
        self.refine_scale = nn.Parameter(torch.tensor(-2.5, dtype=torch.float32))

    def _patch_branch(self, seasonal):
        b, _, c = seasonal.shape
        u = seasonal.transpose(1, 2).reshape(b * c, -1)
        patches = u.unfold(1, self.patch_len, self.stride)
        h = self.patch_proj(patches) + self.patch_pos
        for block in self.patch_blocks:
            h = block(h)
        return self.patch_head(h.reshape(b * c, -1)).reshape(b, c, -1).transpose(1, 2)

    def _direct_branch(self, z):
        b, _, c = z.shape
        ctx = self.direct_encoder(z.transpose(1, 2).reshape(b * c, -1)).reshape(b, c, -1)
        h = ctx[:, :, None, :] + self.horizon_emb[:, None, :, :]
        out = self.direct_decoder(h).squeeze(-1)
        return out.transpose(1, 2)

    def _low_rank_mix(self, y):
        mat = torch.matmul(self.mix_left, self.mix_right.transpose(0, 1)) / math.sqrt(float(self.low_rank))
        mat = torch.tanh(mat)
        mixed = torch.einsum("bhc,cd->bhd", y, mat)
        return y + torch.sigmoid(self.mix_scale) * mixed

    def _descriptors(self, z, trend_hist, seasonal_hist, phase, corr_phase):
        recent = z[:, -min(self.dataset_period, z.shape[1]) :, :]
        vol = recent.std(dim=(1, 2)).clamp_min(1e-5)
        drift = (recent[:, -1, :] - recent[:, 0, :]).abs().mean(dim=1)
        trend_energy = trend_hist.square().mean(dim=(1, 2)).sqrt()
        seasonal_energy = seasonal_hist.square().mean(dim=(1, 2)).sqrt()
        spread = (recent.quantile(0.75, dim=1) - recent.quantile(0.25, dim=1)).abs().mean(dim=1)
        anchor_gap = (phase - z[:, -1:, :]).abs().mean(dim=(1, 2))
        corr_gap = (corr_phase - phase).abs().mean(dim=(1, 2))
        desc = torch.stack([vol, drift, trend_energy, seasonal_energy, spread, anchor_gap, corr_gap], dim=-1)
        return torch.nan_to_num(desc, nan=0.0, posinf=10.0, neginf=0.0)

    def _normalize_target(self, y, stats):
        mean, std = stats
        z = (y - mean) / std
        if self.revin.affine:
            z = z * self.revin.weight + self.revin.bias
        return z

    def forward_norm(self, x):
        z, stats = self.revin(x, "norm")
        trend_hist = moving_average(z, self.kernel_size)
        seasonal_hist = z - trend_hist
        trend = self.trend_head(trend_hist.transpose(1, 2)).transpose(1, 2)
        patch = self._patch_branch(seasonal_hist)
        direct = self._direct_branch(z)
        phase = _torch_phase_profile(z, self.pred_len, self.dataset_period, "mean")
        corr_phase = _torch_corr_phase(z, self.pred_len, self.dataset_period, eta=0.25)
        desc = self._descriptors(z, trend_hist, seasonal_hist, phase, corr_phase).detach()
        gate = torch.softmax(self.branch_gate(desc) + self.branch_prior, dim=-1)
        branches = torch.stack([trend, patch, direct, phase, corr_phase], dim=1)
        y = (branches * gate[:, :, None, None]).sum(dim=1)
        y = self._low_rank_mix(y)
        y = y + torch.sigmoid(self.refine_scale) * self.channel_refine(y)
        return torch.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0), stats

    def forward(self, x):
        y, stats = self.forward_norm(x)
        return self.revin(y, "denorm", stats)

    def training_step_loss(self, x, y, args):
        pred_norm, stats = self.forward_norm(x)
        target_norm = self._normalize_target(y, stats)
        pred = self.revin(pred_norm, "denorm", stats)
        base = loss_fn(pred, y, str(getattr(args, "loss", "huber")))
        norm_loss = torch.nn.functional.smooth_l1_loss(pred_norm, target_norm, beta=0.5)
        target_coarse = _coarse_mean(target_norm, int(getattr(args, "coarse_factor", 4)))
        pred_coarse = _coarse_mean(pred_norm, int(getattr(args, "coarse_factor", 4)))
        coarse = torch.nn.functional.smooth_l1_loss(pred_coarse, target_coarse, beta=0.5)
        beta = float(getattr(args, "coarse_loss_weight", 0.1))
        return pred, base + 0.2 * norm_loss + beta * coarse


def _torch_history_phase_profile(x, period):
    b, length, c = x.shape
    period = max(1, int(period))
    idx = torch.arange(length, device=x.device)
    out = torch.empty_like(x)
    for r in range(period):
        mask = (idx % period) == r
        vals = x[:, mask, :]
        fill = vals.mean(dim=1, keepdim=True) if bool(mask.any()) else x.mean(dim=1, keepdim=True)
        out[:, mask, :] = fill.expand(-1, int(mask.sum().item()), -1)
    return out


class PhaseResidualLowRankMixer(nn.Module):
    """Forecasts future residuals around a regular-grid phase anchor."""

    def __init__(
        self,
        input_len,
        pred_len,
        channels,
        d_model=128,
        depth=3,
        dropout=0.1,
        patch_len=16,
        stride=8,
        kernel_size=25,
        dataset_period=24,
        conv_kernel=3,
        low_rank=4,
    ):
        super().__init__()
        self.revin = RevIN(channels)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.kernel_size = max(3, int(kernel_size) | 1)
        self.dataset_period = max(1, int(dataset_period))
        self.patch_len = min(int(patch_len), self.input_len)
        self.stride = max(1, int(stride))
        self.n_patches = 1 + max(0, (self.input_len - self.patch_len) // self.stride)
        self.low_rank = max(1, min(int(low_rank), self.channels))

        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.patch_pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.patch_blocks = nn.ModuleList([_PatchConvBlock(d_model, dropout, conv_kernel) for _ in range(max(1, depth))])
        self.patch_head = nn.Sequential(
            nn.LayerNorm(self.n_patches * d_model),
            nn.Linear(self.n_patches * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len),
        )
        self.trend_head = nn.Linear(input_len, pred_len)
        self.direct_encoder = nn.Sequential(
            nn.LayerNorm(input_len),
            nn.Linear(input_len, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.horizon_emb = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.direct_decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.branch_logits = nn.Parameter(torch.tensor([0.8, 1.2, 1.0], dtype=torch.float32))
        self.mix_left = nn.Parameter(torch.randn(channels, self.low_rank) * 0.02)
        self.mix_right = nn.Parameter(torch.randn(channels, self.low_rank) * 0.02)
        self.mix_scale = nn.Parameter(torch.tensor(-2.2, dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _patch_branch(self, residual):
        b, _, c = residual.shape
        u = residual.transpose(1, 2).reshape(b * c, -1)
        patches = u.unfold(1, self.patch_len, self.stride)
        h = self.patch_proj(patches) + self.patch_pos
        for block in self.patch_blocks:
            h = block(h)
        return self.patch_head(h.reshape(b * c, -1)).reshape(b, c, -1).transpose(1, 2)

    def _direct_branch(self, residual):
        b, _, c = residual.shape
        ctx = self.direct_encoder(residual.transpose(1, 2).reshape(b * c, -1)).reshape(b, c, -1)
        h = ctx[:, :, None, :] + self.horizon_emb[:, None, :, :]
        return self.direct_decoder(h).squeeze(-1).transpose(1, 2)

    def _low_rank_mix(self, residual):
        mat = torch.matmul(self.mix_left, self.mix_right.transpose(0, 1)) / math.sqrt(float(self.low_rank))
        mat = torch.tanh(mat)
        return residual + torch.sigmoid(self.mix_scale) * torch.einsum("bhc,cd->bhd", residual, mat)

    def _normalize_target(self, y, stats):
        mean, std = stats
        z = (y - mean) / std
        if self.revin.affine:
            z = z * self.revin.weight + self.revin.bias
        return z

    def forward_norm(self, x):
        z, stats = self.revin(x, "norm")
        phase_hist = _torch_history_phase_profile(z, self.dataset_period)
        residual_hist = z - phase_hist
        trend_hist = moving_average(residual_hist, self.kernel_size)
        local_hist = residual_hist - trend_hist
        trend = self.trend_head(trend_hist.transpose(1, 2)).transpose(1, 2)
        patch = self._patch_branch(local_hist)
        direct = self._direct_branch(residual_hist)
        weights = torch.softmax(self.branch_logits, dim=0)
        residual = weights[0] * trend + weights[1] * patch + weights[2] * direct
        residual = self._low_rank_mix(residual)
        phase = _torch_phase_profile(z, self.pred_len, self.dataset_period, "mean")
        scale = torch.sigmoid(self.residual_scale)
        y = phase + scale * residual
        return torch.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0), stats

    def forward(self, x):
        y, stats = self.forward_norm(x)
        return self.revin(y, "denorm", stats)

    def training_step_loss(self, x, y, args):
        pred_norm, stats = self.forward_norm(x)
        target_norm = self._normalize_target(y, stats)
        pred = self.revin(pred_norm, "denorm", stats)
        base = loss_fn(pred, y, str(getattr(args, "loss", "huber")))
        norm_loss = torch.nn.functional.smooth_l1_loss(pred_norm, target_norm, beta=0.5)
        target_coarse = _coarse_mean(target_norm, int(getattr(args, "coarse_factor", 4)))
        pred_coarse = _coarse_mean(pred_norm, int(getattr(args, "coarse_factor", 4)))
        coarse = torch.nn.functional.smooth_l1_loss(pred_coarse, target_coarse, beta=0.5)
        beta = float(getattr(args, "coarse_loss_weight", 0.05))
        return pred, base + 0.2 * norm_loss + beta * coarse


def _torch_phase_profile(x, pred_len, period, reducer="mean"):
    b, length, c = x.shape
    outs = []
    idx = torch.arange(length, device=x.device)
    for h in range(pred_len):
        r = (length + h) % max(1, int(period))
        mask = (idx % max(1, int(period))) == r
        vals = x[:, mask, :] if bool(mask.any()) else x[:, -min(period, length) :, :]
        if reducer == "median":
            outs.append(vals.median(dim=1).values)
        else:
            outs.append(vals.mean(dim=1))
    return torch.stack(outs, dim=1)


def _torch_corr_phase(x, pred_len, period, eta=0.25):
    phase = _torch_phase_profile(x, pred_len, period, "mean")
    recent = x[:, -min(period, x.shape[1]) :, :]
    centered = recent - recent.mean(dim=1, keepdim=True)
    norm = centered.square().sum(dim=1, keepdim=True).clamp_min(1e-10).sqrt()
    u = centered / norm
    corr = torch.bmm(u.transpose(1, 2), u).clamp_min(0.0)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    eye = torch.eye(corr.shape[-1], device=x.device, dtype=x.dtype).unsqueeze(0)
    corr = corr * (1.0 - eye)
    corr = torch.nan_to_num(corr / corr.sum(dim=-1, keepdim=True).clamp_min(1e-5), nan=0.0, posinf=0.0, neginf=0.0)
    mixed = torch.bmm(phase.reshape(-1, phase.shape[-1]).unsqueeze(1), corr.repeat_interleave(pred_len, dim=0).transpose(1, 2))
    mixed = mixed.squeeze(1).reshape(x.shape[0], pred_len, x.shape[-1])
    return (1.0 - eta) * phase + eta * mixed


def _torch_structural_future(x, pred_len, period, name):
    phase = _torch_phase_profile(x, pred_len, period, "mean")
    if name == "phase":
        return phase
    if name == "robust_phase":
        return _torch_phase_profile(x, pred_len, period, "median")
    recent = x[:, -min(period, x.shape[1]) :, :]
    mu_ema = recent.mean(dim=1, keepdim=True)
    mu_phase = phase.mean(dim=1, keepdim=True)
    sigma = recent.std(dim=1, keepdim=True).clamp_min(1e-5)
    if name == "shrink_phase":
        return mu_ema + 0.7 * (phase - mu_phase)
    if name == "winsor_phase":
        return mu_ema + (phase - mu_phase).clamp(-1.5 * sigma, 1.5 * sigma)
    if name == "corr_phase":
        return _torch_corr_phase(x, pred_len, period)
    raise ValueError(name)


def action_compatible_loss(pred, y, x, args):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
    y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
    base = torch.nan_to_num(loss_fn(pred, y, args.loss), nan=1e4, posinf=1e4, neginf=1e4)
    service_weight = float(getattr(args, "ac_service_weight", 0.0))
    if service_weight <= 0.0:
        return base
    period = int(getattr(args, "dataset_period", 24))
    structures = [s for s in str(getattr(args, "structures", "phase,robust_phase,winsor_phase,shrink_phase,corr_phase")).split(",") if s]
    residual = y - pred
    candidate_losses = [torch.mean(residual.square(), dim=(1, 2))]
    dirs = []
    for structure in structures:
        future = _torch_structural_future(x, pred.shape[1], period, structure)
        direction = future - pred
        lam = (residual * direction).sum(dim=(1, 2)) / direction.square().sum(dim=(1, 2)).clamp_min(1e-6)
        lam = lam.detach().clamp(0.0, float(getattr(args, "lambda_max", 1.0)))
        corrected = pred + lam[:, None, None] * direction
        candidate_losses.append(torch.mean((corrected - y).square(), dim=(1, 2)))
        dirs.append(direction)
    losses = torch.nan_to_num(torch.stack(candidate_losses, dim=-1), nan=1e4, posinf=1e4, neginf=1e4).clamp_max(1e4)
    temp = max(1e-4, float(getattr(args, "ac_softmin_temp", 0.05)))
    weights = torch.softmax(-losses / temp, dim=-1)
    served = (weights * losses).sum(dim=-1).mean()
    coh = pred.new_tensor(0.0)
    if len(dirs) > 1 and float(getattr(args, "ac_coherence_weight", 0.0)) > 0:
        flat = [d.reshape(d.shape[0], -1) for d in dirs]
        pairs = []
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                num = (flat[i] * flat[j]).sum(dim=1).square()
                den = flat[i].square().sum(dim=1) * flat[j].square().sum(dim=1) + 1e-6
                pairs.append((num / den).mean())
        coh = torch.stack(pairs).mean() if pairs else coh
    smooth = pred[:, 1:, :].sub(pred[:, :-1, :]).abs().mean() if pred.shape[1] > 1 else pred.new_tensor(0.0)
    total = (
        base
        + service_weight * served
        + float(getattr(args, "ac_coherence_weight", 0.0)) * coh
        + float(getattr(args, "ac_smooth_weight", 0.0)) * smooth
    )
    if not torch.isfinite(total).all():
        return base
    return total


_THIRD_PARTY_ROOT = Path(__file__).resolve().parent / "third_party"


def _load_model_class(module_path: Path, module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name].Model
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official model from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Model


class OfficialFACT(nn.Module):
    def __init__(self, args, input_len, pred_len, channels):
        super().__init__()
        model_cls = _load_model_class(
            _THIRD_PARTY_ROOT / "fact_official" / "models" / "FACT.py",
            "_official_fact_model",
        )
        d_model = int(getattr(args, "d_model", 128))
        fact_d_ff = int(getattr(args, "fact_d_ff", 0))
        if fact_d_ff <= 0:
            fact_d_ff = max(64, d_model)
        cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=input_len,
            pred_len=pred_len,
            enc_in=channels,
            use_norm=bool(getattr(args, "fact_use_norm", True)),
            freq=str(getattr(args, "fact_freq", "x")),
            d_model=d_model,
            dilation=list(getattr(args, "fact_dilation", [1, 2, 1])),
            num_kernels=int(getattr(args, "fact_num_kernels", 4)),
            core=float(getattr(args, "fact_core", 0.5)),
            d_ff=fact_d_ff,
            dropout=float(getattr(args, "dropout", 0.1)),
        )
        self.model = model_cls(cfg)

    def forward(self, x):
        return self.model(x, None, None, None)


class OfficialXLinear(nn.Module):
    def __init__(self, args, input_len, pred_len, channels):
        super().__init__()
        model_cls = _load_model_class(
            _THIRD_PARTY_ROOT / "xlinear_official" / "models" / "XLinear.py",
            "_official_xlinear_model",
        )
        d_model = int(getattr(args, "d_model", 256))
        t_ff = int(getattr(args, "xlinear_t_ff", 0))
        c_ff = int(getattr(args, "xlinear_c_ff", 0))
        head_dropout = float(getattr(args, "xlinear_head_dropout", -1.0))
        t_dropout = float(getattr(args, "xlinear_t_dropout", -1.0))
        if t_ff <= 0:
            t_ff = max(64, d_model // 2)
        if c_ff <= 0:
            c_ff = channels
        if head_dropout < 0:
            head_dropout = float(getattr(args, "dropout", 0.1))
        if t_dropout < 0:
            t_dropout = float(getattr(args, "dropout", 0.1))
        cfg = SimpleNamespace(
            seq_len=input_len,
            pred_len=pred_len,
            d_model=d_model,
            enc_in=channels,
            t_ff=t_ff,
            c_ff=c_ff,
            usenorm=bool(getattr(args, "xlinear_usenorm", True)),
            embed_dropout=float(getattr(args, "xlinear_embed_dropout", 0.0)),
            head_dropout=head_dropout,
            t_dropout=t_dropout,
            c_dropout=float(getattr(args, "xlinear_c_dropout", 0.0)),
            features=str(getattr(args, "xlinear_features", "M")),
        )
        self.model = model_cls(cfg)

    def forward(self, x):
        return self.model(x)


class OfficialGFMixer(nn.Module):
    def __init__(self, args, input_len, pred_len, channels):
        super().__init__()
        root = _THIRD_PARTY_ROOT / "gfmixer_official"
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        model_cls = _load_model_class(root / "models" / "GFMixer.py", "_official_gfmixer_model")
        dataset_period = int(getattr(args, "dataset_period", 24))
        period = getattr(args, "gf_period", None) or [dataset_period]
        patch_len = getattr(args, "gf_patch_len", None) or [1 for _ in period]
        stride = getattr(args, "gf_stride", None) or [1 for _ in period]
        if len(patch_len) == 1 and len(period) > 1:
            patch_len = patch_len * len(period)
        if len(stride) == 1 and len(period) > 1:
            stride = stride * len(period)
        d_model = int(getattr(args, "d_model", 32))
        fc_dropout = float(getattr(args, "gf_fc_dropout", -1.0))
        if fc_dropout < 0:
            fc_dropout = float(getattr(args, "dropout", 0.1))
        cfg = SimpleNamespace(
            enc_in=channels,
            seq_len=input_len,
            pred_len=pred_len,
            e_layers=int(getattr(args, "gf_e_layers", max(1, getattr(args, "depth", 2)))),
            n_heads=int(getattr(args, "gf_n_heads", 4)),
            d_model=d_model,
            d_ff=int(getattr(args, "gf_d_ff", max(64, d_model * 4))),
            dropout=float(getattr(args, "dropout", 0.1)),
            fc_dropout=fc_dropout,
            head_dropout=float(getattr(args, "gf_head_dropout", 0.0)),
            individual=bool(getattr(args, "individual", False)),
            add=bool(getattr(args, "gf_add", False)),
            wo_conv=bool(getattr(args, "gf_wo_conv", False)),
            serial_conv=bool(getattr(args, "gf_serial_conv", False)),
            kernel_list=list(getattr(args, "gf_kernel_list", [3, 7, 11])),
            patch_len=list(patch_len),
            period=list(period),
            stride=list(stride),
            padding_patch=str(getattr(args, "gf_padding_patch", "end")),
            revin=bool(getattr(args, "gf_revin", True)),
            affine=bool(getattr(args, "gf_affine", True)),
            subtract_last=bool(getattr(args, "gf_subtract_last", False)),
            num_kernels=int(getattr(args, "gf_num_kernels", 6)),
            batch_size=int(getattr(args, "batch_size", 128)),
            use_FAT=bool(getattr(args, "gf_use_fat", True)),
            TGB=int(getattr(args, "gf_tgb", 0)),
        )
        self.model = model_cls(cfg)

    def forward(self, x):
        return self.model(x)


class OfficialPhaseFormer(nn.Module):
    def __init__(self, args, input_len, pred_len, channels):
        super().__init__()
        root = _THIRD_PARTY_ROOT / "PhaseFormer_TSL"
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        model_cls = _load_model_class(root / "models" / "PhaseFormer.py", "_official_phaseformer_model")
        period = int(getattr(args, "phaseformer_period_len", 0))
        if period <= 0:
            period = int(getattr(args, "dataset_period", 24))
        latent_dim = int(getattr(args, "phaseformer_latent_dim", getattr(args, "d_model", 32)))
        cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=input_len,
            pred_len=pred_len,
            enc_in=channels,
            period_len=period,
            latent_dim=latent_dim,
            phase_encoder_hidden=int(getattr(args, "phaseformer_encoder_hidden", max(32, latent_dim * 2))),
            predictor_hidden=int(getattr(args, "phaseformer_predictor_hidden", max(64, latent_dim * 4))),
            phase_layers=int(getattr(args, "phaseformer_layers", max(1, getattr(args, "depth", 1)))),
            phase_attn_heads=int(getattr(args, "phaseformer_heads", 4)),
            phase_attn_dropout=float(getattr(args, "dropout", 0.1)),
            phase_attn_use_relpos=bool(getattr(args, "phaseformer_use_relpos", True)),
            phase_attn_window=getattr(args, "phaseformer_attn_window", None),
            phase_attention_dim=getattr(args, "phaseformer_attention_dim", None),
            phase_num_routers=int(getattr(args, "phaseformer_routers", 8)),
            phase_use_pos_embed=bool(getattr(args, "phaseformer_use_pos_embed", True)),
            phase_pos_dropout=float(getattr(args, "phaseformer_pos_dropout", 0.0)),
            use_revin=bool(getattr(args, "phaseformer_use_revin", True)),
            revin_affine=bool(getattr(args, "phaseformer_revin_affine", False)),
            revin_eps=float(getattr(args, "phaseformer_revin_eps", 1e-5)),
        )
        self.model = model_cls(cfg)

    def forward(self, x):
        return self.model(x, None, None, None)


class OfficialDTAF(nn.Module):
    def __init__(self, args, input_len, pred_len, channels):
        super().__init__()
        root = _THIRD_PARTY_ROOT / "DTAF"
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        from ts_benchmark.baselines.dtaf.model.DTAF_model import DTAF as DTAFModel

        d_model = int(getattr(args, "d_model", 32))
        cfg = SimpleNamespace(
            seq_len=input_len,
            pred_len=pred_len,
            enc_in=channels,
            dec_in=channels,
            c_out=channels,
            d_model=d_model,
            moving_avg=int(getattr(args, "dtaf_moving_avg", 25)),
            e_layers=int(getattr(args, "dtaf_layers", max(1, getattr(args, "depth", 1)))),
            patch_len=int(getattr(args, "dtaf_patch_len", getattr(args, "patch_len", 16))),
            stride=int(getattr(args, "dtaf_stride", getattr(args, "patch_stride", 8))),
            dropout=float(getattr(args, "dropout", 0.1)),
            heads=int(getattr(args, "dtaf_heads", 2)),
            expert_num=int(getattr(args, "dtaf_expert_num", 2)),
            kan_div=int(getattr(args, "dtaf_kan_div", 4)),
            aggregated_norm=int(getattr(args, "dtaf_aggregated_norm", 1)),
            k=int(getattr(args, "dtaf_top_freq", 1)),
        )
        self.model = DTAFModel(cfg)

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def build_model(args, channels, pred_len):
    if args.model == "nlinear":
        return NLinear(args.input_len, pred_len, channels, individual=args.individual)
    if args.model == "dlinear":
        return DLinear(args.input_len, pred_len, channels, kernel_size=args.kernel_size, individual=args.individual)
    if args.model == "patchmixer":
        return PatchMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            depth=args.depth,
            dropout=args.dropout,
        )
    if args.model == "tsmixer":
        return TSMixer(args.input_len, pred_len, channels, d_model=args.d_model, depth=args.depth, dropout=args.dropout)
    if args.model == "actsmixer":
        return TSMixer(args.input_len, pred_len, channels, d_model=args.d_model, depth=args.depth, dropout=args.dropout)
    if args.model == "acmixer":
        patch_sizes = getattr(args, "ac_patch_sizes", [16, 32, 64])
        return ActionCompatibleMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            patch_sizes=patch_sizes,
            patch_stride=args.patch_stride,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
        )
    if args.model == "scmixer":
        patch_sizes = getattr(args, "ac_patch_sizes", [8, 16, 32, 64])
        return StructureConditionedMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            patch_sizes=patch_sizes,
            patch_stride=args.patch_stride,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
        )
    if args.model == "sdmixer":
        return SeasonalTrendMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            kernel_size=args.kernel_size,
        )
    if args.model == "pdmixer":
        return ProbabilisticDecomposedMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
            support_size=args.dist_support_size,
            support_scale=args.dist_support_scale,
        )
    if args.model == "ddmixer":
        return DirectDistributionMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
            support_size=args.dist_support_size,
            support_scale=args.dist_support_scale,
        )
    if args.model == "hdmixer":
        return HybridDistributionMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
            support_size=args.dist_support_size,
            support_scale=args.dist_support_scale,
        )
    if args.model == "pcrmixer":
        return PatchConvResidualMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
            conv_kernel=args.pcr_conv_kernel,
        )
    if args.model == "pamixer":
        return PhaseAnchoredPCRMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
            conv_kernel=args.pcr_conv_kernel,
            phase_alpha_init=args.phase_alpha_init,
            phase_alpha_max=args.phase_alpha_max,
        )
    if args.model == "cfpcrmixer":
        return CoarseFinePCRMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
            conv_kernel=args.pcr_conv_kernel,
            coarse_factor=args.coarse_factor,
        )
    if args.model == "pacfmixer":
        return PhaseAnchoredCoarseFineMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
            conv_kernel=args.pcr_conv_kernel,
            phase_alpha_init=args.phase_alpha_init,
            phase_alpha_max=args.phase_alpha_max,
            coarse_factor=args.coarse_factor,
        )
    if args.model == "pahdmixer":
        return PhaseAnchoredHybridDistributionMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
            support_size=args.dist_support_size,
            support_scale=args.dist_support_scale,
            phase_alpha_init=args.phase_alpha_init,
            phase_alpha_max=args.phase_alpha_max,
        )
    if args.model == "sarmixer":
        return SeasonalAnchorLowRankMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
            conv_kernel=args.pcr_conv_kernel,
            low_rank=args.low_rank,
        )
    if args.model == "prmixer":
        return PhaseResidualLowRankMixer(
            args.input_len,
            pred_len,
            channels,
            d_model=args.d_model,
            depth=args.depth,
            dropout=args.dropout,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            kernel_size=args.kernel_size,
            dataset_period=getattr(args, "dataset_period", 24),
            conv_kernel=args.pcr_conv_kernel,
            low_rank=args.low_rank,
        )
    if args.model == "fact":
        return OfficialFACT(args, args.input_len, pred_len, channels)
    if args.model == "xlinear":
        return OfficialXLinear(args, args.input_len, pred_len, channels)
    if args.model == "gfmixer":
        return OfficialGFMixer(args, args.input_len, pred_len, channels)
    if args.model == "phaseformer":
        return OfficialPhaseFormer(args, args.input_len, pred_len, channels)
    if args.model == "dtaf":
        return OfficialDTAF(args, args.input_len, pred_len, channels)
    raise ValueError(args.model)


def loss_fn(pred, y, kind):
    if kind.startswith("decay_"):
        base_kind = kind.split("_", 1)[1]
        steps = torch.arange(pred.shape[1], device=pred.device, dtype=pred.dtype)
        weights = torch.exp(-math.log(2.0) * steps / max(1, pred.shape[1] - 1))
        weights = weights / weights.mean().clamp_min(1e-6)
        weights = weights.view(1, -1, 1)
        if base_kind == "mse":
            return ((pred - y).square() * weights).mean()
        if base_kind == "mae":
            return ((pred - y).abs() * weights).mean()
        return (torch.nn.functional.smooth_l1_loss(pred, y, beta=0.5, reduction="none") * weights).mean()
    if kind == "mse":
        return torch.nn.functional.mse_loss(pred, y)
    if kind == "mae":
        return torch.nn.functional.l1_loss(pred, y)
    return torch.nn.functional.smooth_l1_loss(pred, y, beta=0.5)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    se = 0.0
    ae = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        se += float((pred - yb).square().sum().cpu())
        ae += float((pred - yb).abs().sum().cpu())
        n += pred.numel()
    return {"mse": se / n, "mae": ae / n}


def train_one(bundle, pred_len, args):
    train_ds = SlidingWindowDataset(bundle.train, args.input_len, pred_len, args.max_train_windows, args.stride)
    val_ds = SlidingWindowDataset(bundle.val, args.input_len, pred_len, args.max_val_windows, args.eval_stride)
    test_ds = SlidingWindowDataset(bundle.test, args.input_len, pred_len, args.max_test_windows, args.eval_stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = build_model(args, bundle.train.shape[-1], pred_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_state = None
    wait = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if hasattr(model, "training_step_loss"):
                pred, loss = model.training_step_loss(xb, yb, args)
            else:
                pred = model(xb)
                loss = action_compatible_loss(pred, yb, xb, args) if args.model in {"acmixer", "actsmixer"} else loss_fn(pred, yb, args.loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = evaluate(model, val_loader, device)
        if str(getattr(args, "val_select_metric", "mse")) == "mse_mae":
            score = val["mse"] + float(getattr(args, "val_mae_weight", 0.5)) * val["mae"]
        else:
            score = val["mse"]
        history.append(val["mse"])
        print(f"epoch={epoch+1} val_mse={val['mse']:.4f} val_mae={val['mae']:.4f} score={score:.4f}", flush=True)
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    return {
        "dataset": bundle.name,
        "pred_len": pred_len,
        "model": args.model,
        "input_len": args.input_len,
        "mse": test["mse"],
        "mae": test["mae"],
        "best_val_mse": best,
        "epochs_run": len(history),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("experiments/data"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/results_backbone"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["ETTh1"])
    ap.add_argument("--pred-lens", nargs="+", type=int, default=[96])
    ap.add_argument("--model", choices=["nlinear", "dlinear", "patchmixer", "tsmixer", "actsmixer", "acmixer", "scmixer", "sdmixer", "pdmixer", "ddmixer", "hdmixer", "pcrmixer", "pamixer", "cfpcrmixer", "pacfmixer", "pahdmixer", "sarmixer", "prmixer", "fact", "xlinear", "gfmixer", "phaseformer", "dtaf"], default="patchmixer")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--max-train-windows", type=int, default=0)
    ap.add_argument("--max-val-windows", type=int, default=0)
    ap.add_argument("--max-test-windows", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patch-len", type=int, default=16)
    ap.add_argument("--patch-stride", type=int, default=8)
    ap.add_argument("--pcr-conv-kernel", type=int, default=3)
    ap.add_argument("--phase-alpha-init", type=float, default=0.4)
    ap.add_argument("--phase-alpha-max", type=float, default=0.8)
    ap.add_argument("--coarse-factor", type=int, default=4)
    ap.add_argument("--coarse-loss-weight", type=float, default=0.2)
    ap.add_argument("--coarse-consistency-weight", type=float, default=0.1)
    ap.add_argument("--low-rank", type=int, default=4)
    ap.add_argument("--ac-patch-sizes", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--ac-service-weight", type=float, default=0.25)
    ap.add_argument("--ac-coherence-weight", type=float, default=0.01)
    ap.add_argument("--ac-smooth-weight", type=float, default=0.001)
    ap.add_argument("--ac-softmin-temp", type=float, default=0.05)
    ap.add_argument("--structures", default="phase,robust_phase,winsor_phase,shrink_phase,corr_phase")
    ap.add_argument("--lambda-max", type=float, default=1.0)
    ap.add_argument("--dist-support-size", type=int, default=64)
    ap.add_argument("--dist-support-scale", type=float, default=1.25)
    ap.add_argument("--dist-ce-weight", type=float, default=0.05)
    ap.add_argument("--dist-anchor-weight", type=float, default=0.2)
    ap.add_argument("--dist-prior-weight", type=float, default=0.35)
    ap.add_argument("--dist-prior-width", type=float, default=0.75)
    ap.add_argument("--dist-consistency-weight", type=float, default=0.05)
    ap.add_argument("--dist-max-mix", type=float, default=0.35)
    ap.add_argument("--dist-mix-penalty", type=float, default=0.01)
    ap.add_argument("--kernel-size", type=int, default=25)
    ap.add_argument("--individual", action="store_true")
    ap.add_argument("--fact-core", type=float, default=0.5)
    ap.add_argument("--fact-d-ff", type=int, default=0)
    ap.add_argument("--fact-dilation", nargs="+", type=int, default=[1, 2, 1])
    ap.add_argument("--fact-num-kernels", type=int, default=4)
    ap.add_argument("--fact-freq", default="x")
    ap.add_argument("--fact-no-norm", dest="fact_use_norm", action="store_false")
    ap.set_defaults(fact_use_norm=True)
    ap.add_argument("--xlinear-t-ff", type=int, default=0)
    ap.add_argument("--xlinear-c-ff", type=int, default=0)
    ap.add_argument("--xlinear-features", default="M")
    ap.add_argument("--xlinear-no-norm", dest="xlinear_usenorm", action="store_false")
    ap.set_defaults(xlinear_usenorm=True)
    ap.add_argument("--xlinear-embed-dropout", type=float, default=0.0)
    ap.add_argument("--xlinear-head-dropout", type=float, default=-1.0)
    ap.add_argument("--xlinear-t-dropout", type=float, default=-1.0)
    ap.add_argument("--xlinear-c-dropout", type=float, default=0.0)
    ap.add_argument("--gf-e-layers", type=int, default=2)
    ap.add_argument("--gf-n-heads", type=int, default=4)
    ap.add_argument("--gf-d-ff", type=int, default=128)
    ap.add_argument("--gf-fc-dropout", type=float, default=-1.0)
    ap.add_argument("--gf-head-dropout", type=float, default=0.0)
    ap.add_argument("--gf-kernel-list", nargs="+", type=int, default=[3, 7, 11])
    ap.add_argument("--gf-period", nargs="+", type=int, default=None)
    ap.add_argument("--gf-patch-len", nargs="+", type=int, default=None)
    ap.add_argument("--gf-stride", nargs="+", type=int, default=None)
    ap.add_argument("--gf-num-kernels", type=int, default=6)
    ap.add_argument("--gf-add", action="store_true")
    ap.add_argument("--gf-wo-conv", action="store_true")
    ap.add_argument("--gf-serial-conv", action="store_true")
    ap.add_argument("--gf-padding-patch", default="end")
    ap.add_argument("--gf-no-revin", dest="gf_revin", action="store_false")
    ap.add_argument("--gf-no-affine", dest="gf_affine", action="store_false")
    ap.add_argument("--gf-subtract-last", action="store_true")
    ap.add_argument("--gf-no-fat", dest="gf_use_fat", action="store_false")
    ap.add_argument("--gf-tgb", type=int, default=0)
    ap.set_defaults(gf_revin=True, gf_affine=True, gf_use_fat=True)
    ap.add_argument("--phaseformer-period-len", type=int, default=0)
    ap.add_argument("--phaseformer-latent-dim", type=int, default=32)
    ap.add_argument("--phaseformer-encoder-hidden", type=int, default=64)
    ap.add_argument("--phaseformer-predictor-hidden", type=int, default=128)
    ap.add_argument("--phaseformer-layers", type=int, default=1)
    ap.add_argument("--phaseformer-heads", type=int, default=4)
    ap.add_argument("--phaseformer-routers", type=int, default=8)
    ap.add_argument("--phaseformer-no-relpos", dest="phaseformer_use_relpos", action="store_false")
    ap.add_argument("--phaseformer-no-pos-embed", dest="phaseformer_use_pos_embed", action="store_false")
    ap.add_argument("--phaseformer-pos-dropout", type=float, default=0.0)
    ap.add_argument("--phaseformer-no-revin", dest="phaseformer_use_revin", action="store_false")
    ap.add_argument("--phaseformer-revin-affine", action="store_true")
    ap.add_argument("--phaseformer-revin-eps", type=float, default=1e-5)
    ap.set_defaults(phaseformer_use_relpos=True, phaseformer_use_pos_embed=True, phaseformer_use_revin=True)
    ap.add_argument("--dtaf-layers", type=int, default=1)
    ap.add_argument("--dtaf-patch-len", type=int, default=16)
    ap.add_argument("--dtaf-stride", type=int, default=8)
    ap.add_argument("--dtaf-heads", type=int, default=2)
    ap.add_argument("--dtaf-expert-num", type=int, default=2)
    ap.add_argument("--dtaf-kan-div", type=int, default=4)
    ap.add_argument("--dtaf-aggregated-norm", type=int, default=1)
    ap.add_argument("--dtaf-top-freq", type=int, default=1)
    ap.add_argument("--dtaf-moving-avg", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["mse", "mae", "huber", "decay_mse", "decay_mae", "decay_huber"], default="mse")
    ap.add_argument("--val-select-metric", choices=["mse", "mse_mae"], default="mse")
    ap.add_argument("--val-mae-weight", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.download:
        download_datasets(args.data_dir)
    rows = []
    start = time.time()
    for ds in args.datasets:
        bundle = load_dataset(ds, args.data_dir)
        args.dataset_period = 96 if ds.startswith("ETTm") else 24
        for h in args.pred_lens:
            print(f"[backbone] dataset={ds} pred_len={h} model={args.model} input_len={args.input_len}", flush=True)
            res = train_one(bundle, h, args)
            print(res, flush=True)
            rows.append(res)
            pd.DataFrame(rows).to_csv(args.out_dir / "metrics.csv", index=False)
            (args.out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"done in {(time.time()-start)/60:.1f} min -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
