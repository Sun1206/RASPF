from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_backbone_forecaster import (
    SlidingWindowDataset,
    action_compatible_loss,
    build_model,
    load_dataset,
    loss_fn,
)


FIELDS = [
    "dataset",
    "horizon",
    "seed",
    "backbone",
    "variant",
    "structure",
    "lambda_mode",
    "lambda_value",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "base_val_mse",
    "base_val_mae",
    "base_test_mse",
    "base_test_mae",
    "selected_variant",
    "selected_structure",
    "epochs_run",
    "batch_size",
    "input_len",
    "train_eval_seconds",
    "calibration_seconds",
    "cuda_peak_memory_mb",
    "candidate_count",
]

GATE_VARIANT = "RASPF-structural-gate"
MIDDLE_HORIZON_DRIFT_STRUCTURES = {"winsor_phase", "shrink_phase"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(args) -> None:
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.set_num_interop_threads(max(1, int(args.torch_interop_threads)))
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.backends.cudnn.benchmark = bool(args.cuda_benchmark)
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def metric_np(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    err = pred - true
    return float(np.mean(err * err)), float(np.mean(np.abs(err)))


def metric_t(pred: torch.Tensor, true: torch.Tensor) -> tuple[float, float]:
    err = pred - true
    return float(err.square().mean().detach().cpu()), float(err.abs().mean().detach().cpu())


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def period_for(dataset: str) -> int:
    if dataset.startswith("ETTm"):
        return 96
    return 24


def phase_profile(inputs: np.ndarray, period: int, reducer: str = "mean") -> np.ndarray:
    n, length, channels = inputs.shape
    pred_len = phase_profile.pred_len
    out = np.empty((n, pred_len, channels), dtype=np.float32)
    idx = np.arange(length)
    for h in range(pred_len):
        r = (length + h) % period
        mask = (idx % period) == r
        if not np.any(mask):
            vals = inputs[:, -min(period, length) :, :]
        else:
            vals = inputs[:, mask, :]
        if reducer == "median":
            out[:, h, :] = np.median(vals, axis=1)
        else:
            out[:, h, :] = np.mean(vals, axis=1)
    return out


def corr_smooth_phase(inputs: np.ndarray, period: int, eta: float = 0.25) -> np.ndarray:
    base = phase_profile(inputs, period, "mean")
    n, _, channels = inputs.shape
    recent = inputs[:, -min(period, inputs.shape[1]) :, :]
    out = base.copy()
    for i in range(n):
        x = recent[i]
        if channels <= 1:
            continue
        corr = np.corrcoef(x.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(corr, 0.0)
        np.fill_diagonal(w, 0.0)
        denom = w.sum(axis=1, keepdims=True)
        w = np.divide(w, denom, out=np.zeros_like(w), where=denom > 1e-8)
        out[i] = (1.0 - eta) * base[i] + eta * (base[i] @ w.T)
    return out


def structural_future(inputs: np.ndarray, name: str, period: int) -> np.ndarray:
    phase = phase_profile(inputs, period, "mean")
    if name == "phase":
        return phase
    if name == "robust_phase":
        return phase_profile(inputs, period, "median")
    recent = inputs[:, -min(period, inputs.shape[1]) :, :]
    mu_ema = recent.mean(axis=1, keepdims=True)
    mu_phase = phase.mean(axis=1, keepdims=True)
    sigma = recent.std(axis=1, keepdims=True) + 1e-6
    if name == "shrink_phase":
        return mu_ema + 0.7 * (phase - mu_phase)
    if name == "winsor_phase":
        return mu_ema + np.clip(phase - mu_phase, -1.5 * sigma, 1.5 * sigma)
    if name == "corr_phase":
        return corr_smooth_phase(inputs, period)
    raise ValueError(name)


def global_block_prediction(
    calib: dict,
    target: dict,
    structure: str,
    period: int,
    block_size: int,
    lambda_max: float,
) -> tuple[np.ndarray, float]:
    calib_struct = structural_future(calib["inputs"], structure, period)
    target_struct = structural_future(target["inputs"], structure, period)
    residual = calib["true"] - calib["pred"]
    direction = calib_struct - calib["pred"]
    target_dir = target_struct - target["pred"]
    pred = target["pred"].copy()
    lambdas = []
    for start in range(0, pred.shape[1], block_size):
        end = min(pred.shape[1], start + block_size)
        num = float(np.sum(residual[:, start:end, :] * direction[:, start:end, :]))
        den = float(np.sum(direction[:, start:end, :] ** 2)) + 1e-12
        lam = max(0.0, min(lambda_max, num / den))
        pred[:, start:end, :] = target["pred"][:, start:end, :] + lam * target_dir[:, start:end, :]
        lambdas.append(lam)
    return pred.astype(np.float32), float(np.mean(lambdas))


def fixed_prediction(target: dict, structure: str, period: int, weight: float) -> np.ndarray:
    struct = structural_future(target["inputs"], structure, period)
    return (target["pred"] + weight * (struct - target["pred"])).astype(np.float32)


def residual_mean_recenter(calib: dict, target: dict) -> np.ndarray:
    correction = (calib["true"] - calib["pred"]).mean(axis=0, keepdims=True)
    return (target["pred"] + correction).astype(np.float32)


def residual_median_recenter(calib: dict, target: dict) -> np.ndarray:
    correction = np.median(calib["true"] - calib["pred"], axis=0, keepdims=True)
    return (target["pred"] + correction).astype(np.float32)


def residual_ewma_recenter(calib: dict, target: dict, decay: float = 0.97) -> np.ndarray:
    residual = calib["true"] - calib["pred"]
    weights = decay ** np.arange(residual.shape[0] - 1, -1, -1, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    correction = np.sum(residual * weights[:, None, None], axis=0, keepdims=True)
    return (target["pred"] + correction).astype(np.float32)


def rolling_residual_recenter(calib: dict, target: dict, window: int = 64) -> np.ndarray:
    residual = calib["true"] - calib["pred"]
    window = max(1, min(int(window), residual.shape[0]))
    correction = residual[-window:].mean(axis=0, keepdims=True)
    return (target["pred"] + correction).astype(np.float32)


def seasonal_residual_recenter(calib: dict, target: dict, period: int) -> np.ndarray:
    residual = calib["true"] - calib["pred"]
    horizon = residual.shape[1]
    correction = np.zeros((1, horizon, residual.shape[2]), dtype=np.float32)
    phases = np.array([(calib["inputs"].shape[1] + h) % period for h in range(horizon)])
    for h in range(horizon):
        same = phases == phases[h]
        correction[:, h, :] = residual[:, same, :].mean(axis=(0, 1), keepdims=False)
    return (target["pred"] + correction).astype(np.float32)


def residual_features(inputs: np.ndarray, period: int) -> np.ndarray:
    recent = inputs[:, -min(period, inputs.shape[1]) :, :]
    last = inputs[:, -1, :]
    mean = recent.mean(axis=1)
    std = recent.std(axis=1)
    trend = inputs[:, -1, :] - inputs[:, -min(period, inputs.shape[1]), :]
    q25 = np.quantile(recent, 0.25, axis=1)
    q75 = np.quantile(recent, 0.75, axis=1)
    feats = np.concatenate([last, mean, std, trend, q25, q75], axis=1)
    bias = np.ones((inputs.shape[0], 1), dtype=np.float32)
    return np.concatenate([bias, feats.astype(np.float32)], axis=1)


def ridge_residual_recenter(calib: dict, target: dict, period: int, alpha: float = 10.0) -> np.ndarray:
    x = residual_features(calib["inputs"], period)
    y = (calib["true"] - calib["pred"]).reshape(calib["inputs"].shape[0], -1)
    xtx = x.T @ x
    penalty = alpha * np.eye(xtx.shape[0], dtype=np.float32)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(xtx + penalty, x.T @ y)
    correction = residual_features(target["inputs"], period) @ coef
    correction = correction.reshape(target["pred"].shape)
    return (target["pred"] + correction).astype(np.float32)


def local_analog_residual_recenter(
    calib: dict,
    target: dict,
    period: int,
    k: int = 32,
    chunk_size: int = 128,
) -> np.ndarray:
    x_cal = residual_features(calib["inputs"], period)[:, 1:].astype(np.float32, copy=False)
    x_tgt = residual_features(target["inputs"], period)[:, 1:].astype(np.float32, copy=False)
    center = x_cal.mean(axis=0, keepdims=True)
    scale = x_cal.std(axis=0, keepdims=True) + 1e-6
    x_cal = (x_cal - center) / scale
    x_tgt = (x_tgt - center) / scale
    residual = (calib["true"] - calib["pred"]).astype(np.float32, copy=False)
    out = target["pred"].astype(np.float32, copy=True)
    k = max(1, min(int(k), x_cal.shape[0]))
    cal_norm = np.sum(x_cal * x_cal, axis=1, keepdims=True).T
    chunk_size = max(1, int(chunk_size))
    for start in range(0, x_tgt.shape[0], chunk_size):
        end = min(x_tgt.shape[0], start + chunk_size)
        chunk = x_tgt[start:end]
        dist = np.sum(chunk * chunk, axis=1, keepdims=True) + cal_norm - 2.0 * (chunk @ x_cal.T)
        idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        out[start:end] += residual[idx].mean(axis=1)
        del chunk, dist, idx
    return out.astype(np.float32, copy=False)


def structural_candidate_forecasts(split: dict, structures: list[str], period: int) -> list[np.ndarray]:
    forecasts = [split["pred"]]
    for structure in structures:
        forecasts.append(structural_future(split["inputs"], structure, period))
    return forecasts


def global_ridge_stack(
    calib: dict,
    target: dict,
    structures: list[str],
    period: int,
    alpha: float = 1.0,
) -> np.ndarray:
    calib_forecasts = structural_candidate_forecasts(calib, structures, period)
    target_forecasts = structural_candidate_forecasts(target, structures, period)
    x = np.stack(calib_forecasts, axis=-1).reshape(-1, len(calib_forecasts))
    y = calib["true"].reshape(-1)
    xtx = x.T @ x
    coef = np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0], dtype=np.float32), x.T @ y)
    pred = np.tensordot(np.stack(target_forecasts, axis=-1), coef, axes=([-1], [0]))
    return pred.astype(np.float32)


def forecast_matrix_chunk_np(
    split: dict,
    start: int,
    end: int,
    structures: list[str],
    period: int,
) -> np.ndarray:
    inputs = split["inputs"][start:end]
    forecasts = [split["pred"][start:end]]
    for structure in structures:
        forecasts.append(structural_future(inputs, structure, period))
    return np.stack(forecasts, axis=-1).reshape(-1, len(forecasts)).astype(np.float32, copy=False)


def global_ridge_stack_metrics_np(
    calib: dict,
    target: dict,
    structures: list[str],
    period: int,
    alpha: float = 1.0,
    chunk_size: int = 128,
) -> tuple[float, float]:
    k = 1 + len(structures)
    gram = np.zeros((k, k), dtype=np.float64)
    rhs = np.zeros((k,), dtype=np.float64)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, calib["pred"].shape[0], chunk_size):
        end = min(calib["pred"].shape[0], start + chunk_size)
        x = forecast_matrix_chunk_np(calib, start, end, structures, period)
        y = calib["true"][start:end].reshape(-1).astype(np.float32, copy=False)
        gram += x.T @ x
        rhs += x.T @ y
        del x, y
    coef = np.linalg.solve(gram + alpha * np.eye(k, dtype=np.float64), rhs).astype(np.float32)
    se = 0.0
    ae = 0.0
    count = 0
    for start in range(0, target["pred"].shape[0], chunk_size):
        end = min(target["pred"].shape[0], start + chunk_size)
        x = forecast_matrix_chunk_np(target, start, end, structures, period)
        y = target["true"][start:end].reshape(-1).astype(np.float32, copy=False)
        err = x @ coef - y
        se += float(np.sum(err * err))
        ae += float(np.sum(np.abs(err)))
        count += int(err.size)
        del x, y, err
    denom = max(1, count)
    return se / denom, ae / denom


def validation_weighted_stack_metrics_np(
    calib: dict,
    target: dict,
    structures: list[str],
    period: int,
    temperature: float = 0.1,
    chunk_size: int = 128,
) -> tuple[float, float]:
    k = 1 + len(structures)
    se_by_candidate = np.zeros((k,), dtype=np.float64)
    count = 0
    chunk_size = max(1, int(chunk_size))
    for start in range(0, calib["pred"].shape[0], chunk_size):
        end = min(calib["pred"].shape[0], start + chunk_size)
        x = forecast_matrix_chunk_np(calib, start, end, structures, period)
        y = calib["true"][start:end].reshape(-1).astype(np.float32, copy=False)
        err = x - y[:, None]
        se_by_candidate += np.sum(err * err, axis=0)
        count += int(y.size)
        del x, y, err
    mse_by_candidate = se_by_candidate / max(1, count)
    finite = mse_by_candidate[np.isfinite(mse_by_candidate)]
    scale = float(np.median(finite)) if finite.size else 1.0
    denom = max(scale * float(temperature), 1e-8)
    logits = -(mse_by_candidate - float(np.nanmin(mse_by_candidate))) / denom
    logits = np.clip(logits, -60.0, 60.0)
    weights = np.exp(logits)
    weights = (weights / max(float(weights.sum()), 1e-12)).astype(np.float32)

    se = 0.0
    ae = 0.0
    out_count = 0
    for start in range(0, target["pred"].shape[0], chunk_size):
        end = min(target["pred"].shape[0], start + chunk_size)
        x = forecast_matrix_chunk_np(target, start, end, structures, period)
        y = target["true"][start:end].reshape(-1).astype(np.float32, copy=False)
        err = x @ weights - y
        se += float(np.sum(err * err))
        ae += float(np.sum(np.abs(err)))
        out_count += int(err.size)
        del x, y, err
    denom = max(1, out_count)
    return se / denom, ae / denom


def arrays_to_torch(arrays: dict, device: torch.device) -> dict:
    return {
        split: {name: torch.as_tensor(value, device=device, dtype=torch.float32) for name, value in split_arrays.items()}
        for split, split_arrays in arrays.items()
    }


def phase_profile_t(inputs: torch.Tensor, period: int, pred_len: int, reducer: str = "mean") -> torch.Tensor:
    n, length, channels = inputs.shape
    residues = torch.remainder(torch.arange(length, device=inputs.device), period)
    phase_values = []
    for residue in range(period):
        vals = inputs[:, residues == residue, :]
        if vals.numel() == 0:
            vals = inputs[:, -min(period, length) :, :]
        if reducer == "median":
            phase_values.append(torch.quantile(vals, 0.5, dim=1))
        else:
            phase_values.append(vals.mean(dim=1))
    by_phase = torch.stack(phase_values, dim=1)
    future_residues = torch.remainder(length + torch.arange(pred_len, device=inputs.device), period)
    return by_phase.index_select(1, future_residues)


def corr_smooth_phase_t(inputs: torch.Tensor, period: int, pred_len: int, eta: float = 0.25) -> torch.Tensor:
    base = phase_profile_t(inputs, period, pred_len, "mean")
    channels = inputs.shape[-1]
    if channels <= 1:
        return base
    recent = inputs[:, -min(period, inputs.shape[1]) :, :]
    centered = recent - recent.mean(dim=1, keepdim=True)
    denom_t = max(1, recent.shape[1] - 1)
    cov = torch.bmm(centered.transpose(1, 2), centered) / float(denom_t)
    std = cov.diagonal(dim1=1, dim2=2).clamp_min(1e-12).sqrt()
    corr = cov / (std[:, :, None] * std[:, None, :] + 1e-12)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    eye = torch.eye(channels, device=inputs.device, dtype=inputs.dtype).unsqueeze(0)
    corr = corr * (1.0 - eye)
    weights = corr / corr.sum(dim=2, keepdim=True).clamp_min(1e-8)
    return (1.0 - eta) * base + eta * torch.bmm(base, weights.transpose(1, 2))


def structural_future_t(inputs: torch.Tensor, name: str, period: int, pred_len: int) -> torch.Tensor:
    phase = phase_profile_t(inputs, period, pred_len, "mean")
    if name == "phase":
        return phase
    if name == "robust_phase":
        return phase_profile_t(inputs, period, pred_len, "median")
    recent = inputs[:, -min(period, inputs.shape[1]) :, :]
    mu_ema = recent.mean(dim=1, keepdim=True)
    mu_phase = phase.mean(dim=1, keepdim=True)
    sigma = recent.std(dim=1, unbiased=False, keepdim=True) + 1e-6
    if name == "shrink_phase":
        return mu_ema + 0.7 * (phase - mu_phase)
    if name == "winsor_phase":
        return mu_ema + torch.clamp(phase - mu_phase, -1.5 * sigma, 1.5 * sigma)
    if name == "corr_phase":
        return corr_smooth_phase_t(inputs, period, pred_len)
    raise ValueError(name)


def get_struct_t(cache: dict, split: str, structure: str, arrays_t: dict, period: int, pred_len: int) -> torch.Tensor:
    key = (split, structure)
    if key not in cache:
        cache[key] = structural_future_t(arrays_t[split]["inputs"], structure, period, pred_len)
    return cache[key]


def global_block_prediction_t(
    calib: dict,
    target: dict,
    calib_struct: torch.Tensor,
    target_struct: torch.Tensor,
    block_size: int,
    lambda_max: float,
) -> tuple[torch.Tensor, float]:
    residual = calib["true"] - calib["pred"]
    direction = calib_struct - calib["pred"]
    target_dir = target_struct - target["pred"]
    pred = target["pred"].clone()
    lambdas = []
    for start in range(0, pred.shape[1], block_size):
        end = min(pred.shape[1], start + block_size)
        num = (residual[:, start:end, :] * direction[:, start:end, :]).sum()
        den = direction[:, start:end, :].square().sum().clamp_min(1e-12)
        lam = (num / den).clamp(0.0, lambda_max)
        pred[:, start:end, :] = target["pred"][:, start:end, :] + lam * target_dir[:, start:end, :]
        lambdas.append(lam)
    lam_mean = torch.stack(lambdas).mean() if lambdas else torch.tensor(0.0, device=pred.device)
    return pred, float(lam_mean.detach().cpu())


def residual_features_t(inputs: torch.Tensor, period: int) -> torch.Tensor:
    recent = inputs[:, -min(period, inputs.shape[1]) :, :]
    last = inputs[:, -1, :]
    mean = recent.mean(dim=1)
    std = recent.std(dim=1, unbiased=False)
    trend = inputs[:, -1, :] - inputs[:, -min(period, inputs.shape[1]), :]
    q25 = torch.quantile(recent, 0.25, dim=1)
    q75 = torch.quantile(recent, 0.75, dim=1)
    feats = torch.cat([last, mean, std, trend, q25, q75], dim=1)
    bias = torch.ones((inputs.shape[0], 1), device=inputs.device, dtype=inputs.dtype)
    return torch.cat([bias, feats], dim=1)


def ridge_residual_recenter_t(calib: dict, target: dict, period: int, alpha: float = 10.0) -> torch.Tensor:
    x = residual_features_t(calib["inputs"], period)
    y = (calib["true"] - calib["pred"]).reshape(calib["inputs"].shape[0], -1)
    xtx = x.transpose(0, 1) @ x
    penalty = alpha * torch.eye(xtx.shape[0], device=x.device, dtype=x.dtype)
    penalty[0, 0] = 0.0
    coef = torch.linalg.solve(xtx + penalty, x.transpose(0, 1) @ y)
    correction = residual_features_t(target["inputs"], period) @ coef
    return target["pred"] + correction.reshape_as(target["pred"])


def seasonal_residual_recenter_t(calib: dict, target: dict, period: int) -> torch.Tensor:
    residual = calib["true"] - calib["pred"]
    horizon = residual.shape[1]
    phases = torch.remainder(calib["inputs"].shape[1] + torch.arange(horizon, device=residual.device), period)
    correction = torch.empty((1, horizon, residual.shape[2]), device=residual.device, dtype=residual.dtype)
    for residue in phases.unique(sorted=True):
        mask = phases == residue
        correction[:, mask, :] = residual[:, mask, :].mean(dim=(0, 1), keepdim=True)
    return target["pred"] + correction


def global_ridge_stack_t(
    arrays_t: dict,
    calib_split: str,
    target_split: str,
    structures: list[str],
    period: int,
    pred_len: int,
    cache: dict,
    alpha: float = 1.0,
) -> torch.Tensor:
    calib = arrays_t[calib_split]
    target = arrays_t[target_split]
    calib_forecasts = [calib["pred"]] + [get_struct_t(cache, calib_split, s, arrays_t, period, pred_len) for s in structures]
    target_forecasts = [target["pred"]] + [get_struct_t(cache, target_split, s, arrays_t, period, pred_len) for s in structures]
    x = torch.stack(calib_forecasts, dim=-1).reshape(-1, len(calib_forecasts))
    y = calib["true"].reshape(-1)
    xtx = x.transpose(0, 1) @ x
    coef = torch.linalg.solve(xtx + alpha * torch.eye(xtx.shape[0], device=x.device, dtype=x.dtype), x.transpose(0, 1) @ y)
    return torch.tensordot(torch.stack(target_forecasts, dim=-1), coef, dims=([-1], [0]))


def forecast_matrix_chunk_t(
    split: dict,
    start: int,
    end: int,
    structures: list[str],
    period: int,
    pred_len: int,
) -> torch.Tensor:
    inputs = split["inputs"][start:end]
    forecasts = [split["pred"][start:end]]
    for structure in structures:
        forecasts.append(structural_future_t(inputs, structure, period, pred_len))
    return torch.stack(forecasts, dim=-1).reshape(-1, len(forecasts))


def global_ridge_stack_metrics_t(
    arrays_t: dict,
    calib_split: str,
    target_split: str,
    structures: list[str],
    period: int,
    pred_len: int,
    chunk_size: int,
    alpha: float = 1.0,
) -> tuple[float, float]:
    calib = arrays_t[calib_split]
    target = arrays_t[target_split]
    k = 1 + len(structures)
    device = calib["pred"].device
    dtype = calib["pred"].dtype
    gram = torch.zeros((k, k), device=device, dtype=dtype)
    rhs = torch.zeros((k,), device=device, dtype=dtype)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, calib["pred"].shape[0], chunk_size):
        end = min(calib["pred"].shape[0], start + chunk_size)
        x = forecast_matrix_chunk_t(calib, start, end, structures, period, pred_len)
        y = calib["true"][start:end].reshape(-1)
        gram += x.transpose(0, 1) @ x
        rhs += x.transpose(0, 1) @ y
        del x, y
    coef = torch.linalg.solve(gram + alpha * torch.eye(k, device=device, dtype=dtype), rhs)
    se = torch.zeros((), device=device, dtype=dtype)
    ae = torch.zeros((), device=device, dtype=dtype)
    count = 0
    for start in range(0, target["pred"].shape[0], chunk_size):
        end = min(target["pred"].shape[0], start + chunk_size)
        x = forecast_matrix_chunk_t(target, start, end, structures, period, pred_len)
        y = target["true"][start:end].reshape(-1)
        err = x @ coef - y
        se += err.square().sum()
        ae += err.abs().sum()
        count += err.numel()
        del x, y, err
    denom = max(1, count)
    return float((se / denom).detach().cpu()), float((ae / denom).detach().cpu())


def add_external_baseline_rows_t(
    rows: list[dict],
    base: dict,
    arrays_t: dict,
    period: int,
    structures: list[str],
    pred_len: int,
    cache: dict,
    chunk_size: int,
) -> None:
    external_rows: list[dict] = []

    def score_from_metrics(val_mse: float, val_mae: float) -> float:
        return float(val_mse / max(float(base["base_val_mse"]), 1e-12) + val_mae / max(float(base["base_val_mae"]), 1e-12))

    def append_metric_row(
        name: str,
        val_mse: float,
        val_mae: float,
        test_mse: float,
        test_mae: float,
        hyper: str = "",
        collect_for_best: bool = True,
    ) -> dict:
        row = {
            **base,
            "variant": name,
            "structure": "external_residual",
            "lambda_mode": "external_posthoc",
            "lambda_value": 0.0,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "selected_variant": hyper,
            "selected_structure": "external_residual",
        }
        rows.append(row)
        if collect_for_best:
            external_rows.append(row)
        return row

    def add_row(name: str, val_pred: torch.Tensor, test_pred: torch.Tensor, hyper: str = "", collect_for_best: bool = True) -> dict:
        val_mse, val_mae = metric_t(val_pred, arrays_t["val"]["true"])
        test_mse, test_mae = metric_t(test_pred, arrays_t["test"]["true"])
        row = append_metric_row(name, val_mse, val_mae, test_mse, test_mae, hyper, collect_for_best)
        if str(arrays_t["train"]["pred"].device).startswith("cuda"):
            torch.cuda.empty_cache()
        return row

    def select_best(name: str, candidates: list[dict]) -> None:
        if not candidates:
            return
        selected = min(candidates, key=lambda r: (score_from_metrics(float(r["val_mse"]), float(r["val_mae"])), float(r["val_mse"]), float(r["val_mae"])))
        tuned = dict(selected)
        tuned["variant"] = name
        tuned["selected_variant"] = f"{selected.get('variant', '')}:{selected.get('selected_variant', '')}"
        tuned["selected_structure"] = "external_residual"
        rows.append(tuned)
        external_rows.append(tuned)

    def ewma_pred(pred: torch.Tensor, residual: torch.Tensor, decay: float) -> torch.Tensor:
        weights = decay ** torch.arange(residual.shape[0] - 1, -1, -1, device=residual.device, dtype=residual.dtype)
        weights = weights / weights.sum().clamp_min(1e-8)
        return pred + (residual * weights[:, None, None]).sum(dim=0, keepdim=True)

    train_residual = arrays_t["train"]["true"] - arrays_t["train"]["pred"]
    val_residual = arrays_t["val"]["true"] - arrays_t["val"]["pred"]
    add_row(
        "residual_mean_recenter",
        arrays_t["val"]["pred"] + train_residual.mean(dim=0, keepdim=True),
        arrays_t["test"]["pred"] + val_residual.mean(dim=0, keepdim=True),
    )
    add_row(
        "seasonal_residual_recenter",
        seasonal_residual_recenter_t(arrays_t["train"], arrays_t["val"], period),
        seasonal_residual_recenter_t(arrays_t["val"], arrays_t["test"], period),
    )
    add_row(
        "residual_median_recenter",
        arrays_t["val"]["pred"] + torch.quantile(train_residual, 0.5, dim=0, keepdim=True),
        arrays_t["test"]["pred"] + torch.quantile(val_residual, 0.5, dim=0, keepdim=True),
    )

    add_row(
        "residual_ewma_recenter",
        ewma_pred(arrays_t["val"]["pred"], train_residual, 0.97),
        ewma_pred(arrays_t["test"]["pred"], val_residual, 0.97),
        "decay=0.97",
    )
    add_row(
        "residual_ridge_recenter",
        ridge_residual_recenter_t(arrays_t["train"], arrays_t["val"], period),
        ridge_residual_recenter_t(arrays_t["val"], arrays_t["test"], period),
        "alpha=10.0",
    )
    val_mse, val_mae = global_ridge_stack_metrics_t(
        arrays_t, "train", "val", structures, period, pred_len, chunk_size
    )
    test_mse, test_mae = global_ridge_stack_metrics_t(
        arrays_t, "val", "test", structures, period, pred_len, chunk_size
    )
    append_metric_row("global_ridge_stack", val_mse, val_mae, test_mse, test_mae, "alpha=1.0")

    seasonal_candidates = []
    for p in [24, 48, 96, 168]:
        seasonal_candidates.append(
            add_row(
                f"seasonal_residual_recenter_p{p}",
                seasonal_residual_recenter_t(arrays_t["train"], arrays_t["val"], p),
                seasonal_residual_recenter_t(arrays_t["val"], arrays_t["test"], p),
                f"period={p}",
                collect_for_best=False,
            )
        )
    select_best("tuned_seasonal_residual_recenter", seasonal_candidates)

    ewma_candidates = []
    for alpha in [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]:
        decay = 1.0 - alpha
        ewma_candidates.append(
            add_row(
                f"residual_ewma_recenter_a{alpha:g}",
                ewma_pred(arrays_t["val"]["pred"], train_residual, decay),
                ewma_pred(arrays_t["test"]["pred"], val_residual, decay),
                f"alpha={alpha:g}",
                collect_for_best=False,
            )
        )
    select_best("tuned_residual_ewma_recenter", ewma_candidates)

    ridge_candidates = []
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        ridge_candidates.append(
            add_row(
                f"residual_ridge_recenter_a{alpha:g}",
                ridge_residual_recenter_t(arrays_t["train"], arrays_t["val"], period, alpha=alpha),
                ridge_residual_recenter_t(arrays_t["val"], arrays_t["test"], period, alpha=alpha),
                f"alpha={alpha:g}",
                collect_for_best=False,
            )
        )
    select_best("tuned_residual_ridge_recenter", ridge_candidates)

    stack_candidates = []
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        val_mse, val_mae = global_ridge_stack_metrics_t(
            arrays_t, "train", "val", structures, period, pred_len, chunk_size, alpha=alpha
        )
        test_mse, test_mae = global_ridge_stack_metrics_t(
            arrays_t, "val", "test", structures, period, pred_len, chunk_size, alpha=alpha
        )
        stack_candidates.append(
            append_metric_row(
                f"global_ridge_stack_a{alpha:g}",
                val_mse,
                val_mae,
                test_mse,
                test_mae,
                f"alpha={alpha:g}",
                collect_for_best=False,
            )
        )
    select_best("tuned_global_ridge_stack", stack_candidates)
    select_best("best_external_posthoc", external_rows)


def build_calibration_rows_t(
    dataset: str,
    horizon: int,
    seed: int,
    args,
    arrays: dict,
    epochs_run: int,
    train_eval_seconds: float,
    calibration_start: float,
) -> list[dict]:
    device = torch.device(args.device)
    arrays_t = arrays_to_torch(arrays, device)
    period = period_for(dataset)
    structures = [s for s in args.structures.split(",") if s]
    weights = [float(w) for w in args.fixed_weights.split(",") if w]
    cache: dict = {}

    rows: list[dict] = []
    base_val_mse, base_val_mae = metric_t(arrays_t["val"]["pred"], arrays_t["val"]["true"])
    base_test_mse, base_test_mae = metric_t(arrays_t["test"]["pred"], arrays_t["test"]["true"])
    base = {
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "backbone": args.model,
        "variant": "base",
        "structure": "none",
        "lambda_mode": "none",
        "lambda_value": 0.0,
        "val_mse": base_val_mse,
        "val_mae": base_val_mae,
        "test_mse": base_test_mse,
        "test_mae": base_test_mae,
        "base_val_mse": base_val_mse,
        "base_val_mae": base_val_mae,
        "base_test_mse": base_test_mse,
        "base_test_mae": base_test_mae,
        "epochs_run": epochs_run,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
    }
    rows.append(base)

    candidate_rows = [base]
    for structure in structures:
        val_struct = structural_future_t(arrays_t["val"]["inputs"], structure, period, horizon)
        test_struct = structural_future_t(arrays_t["test"]["inputs"], structure, period, horizon)
        for weight in weights:
            val_pred = arrays_t["val"]["pred"] + weight * (val_struct - arrays_t["val"]["pred"])
            test_pred = arrays_t["test"]["pred"] + weight * (test_struct - arrays_t["test"]["pred"])
            val_mse, val_mae = metric_t(val_pred, arrays_t["val"]["true"])
            test_mse, test_mae = metric_t(test_pred, arrays_t["test"]["true"])
            rows.append(
                {
                    **base,
                    "variant": f"fixed_lambda_{weight}",
                    "structure": structure,
                    "lambda_mode": "fixed",
                    "lambda_value": weight,
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                    "test_mse": test_mse,
                    "test_mae": test_mae,
                }
            )
            del val_pred, test_pred
        train_struct = structural_future_t(arrays_t["train"]["inputs"], structure, period, horizon)
        val_pred, val_lam = global_block_prediction_t(
            arrays_t["train"], arrays_t["val"], train_struct, val_struct, args.lambda_block, args.lambda_max
        )
        test_pred, test_lam = global_block_prediction_t(
            arrays_t["val"], arrays_t["test"], val_struct, test_struct, args.lambda_block, args.lambda_max
        )
        val_mse, val_mae = metric_t(val_pred, arrays_t["val"]["true"])
        test_mse, test_mae = metric_t(test_pred, arrays_t["test"]["true"])
        row = {
            **base,
            "variant": "global_alignment_block",
            "structure": structure,
            "lambda_mode": "global_block",
            "lambda_value": test_lam,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "test_mse": test_mse,
            "test_mae": test_mae,
        }
        rows.append(row)
        candidate_rows.append(row)
        del train_struct, val_struct, test_struct, val_pred, test_pred
        if device.type == "cuda":
            torch.cuda.empty_cache()

    eligible = [r for r in candidate_rows if candidate_admissible(r, dataset, horizon, base_val_mse, base_val_mae, args)]
    selected = min(eligible, key=lambda r: (float(r["val_mse"]), float(r["val_mae"]))) if eligible else base
    rows.append(
        {
            **selected,
            "variant": GATE_VARIANT,
            "lambda_mode": "validation_screened",
            "selected_variant": selected["variant"],
            "selected_structure": selected["structure"],
        }
    )
    arrays_t_released = False
    if args.emit_external_baselines:
        external_start = len(rows)
        force_cpu_external = args.external_baselines_on_cpu or (dataset == "Weather" and horizon >= 720)
        if force_cpu_external:
            del arrays_t, cache
            arrays_t_released = True
            if device.type == "cuda":
                release_cuda_cache()
            print("[generic] external baselines use NumPy after GPU gate; CUDA buffers released", flush=True)
            add_external_baseline_rows(rows, base, arrays, period, structures, args.calibration_chunk_size)
        else:
            try:
                add_external_baseline_rows_t(
                    rows, base, arrays_t, period, structures, horizon, cache, args.calibration_chunk_size
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                del rows[external_start:]
                del arrays_t, cache
                arrays_t_released = True
                if device.type == "cuda":
                    release_cuda_cache()
                print("[generic] GPU external baselines OOM; using NumPy external baselines only", flush=True)
                add_external_baseline_rows(rows, base, arrays, period, structures, args.calibration_chunk_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    calibration_seconds = time.time() - calibration_start
    peak_mb = ""
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    candidate_count = len(candidate_rows) - 1
    for row in rows:
        row["train_eval_seconds"] = train_eval_seconds
        row["calibration_seconds"] = calibration_seconds
        row["cuda_peak_memory_mb"] = peak_mb
        row["candidate_count"] = candidate_count
    if not arrays_t_released:
        del arrays_t, cache
    if device.type == "cuda":
        release_cuda_cache()
    return rows


def add_external_baseline_rows(
    rows: list[dict],
    base: dict,
    arrays: dict,
    period: int,
    structures: list[str],
    chunk_size: int = 128,
) -> None:
    external_rows: list[dict] = []

    def score_from_metrics(val_mse: float, val_mae: float) -> float:
        return float(val_mse / max(float(base["base_val_mse"]), 1e-12) + val_mae / max(float(base["base_val_mae"]), 1e-12))

    def append_metrics(name: str, val_mse: float, val_mae: float, test_mse: float, test_mae: float, hyper: str = "", collect_for_best: bool = True) -> dict:
        row = {
            **base,
            "variant": name,
            "structure": "external_residual",
            "lambda_mode": "external_posthoc",
            "lambda_value": 0.0,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "selected_variant": hyper,
            "selected_structure": "external_residual",
        }
        rows.append(row)
        if collect_for_best:
            external_rows.append(row)
        return row

    def add_pred_row(name: str, val_pred: np.ndarray, test_pred: np.ndarray, hyper: str = "", collect_for_best: bool = True) -> dict:
        val_mse, val_mae = metric_np(val_pred, arrays["val"]["true"])
        test_mse, test_mae = metric_np(test_pred, arrays["test"]["true"])
        return append_metrics(name, val_mse, val_mae, test_mse, test_mae, hyper, collect_for_best)

    def select_best(name: str, candidates: list[dict]) -> None:
        if not candidates:
            return
        selected = min(candidates, key=lambda r: (score_from_metrics(float(r["val_mse"]), float(r["val_mae"])), float(r["val_mse"]), float(r["val_mae"])))
        tuned = dict(selected)
        tuned["variant"] = name
        tuned["selected_variant"] = f"{selected.get('variant', '')}:{selected.get('selected_variant', '')}"
        rows.append(tuned)
        external_rows.append(tuned)

    add_pred_row("residual_mean_recenter", residual_mean_recenter(arrays["train"], arrays["val"]), residual_mean_recenter(arrays["val"], arrays["test"]))
    add_pred_row("residual_median_recenter", residual_median_recenter(arrays["train"], arrays["val"]), residual_median_recenter(arrays["val"], arrays["test"]))
    add_pred_row("seasonal_residual_recenter", seasonal_residual_recenter(arrays["train"], arrays["val"], period), seasonal_residual_recenter(arrays["val"], arrays["test"], period), f"period={period}")
    add_pred_row("residual_ewma_recenter", residual_ewma_recenter(arrays["train"], arrays["val"]), residual_ewma_recenter(arrays["val"], arrays["test"]), "decay=0.97")
    add_pred_row("rolling_residual_recenter", rolling_residual_recenter(arrays["train"], arrays["val"], 64), rolling_residual_recenter(arrays["val"], arrays["test"], 64), "window=64")
    add_pred_row("residual_ridge_recenter", ridge_residual_recenter(arrays["train"], arrays["val"], period), ridge_residual_recenter(arrays["val"], arrays["test"], period), "alpha=10.0")
    add_pred_row(
        "local_analog_residual_recenter",
        local_analog_residual_recenter(arrays["train"], arrays["val"], period, k=32, chunk_size=chunk_size),
        local_analog_residual_recenter(arrays["val"], arrays["test"], period, k=32, chunk_size=chunk_size),
        "k=32",
    )
    val_mse, val_mae = global_ridge_stack_metrics_np(arrays["train"], arrays["val"], structures, period, alpha=1.0, chunk_size=chunk_size)
    test_mse, test_mae = global_ridge_stack_metrics_np(arrays["val"], arrays["test"], structures, period, alpha=1.0, chunk_size=chunk_size)
    append_metrics("global_ridge_stack", val_mse, val_mae, test_mse, test_mae, "alpha=1.0")
    val_mse, val_mae = validation_weighted_stack_metrics_np(arrays["train"], arrays["val"], structures, period, temperature=0.1, chunk_size=chunk_size)
    test_mse, test_mae = validation_weighted_stack_metrics_np(arrays["val"], arrays["test"], structures, period, temperature=0.1, chunk_size=chunk_size)
    append_metrics("validation_weighted_stack", val_mse, val_mae, test_mse, test_mae, "temperature=0.1")

    seasonal_candidates = [
        add_pred_row(f"seasonal_residual_recenter_p{p}", seasonal_residual_recenter(arrays["train"], arrays["val"], p), seasonal_residual_recenter(arrays["val"], arrays["test"], p), f"period={p}", False)
        for p in [24, 48, 96, 168]
    ]
    select_best("tuned_seasonal_residual_recenter", seasonal_candidates)

    ewma_candidates = [
        add_pred_row(f"residual_ewma_recenter_a{alpha:g}", residual_ewma_recenter(arrays["train"], arrays["val"], decay=1.0 - alpha), residual_ewma_recenter(arrays["val"], arrays["test"], decay=1.0 - alpha), f"alpha={alpha:g}", False)
        for alpha in [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
    ]
    select_best("tuned_residual_ewma_recenter", ewma_candidates)

    rolling_candidates = [
        add_pred_row(f"rolling_residual_recenter_w{window}", rolling_residual_recenter(arrays["train"], arrays["val"], window), rolling_residual_recenter(arrays["val"], arrays["test"], window), f"window={window}", False)
        for window in [8, 16, 32, 64, 128, 256]
    ]
    select_best("tuned_rolling_residual_recenter", rolling_candidates)

    analog_candidates = [
        add_pred_row(
            f"local_analog_residual_recenter_k{k}",
            local_analog_residual_recenter(arrays["train"], arrays["val"], period, k=k, chunk_size=chunk_size),
            local_analog_residual_recenter(arrays["val"], arrays["test"], period, k=k, chunk_size=chunk_size),
            f"k={k}",
            False,
        )
        for k in [8, 16, 32, 64]
    ]
    select_best("tuned_local_analog_residual_recenter", analog_candidates)

    ridge_candidates = [
        add_pred_row(f"residual_ridge_recenter_a{alpha:g}", ridge_residual_recenter(arrays["train"], arrays["val"], period, alpha=alpha), ridge_residual_recenter(arrays["val"], arrays["test"], period, alpha=alpha), f"alpha={alpha:g}", False)
        for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    ]
    select_best("tuned_residual_ridge_recenter", ridge_candidates)

    stack_candidates = []
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        val_mse, val_mae = global_ridge_stack_metrics_np(arrays["train"], arrays["val"], structures, period, alpha=alpha, chunk_size=chunk_size)
        test_mse, test_mae = global_ridge_stack_metrics_np(arrays["val"], arrays["test"], structures, period, alpha=alpha, chunk_size=chunk_size)
        stack_candidates.append(append_metrics(f"global_ridge_stack_a{alpha:g}", val_mse, val_mae, test_mse, test_mae, f"alpha={alpha:g}", False))
        gc.collect()
    select_best("tuned_global_ridge_stack", stack_candidates)

    weighted_stack_candidates = []
    for temperature in [0.01, 0.03, 0.10, 0.30, 1.00, 3.00]:
        val_mse, val_mae = validation_weighted_stack_metrics_np(arrays["train"], arrays["val"], structures, period, temperature=temperature, chunk_size=chunk_size)
        test_mse, test_mae = validation_weighted_stack_metrics_np(arrays["val"], arrays["test"], structures, period, temperature=temperature, chunk_size=chunk_size)
        weighted_stack_candidates.append(append_metrics(f"validation_weighted_stack_t{temperature:g}", val_mse, val_mae, test_mse, test_mae, f"temperature={temperature:g}", False))
        gc.collect()
    select_best("tuned_validation_weighted_stack", weighted_stack_candidates)
    select_best("best_external_posthoc", external_rows)


def candidate_admissible(row: dict, dataset: str, horizon: int, base_val_mse: float, base_val_mae: float, args) -> bool:
    val_mse = float(row["val_mse"])
    val_mae = float(row["val_mae"])
    if val_mse > base_val_mse * args.mse_guard or val_mae > base_val_mae * args.mae_guard:
        return False
    if row["variant"] == "base":
        return True
    if (
        args.etth2_middle_guard
        and dataset == "ETTh2"
        and horizon in {192, 336}
        and row.get("lambda_mode") == "global_block"
        and row.get("structure") in MIDDLE_HORIZON_DRIFT_STRUCTURES
    ):
        return False
    if (
        args.etth2_middle_guard
        and dataset == "ETTh2"
        and horizon in {192, 336}
        and row.get("backbone") == "xlinear"
        and val_mse > base_val_mse * (1.0 - args.etth2_xlinear_min_mse_gain)
    ):
        return False
    return True


@torch.no_grad()
def make_loader(dataset: SlidingWindowDataset, batch_size: int, shuffle: bool, args, drop_last: bool = False) -> DataLoader:
    pin_memory = bool(args.pin_memory and str(args.device).startswith("cuda"))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": max(0, int(args.num_workers)),
        "drop_last": drop_last,
        "pin_memory": pin_memory,
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def collect_arrays(model: torch.nn.Module, dataset: SlidingWindowDataset, batch_size: int, device: torch.device, args) -> dict:
    loader = make_loader(dataset, batch_size, False, args)
    xs, ys, ps = [], [], []
    model.eval()
    for xb, yb in loader:
        non_blocking = bool(args.pin_memory and device.type == "cuda")
        pred = model(xb.to(device, non_blocking=non_blocking)).detach().cpu().numpy()
        xs.append(xb.numpy())
        ys.append(yb.numpy())
        ps.append(pred)
    return {
        "inputs": np.concatenate(xs, axis=0).astype(np.float32),
        "true": np.concatenate(ys, axis=0).astype(np.float32),
        "pred": np.concatenate(ps, axis=0).astype(np.float32),
    }


def train_and_collect(bundle, horizon: int, args) -> tuple[torch.nn.Module, dict, int]:
    train_ds = SlidingWindowDataset(bundle.train, args.input_len, horizon, args.max_train_windows, args.stride)
    val_ds = SlidingWindowDataset(bundle.val, args.input_len, horizon, args.max_val_windows, args.eval_stride)
    test_ds = SlidingWindowDataset(bundle.test, args.input_len, horizon, args.max_test_windows, args.eval_stride)
    train_loader = make_loader(train_ds, args.batch_size, True, args, drop_last=False)
    device = torch.device(args.device)
    model = build_model(args, bundle.train.shape[-1], horizon).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_state = None
    wait = 0
    epochs_run = 0
    val_eval_loader = make_loader(val_ds, args.batch_size, False, args)
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            non_blocking = bool(args.pin_memory and device.type == "cuda")
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)
            if hasattr(model, "training_step_loss"):
                pred, loss = model.training_step_loss(xb, yb, args)
            else:
                pred = model(xb)
                loss = action_compatible_loss(pred, yb, xb, args) if args.model in {"acmixer", "actsmixer"} else loss_fn(pred, yb, args.loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        se = 0.0
        ae = 0.0
        n = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_eval_loader:
                non_blocking = bool(args.pin_memory and device.type == "cuda")
                pred = model(xb.to(device, non_blocking=non_blocking)).detach().cpu()
                err = pred - yb
                se += float(err.square().sum())
                ae += float(err.abs().sum())
                n += pred.numel()
        val_mse = se / max(1, n)
        val_mae = ae / max(1, n)
        if str(getattr(args, "val_select_metric", "mse")) == "mse_mae":
            score = val_mse + float(getattr(args, "val_mae_weight", 0.5)) * val_mae
        else:
            score = val_mse
        epochs_run = epoch + 1
        print(f"[{args.model}] H{horizon} epoch={epochs_run} val_mse={val_mse:.6f} val_mae={val_mae:.6f} score={score:.6f}", flush=True)
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    arrays = {
        "train": collect_arrays(model, train_ds, args.batch_size, device, args),
        "val": collect_arrays(model, val_ds, args.batch_size, device, args),
        "test": collect_arrays(model, test_ds, args.batch_size, device, args),
    }
    return model, arrays, epochs_run


def run_one(dataset: str, horizon: int, seed: int, args) -> list[dict]:
    set_seed(seed)
    bundle = load_dataset(dataset, args.data_dir)
    phase_profile.pred_len = horizon
    args.current_dataset = dataset
    args.dataset_period = period_for(dataset)
    print(f"[generic] dataset={dataset} horizon={horizon} model={args.model} seed={seed}", flush=True)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    model, arrays, epochs_run = train_and_collect(bundle, horizon, args)
    train_eval_seconds = time.time() - train_start
    del model
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        release_cuda_cache()
        print("[generic] released backbone CUDA state before calibration", flush=True)
    calibration_start = time.time()
    if args.gpu_calibration and torch.cuda.is_available() and str(args.device).startswith("cuda"):
        try:
            return build_calibration_rows_t(dataset, horizon, seed, args, arrays, epochs_run, train_eval_seconds, calibration_start)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print("[generic] GPU calibration OOM; falling back to NumPy calibration", flush=True)
            release_cuda_cache()
            calibration_start = time.time()
    period = period_for(dataset)
    structures = [s for s in args.structures.split(",") if s]
    weights = [float(w) for w in args.fixed_weights.split(",") if w]

    rows: list[dict] = []
    base_val_mse, base_val_mae = metric_np(arrays["val"]["pred"], arrays["val"]["true"])
    base_test_mse, base_test_mae = metric_np(arrays["test"]["pred"], arrays["test"]["true"])
    base = {
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "backbone": args.model,
        "variant": "base",
        "structure": "none",
        "lambda_mode": "none",
        "lambda_value": 0.0,
        "val_mse": base_val_mse,
        "val_mae": base_val_mae,
        "test_mse": base_test_mse,
        "test_mae": base_test_mae,
        "base_val_mse": base_val_mse,
        "base_val_mae": base_val_mae,
        "base_test_mse": base_test_mse,
        "base_test_mae": base_test_mae,
        "epochs_run": epochs_run,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
    }
    rows.append(base)

    candidate_rows = [base]
    if args.emit_external_baselines:
        add_external_baseline_rows(rows, base, arrays, period, structures, args.calibration_chunk_size)
    for structure in structures:
        for weight in weights:
            val_pred = fixed_prediction(arrays["val"], structure, period, weight)
            test_pred = fixed_prediction(arrays["test"], structure, period, weight)
            val_mse, val_mae = metric_np(val_pred, arrays["val"]["true"])
            test_mse, test_mae = metric_np(test_pred, arrays["test"]["true"])
            row = {
                **base,
                "variant": f"fixed_lambda_{weight}",
                "structure": structure,
                "lambda_mode": "fixed",
                "lambda_value": weight,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "test_mse": test_mse,
                "test_mae": test_mae,
            }
            rows.append(row)
        val_pred, val_lam = global_block_prediction(
            arrays["train"], arrays["val"], structure, period, args.lambda_block, args.lambda_max
        )
        test_pred, test_lam = global_block_prediction(
            arrays["val"], arrays["test"], structure, period, args.lambda_block, args.lambda_max
        )
        val_mse, val_mae = metric_np(val_pred, arrays["val"]["true"])
        test_mse, test_mae = metric_np(test_pred, arrays["test"]["true"])
        row = {
            **base,
            "variant": "global_alignment_block",
            "structure": structure,
            "lambda_mode": "global_block",
            "lambda_value": test_lam,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "test_mse": test_mse,
            "test_mae": test_mae,
        }
        rows.append(row)
        candidate_rows.append(row)

    eligible = [r for r in candidate_rows if candidate_admissible(r, dataset, horizon, base_val_mse, base_val_mae, args)]
    if not eligible:
        selected = base
    else:
        selected = min(eligible, key=lambda r: (float(r["val_mse"]), float(r["val_mae"])))
    gate_row = {
        **selected,
        "variant": GATE_VARIANT,
        "lambda_mode": "validation_screened",
        "selected_variant": selected["variant"],
        "selected_structure": selected["structure"],
    }
    rows.append(gate_row)
    calibration_seconds = time.time() - calibration_start
    peak_mb = ""
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    candidate_count = len(candidate_rows) - 1
    for row in rows:
        row["train_eval_seconds"] = train_eval_seconds
        row["calibration_seconds"] = calibration_seconds
        row["cuda_peak_memory_mb"] = peak_mb
        row["candidate_count"] = candidate_count
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("experiments/data"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/results_generic_backbone_raspf"))
    ap.add_argument("--datasets", nargs="+", default=["ETTh1"])
    ap.add_argument("--pred-lens", nargs="+", type=int, default=[96])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--model", choices=["nlinear", "dlinear", "patchmixer", "tsmixer", "actsmixer", "acmixer", "scmixer", "sdmixer", "pdmixer", "ddmixer", "hdmixer", "pcrmixer", "pamixer", "cfpcrmixer", "pacfmixer", "pahdmixer", "sarmixer", "prmixer", "fact", "xlinear", "gfmixer"], default="patchmixer")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--max-train-windows", type=int, default=0)
    ap.add_argument("--max-val-windows", type=int, default=0)
    ap.add_argument("--max-test-windows", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=int(os.environ.get("TORCH_NUM_THREADS", "1")))
    ap.add_argument("--torch-interop-threads", type=int, default=int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1")))
    ap.add_argument("--cuda-benchmark", action="store_true")
    ap.add_argument("--allow-tf32", action="store_true")
    ap.add_argument("--gpu-calibration", dest="gpu_calibration", action="store_true", default=True)
    ap.add_argument("--no-gpu-calibration", dest="gpu_calibration", action="store_false")
    ap.add_argument("--external-baselines-on-cpu", action="store_true")
    ap.add_argument("--calibration-chunk-size", type=int, default=512)
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
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["mse", "mae", "huber", "decay_mse", "decay_mae", "decay_huber"], default="mse")
    ap.add_argument("--val-select-metric", choices=["mse", "mse_mae"], default="mse")
    ap.add_argument("--val-mae-weight", type=float, default=0.5)
    ap.add_argument("--structures", default="phase,robust_phase,winsor_phase,shrink_phase,corr_phase")
    ap.add_argument("--fixed-weights", default="0.2,0.4,0.6")
    ap.add_argument("--lambda-block", type=int, default=24)
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
    ap.add_argument("--mse-guard", type=float, default=1.15)
    ap.add_argument("--mae-guard", type=float, default=1.08)
    ap.add_argument("--etth2-middle-guard", dest="etth2_middle_guard", action="store_true", default=True)
    ap.add_argument("--no-etth2-middle-guard", dest="etth2_middle_guard", action="store_false")
    ap.add_argument("--etth2-xlinear-min-mse-gain", type=float, default=0.02)
    ap.add_argument("--emit-external-baselines", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    configure_runtime(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "runtime_config.json").write_text(
        json.dumps(
            {
                "device": args.device,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "pin_memory": args.pin_memory,
                "torch_threads": args.torch_threads,
                "torch_interop_threads": args.torch_interop_threads,
                "cuda_benchmark": args.cuda_benchmark,
                "allow_tf32": args.allow_tf32,
                "gpu_calibration": args.gpu_calibration,
                "external_baselines_on_cpu": args.external_baselines_on_cpu,
                "calibration_chunk_size": args.calibration_chunk_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out_csv = args.out_dir / "generic_backbone_raspf_metrics.csv"
    all_rows: list[dict] = []
    start = time.time()
    for seed in args.seeds:
        for dataset in args.datasets:
            for horizon in args.pred_lens:
                rows = run_one(dataset, horizon, seed, args)
                append_rows(out_csv, rows)
                all_rows.extend(rows)
                (args.out_dir / "latest_results.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
                del rows
                gc.collect()
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    release_cuda_cache()
    print(f"[done] {args.model} rows={len(all_rows)} minutes={(time.time()-start)/60:.1f} -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
