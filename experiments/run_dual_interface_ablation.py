import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from run_generic_backbone_raspf_protocol import (
    GATE_VARIANT,
    build_calibration_rows_t,
    make_loader,
    metric_np,
    release_cuda_cache,
    set_seed,
    train_and_collect,
)
from run_sdrc_experiments import load_dataset
from train_backbone_forecaster import SlidingWindowDataset


class InputActionDataset(Dataset):
    def __init__(self, base: SlidingWindowDataset, action: str):
        self.base = base
        self.action = action

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.action.startswith("revin_"):
            return x, y
        return apply_input_action(x, self.action), y


def moving_average_1d(x: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    pad = (kernel - 1) // 2
    left = x[:1].repeat(pad, 1)
    right = x[-1:].repeat(pad, 1)
    padded = torch.cat([left, x, right], dim=0)
    return torch.nn.functional.avg_pool1d(
        padded.T.unsqueeze(0), kernel_size=kernel, stride=1
    ).squeeze(0).T


def apply_input_action(x: torch.Tensor, action: str) -> torch.Tensor:
    if action == "identity":
        return x
    if action == "smooth5":
        return 0.75 * x + 0.25 * moving_average_1d(x, 5)
    if action == "winsor_iqr":
        q1 = torch.quantile(x, 0.25, dim=0, keepdim=True)
        q3 = torch.quantile(x, 0.75, dim=0, keepdim=True)
        iqr = (q3 - q1).clamp_min(1e-4)
        lo = q1 - 2.5 * iqr
        hi = q3 + 2.5 * iqr
        return torch.minimum(torch.maximum(x, lo), hi)
    if action == "robust_scale":
        med = x.median(dim=0, keepdim=True).values
        mad = (x - med).abs().median(dim=0, keepdim=True).values.clamp_min(1e-3)
        z = ((x - med) / (1.4826 * mad)).clamp(-4.0, 4.0)
        return med + z * (1.4826 * mad)
    if action == "unsafe_detrend":
        t = torch.linspace(0.0, 1.0, x.shape[0], dtype=x.dtype, device=x.device).view(-1, 1)
        slope = x[-1:] - x[:1]
        trend = x[:1] + t * slope
        return x - 0.7 * trend + 0.7 * x[-1:]
    raise ValueError(f"Unknown input action: {action}")


def parse_revin_action(action: str) -> tuple[str, float]:
    parts = action.split("_")
    if len(parts) < 3 or parts[0] != "revin":
        raise ValueError(f"Invalid RevIN action: {action}")
    kind = parts[1]
    eta_token = parts[2]
    eta_map = {"025": 0.25, "05": 0.5, "075": 0.75, "1": 1.0}
    eta = eta_map.get(eta_token)
    if eta is None:
        eta = float(eta_token)
    return kind, eta


def ema_stats(x: torch.Tensor, decay: float = 0.92) -> tuple[torch.Tensor, torch.Tensor]:
    length = x.shape[1]
    weights = decay ** torch.arange(length - 1, -1, -1, dtype=x.dtype, device=x.device)
    weights = weights / weights.sum().clamp_min(1e-8)
    w = weights.view(1, length, 1)
    mu = (x * w).sum(dim=1, keepdim=True)
    var = ((x - mu).square() * w).sum(dim=1, keepdim=True)
    return mu, var.sqrt().clamp_min(1e-5)


def robust_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    med = x.median(dim=1, keepdim=True).values
    mad = (x - med).abs().median(dim=1, keepdim=True).values
    return med, (1.4826 * mad).clamp_min(1e-5)


def revin_transform_batch(
    x: torch.Tensor,
    action: str,
    train_mean: torch.Tensor,
    train_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kind, eta = parse_revin_action(action)
    if kind in {"mean", "std"}:
        inst_mu = x.mean(dim=1, keepdim=True)
        inst_std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
    elif kind in {"median", "mad", "robust"}:
        inst_mu, inst_std = robust_stats(x)
    elif kind == "ema":
        inst_mu, inst_std = ema_stats(x)
    else:
        raise ValueError(f"Unknown RevIN statistic source: {kind}")
    mu0 = train_mean.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
    std0 = train_std.to(device=x.device, dtype=x.dtype).view(1, 1, -1).clamp_min(1e-5)
    mu = (1.0 - eta) * mu0 + eta * inst_mu
    std = (1.0 - eta) * std0 + eta * inst_std
    return (x - mu) / std.clamp_min(1e-5), mu, std


@torch.no_grad()
def collect_arrays_with_action(
    model,
    dataset,
    action: str,
    batch_size: int,
    device: torch.device,
    args,
    train_mean: torch.Tensor | None = None,
    train_std: torch.Tensor | None = None,
) -> dict:
    loader = make_loader(InputActionDataset(dataset, action), batch_size, False, args)
    xs, ys, ps = [], [], []
    model.eval()
    for xb, yb in loader:
        non_blocking = bool(args.pin_memory and device.type == "cuda")
        xb_dev = xb.to(device, non_blocking=non_blocking)
        if action.startswith("revin_"):
            if train_mean is None or train_std is None:
                raise ValueError("RevIN actions require training statistics.")
            xb_model, mu, std = revin_transform_batch(xb_dev, action, train_mean, train_std)
            pred = model(xb_model)
            pred = pred * std + mu
            pred = pred.detach().cpu().numpy()
        else:
            pred = model(xb_dev).detach().cpu().numpy()
        xs.append(xb.numpy())
        ys.append(yb.numpy())
        ps.append(pred)
    return {
        "inputs": np.concatenate(xs, axis=0).astype(np.float32),
        "true": np.concatenate(ys, axis=0).astype(np.float32),
        "pred": np.concatenate(ps, axis=0).astype(np.float32),
    }


def collect_action_arrays(model, bundle, horizon: int, action: str, args) -> dict:
    train_ds = SlidingWindowDataset(bundle.train, args.input_len, horizon, args.max_train_windows, args.stride)
    val_ds = SlidingWindowDataset(bundle.val, args.input_len, horizon, args.max_val_windows, args.eval_stride)
    test_ds = SlidingWindowDataset(bundle.test, args.input_len, horizon, args.max_test_windows, args.eval_stride)
    device = torch.device(args.device)
    train_mean = torch.from_numpy(bundle.train.astype("float32").mean(axis=0))
    train_std = torch.from_numpy(bundle.train.astype("float32").std(axis=0)).clamp_min(1e-5)
    return {
        "train": collect_arrays_with_action(model, train_ds, action, args.batch_size, device, args, train_mean, train_std),
        "val": collect_arrays_with_action(model, val_ds, action, args.batch_size, device, args, train_mean, train_std),
        "test": collect_arrays_with_action(model, test_ds, action, args.batch_size, device, args, train_mean, train_std),
    }


def select_by_val(rows: list[dict]) -> dict:
    return min(rows, key=lambda r: (float(r["val_mse"]), float(r["val_mae"])))


def summarize_action_rows(rows: list[dict]) -> dict:
    base_identity = next(r for r in rows if r["input_action"] == "identity" and r["variant"] == "base")
    input_only_pool = [r for r in rows if r["variant"] == "base"]
    output_only_pool = [r for r in rows if r["input_action"] == "identity" and r["variant"] == GATE_VARIANT]
    input_output_pool = [r for r in rows if r["variant"] == GATE_VARIANT]
    unsafe_pool = [r for r in rows if r["input_action"] == "unsafe_detrend" and r["variant"] == "base"]
    selected = {
        "base": base_identity,
        "input_only": select_by_val(input_only_pool),
        "output_only": select_by_val(output_only_pool) if output_only_pool else base_identity,
        "input_output": select_by_val(input_output_pool) if input_output_pool else base_identity,
        "unsafe_input": select_by_val(unsafe_pool) if unsafe_pool else None,
    }
    out = {
        "dataset": base_identity["dataset"],
        "horizon": base_identity["horizon"],
        "seed": base_identity["seed"],
        "backbone": base_identity["backbone"],
        "base_mse": base_identity["test_mse"],
        "base_mae": base_identity["test_mae"],
    }
    for name, row in selected.items():
        if row is None:
            continue
        out[f"{name}_input_action"] = row.get("input_action", "")
        out[f"{name}_variant"] = row.get("variant", "")
        out[f"{name}_structure"] = row.get("structure", "")
        out[f"{name}_mse"] = row["test_mse"]
        out[f"{name}_mae"] = row["test_mae"]
        out[f"{name}_mse_gain_pct"] = (base_identity["test_mse"] - row["test_mse"]) / base_identity["test_mse"] * 100.0
        out[f"{name}_mae_gain_pct"] = (base_identity["test_mae"] - row["test_mae"]) / base_identity["test_mae"] * 100.0
    return out


def run_one(dataset: str, horizon: int, seed: int, args) -> tuple[list[dict], dict]:
    set_seed(seed)
    bundle = load_dataset(dataset, args.data_dir)
    args.current_dataset = dataset
    args.dataset_period = 24 if dataset.startswith("ETTh") else 96 if dataset.startswith("ETTm") else 24
    print(f"[dual] dataset={dataset} horizon={horizon} model={args.model} seed={seed}", flush=True)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    train_start = time.time()
    model, identity_arrays, epochs_run = train_and_collect(bundle, horizon, args)
    train_eval_seconds = time.time() - train_start

    all_rows: list[dict] = []
    for action in args.input_actions.split(","):
        action = action.strip()
        if not action:
            continue
        if action == "identity":
            arrays = identity_arrays
        else:
            arrays = collect_action_arrays(model, bundle, horizon, action, args)
        calibration_start = time.time()
        rows = build_calibration_rows_t(dataset, horizon, seed, args, arrays, epochs_run, train_eval_seconds, calibration_start)
        for row in rows:
            row["input_action"] = action
            row["interface"] = "input+output" if row["variant"] != "base" else ("base" if action == "identity" else "input_only")
            if action != "identity" and row["variant"] == GATE_VARIANT:
                row["interface"] = "input+output_gate"
        all_rows.extend(rows)
        if action != "identity":
            del arrays
            release_cuda_cache()

    del model, identity_arrays
    release_cuda_cache()
    return all_rows, summarize_action_rows(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("experiments/data"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["ETTh1"])
    ap.add_argument("--pred-lens", nargs="+", type=int, default=[96])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026])
    ap.add_argument("--model", choices=["xlinear", "fact", "gfmixer", "patchmixer", "tsmixer", "dlinear", "phaseformer", "dtaf"], default="xlinear")
    ap.add_argument("--input-actions", default="identity,winsor_iqr,smooth5,robust_scale,unsafe_detrend")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--max-train-windows", type=int, default=0)
    ap.add_argument("--max-val-windows", type=int, default=0)
    ap.add_argument("--max-test-windows", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patch-len", type=int, default=16)
    ap.add_argument("--patch-stride", type=int, default=8)
    ap.add_argument("--kernel-size", type=int, default=25)
    ap.add_argument("--individual", action="store_true")
    ap.add_argument("--structures", default="phase,robust_phase,winsor_phase,corr_phase")
    ap.add_argument("--fixed-weights", default="0.2,0.4")
    ap.add_argument("--lambda-block", type=int, default=24)
    ap.add_argument("--lambda-max", type=float, default=0.75)
    ap.add_argument("--calibration-chunk-size", type=int, default=1024)
    ap.add_argument("--emit-external-baselines", action="store_true")
    ap.add_argument("--external-baselines-on-cpu", action="store_true")
    ap.add_argument("--mse-guard", type=float, default=1.15)
    ap.add_argument("--mae-guard", type=float, default=1.08)
    ap.add_argument("--etth2-middle-guard", action="store_true")
    ap.add_argument("--etth2-xlinear-min-mse-gain", type=float, default=0.002)
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
    ap.add_argument("--val-select-metric", choices=["mse", "mse_mae"], default="mse_mae")
    ap.add_argument("--val-mae-weight", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--allow-tf32", action="store_true")
    ap.add_argument("--cuda-benchmark", action="store_true")
    args = ap.parse_args()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.cuda_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    for dataset in args.datasets:
        for horizon in args.pred_lens:
            for seed in args.seeds:
                rows, summary = run_one(dataset, horizon, seed, args)
                all_rows.extend(rows)
                summary_rows.append(summary)
                pd.DataFrame(all_rows).to_csv(args.out_dir / "dual_interface_ablation_rows.csv", index=False)
                pd.DataFrame(summary_rows).to_csv(args.out_dir / "dual_interface_ablation_summary.csv", index=False)
                print("[dual] saved partial results", flush=True)
    print(f"[dual] complete rows={len(all_rows)} summaries={len(summary_rows)} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
