from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


GATE_VARIANT = "RASPF-structural-gate"


def read_rows(run_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in run_root.rglob("dual_interface_ablation_rows.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"[summarize] skip unreadable {path}: {exc}", flush=True)
            continue
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    for col in ["dataset", "backbone", "method", "variant", "structure", "input_action", "interface"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in ["horizon", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    numeric_cols = [
        "val_mse",
        "val_mae",
        "test_mse",
        "test_mae",
        "base_val_mse",
        "base_val_mae",
        "base_test_mse",
        "base_test_mae",
        "lambda_value",
        "train_eval_seconds",
        "calibration_seconds",
        "cuda_peak_memory_mb",
        "candidate_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pick_min(rows: pd.DataFrame, default: pd.Series | None = None, metric: str = "mse_mae") -> pd.Series | None:
    if rows.empty:
        return default
    if metric == "mse":
        sort_cols = ["val_mse", "val_mae", "test_mse", "test_mae"]
    elif metric == "mae":
        sort_cols = ["val_mae", "val_mse", "test_mae", "test_mse"]
    else:
        score = rows["val_mse"].astype(float) / max(float(rows["base_val_mse"].iloc[0]), 1e-12)
        score = score + 0.5 * rows["val_mae"].astype(float) / max(float(rows["base_val_mae"].iloc[0]), 1e-12)
        tmp = rows.assign(_score=score)
        return tmp.sort_values(["_score", "val_mse", "val_mae"], kind="mergesort").iloc[0].drop(labels=["_score"])
    return rows.sort_values(sort_cols, kind="mergesort").iloc[0]


def identity_base(group: pd.DataFrame) -> pd.Series | None:
    base = group[(group["variant"] == "base") & (group["input_action"] == "identity")]
    if base.empty:
        base = group[group["variant"] == "base"]
    return pick_min(base)


def is_revin(value: object) -> bool:
    return str(value).startswith("revin_")


def gain(base: pd.Series, row: pd.Series, metric: str) -> float:
    denom = float(base[f"test_{metric}"])
    if not np.isfinite(denom) or denom == 0.0:
        return np.nan
    return (denom - float(row[f"test_{metric}"])) / denom * 100.0


def row_record(base: pd.Series, row: pd.Series, label: str, label_col: str = "method") -> dict:
    return {
        "dataset": base["dataset"],
        "horizon": int(base["horizon"]),
        "seed": int(base["seed"]) if not pd.isna(base["seed"]) else 0,
        "backbone": str(base["backbone"]).lower(),
        label_col: label,
        "input_action": row.get("input_action", ""),
        "variant": row.get("variant", ""),
        "structure": row.get("structure", ""),
        "lambda_value": row.get("lambda_value", ""),
        "mse": float(row["test_mse"]),
        "mae": float(row["test_mae"]),
        "base_mse": float(base["test_mse"]),
        "base_mae": float(base["test_mae"]),
        "mse_gain_pct": gain(base, row, "mse"),
        "mae_gain_pct": gain(base, row, "mae"),
        "mse_win": int(float(row["test_mse"]) < float(base["test_mse"])),
        "mae_win": int(float(row["test_mae"]) < float(base["test_mae"])),
        "both_win": int(float(row["test_mse"]) < float(base["test_mse"]) and float(row["test_mae"]) < float(base["test_mae"])),
        "any_harm": int(float(row["test_mse"]) > float(base["test_mse"]) or float(row["test_mae"]) > float(base["test_mae"])),
    }


def admissible(candidates: pd.DataFrame, base: pd.Series, mse_guard: float, mae_guard: float) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    return candidates[
        (candidates["val_mse"] <= float(base["val_mse"]) * mse_guard)
        & (candidates["val_mae"] <= float(base["val_mae"]) * mae_guard)
        & (candidates["lambda_value"].fillna(1.0).astype(float) > 1e-9)
    ]


def summarize_group(group: pd.DataFrame, mse_guard: float, mae_guard: float) -> dict[str, list[dict]]:
    base = identity_base(group)
    if base is None:
        return {name: [] for name in ["main", "control", "external", "ablation", "operator", "runtime"]}
    gate_rows = group[group["variant"] == GATE_VARIANT]
    full = pick_min(gate_rows, base)
    output_only = pick_min(gate_rows[gate_rows["input_action"] == "identity"], base)
    input_only = pick_min(group[(group["variant"] == "base") & (group["input_action"] != "identity")], base)
    fixed_revin = pick_min(group[(group["variant"] == "base") & (group["input_action"] == "revin_mean_1")], base)
    adarevin = pick_min(group[(group["variant"] == "base") & (group["input_action"].map(is_revin) | (group["input_action"] == "identity"))], base)
    no_adarevin = pick_min(gate_rows[~gate_rows["input_action"].map(is_revin)], base)
    correction_pool = group[
        group["variant"].astype(str).str.contains("global_alignment", na=False)
        | group["variant"].astype(str).str.startswith("fixed_lambda")
    ]
    no_admission = pick_min(correction_pool, base, metric="mse")
    no_fallback_pool = admissible(correction_pool, base, mse_guard, mae_guard)
    no_fallback = pick_min(no_fallback_pool, pick_min(correction_pool, base), metric="mse")
    fixed_lambda = pick_min(group[group["variant"].astype(str).str.startswith("fixed_lambda")], base)
    accuracy_only = pick_min(gate_rows, base, metric="mse")

    external_pool = group[
        (group.get("lambda_mode", pd.Series("", index=group.index)).astype(str) == "external_posthoc")
        | (group.get("structure", pd.Series("", index=group.index)).astype(str) == "external_residual")
    ]
    best_external = pick_min(external_pool, base, metric="mse_mae")

    out: dict[str, list[dict]] = {name: [] for name in ["main", "control", "external", "ablation", "operator", "runtime"]}
    out["main"].append(row_record(base, base, "Backbone-only"))
    out["main"].append(row_record(base, full, "Full RASPF"))
    out["control"].append(row_record(base, accuracy_only, "Accuracy-only selector", "policy"))
    out["control"].append(row_record(base, full, "RASPF governor", "policy"))
    out["external"].append(row_record(base, full, "RASPF governor", "baseline"))
    out["external"].append(row_record(base, best_external, "Best external post-hoc", "baseline"))
    for variant, part in external_pool.groupby("variant", dropna=False):
        row = pick_min(part, base, metric="mse_mae")
        out["external"].append(row_record(base, row, str(variant), "baseline"))
    for label, row in [
        ("Output-only", output_only),
        ("Input-only", input_only),
        ("Fixed RevIN", fixed_revin),
        ("AdaRevIN-only", adarevin),
        ("No AdaRevIN", no_adarevin),
        ("No admission", no_admission),
        ("No fallback", no_fallback),
        ("Fixed-lambda", fixed_lambda),
        ("Full RASPF", full),
    ]:
        out["ablation"].append(row_record(base, row, label, "mode"))
    output_candidates = gate_rows[gate_rows["structure"].astype(str) != "none"]
    for structure, part in output_candidates.groupby("structure", dropna=False):
        row = pick_min(part, base, metric="mse_mae")
        out["operator"].append(row_record(base, row, str(structure), "operator"))

    train_sec = pd.to_numeric(group.get("train_eval_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
    calib_sec = pd.to_numeric(group.get("calibration_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
    peak_mem = pd.to_numeric(group.get("cuda_peak_memory_mb", pd.Series(dtype=float)), errors="coerce").dropna()
    refresh_sec = float(calib_sec.max()) if len(calib_sec) else np.nan
    train_eval_sec = float(train_sec.max()) if len(train_sec) else np.nan
    out["runtime"].append(
        {
            "dataset": base["dataset"],
            "horizon": int(base["horizon"]),
            "seed": int(base["seed"]) if not pd.isna(base["seed"]) else 0,
            "backbone": str(base["backbone"]).lower(),
            "train_eval_seconds": train_eval_sec,
            "refresh_seconds": refresh_sec,
            "total_seconds": train_eval_sec + refresh_sec if np.isfinite(train_eval_sec) and np.isfinite(refresh_sec) else np.nan,
            "refresh_overhead_pct": refresh_sec / train_eval_sec * 100.0 if np.isfinite(train_eval_sec) and train_eval_sec > 0 else np.nan,
            "cuda_peak_memory_mb": float(peak_mem.max()) if len(peak_mem) else np.nan,
            "full_mse_gain_pct": gain(base, full, "mse"),
            "full_mae_gain_pct": gain(base, full, "mae"),
        }
    )
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 2026) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def sign_test_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return np.nan
    k = min(wins, losses)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def aggregate_method_table(rows: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    records = []
    for label, part in rows.groupby(label_col, dropna=False):
        mse_ci = bootstrap_ci(part["mse_gain_pct"].to_numpy(dtype=float))
        mae_ci = bootstrap_ci(part["mae_gain_pct"].to_numpy(dtype=float))
        records.append(
            {
                label_col: label,
                "rows": len(part),
                "mse_gain_rows": int((part["mse_gain_pct"] > 0).sum()),
                "mae_gain_rows": int((part["mae_gain_pct"] > 0).sum()),
                "both_metric_gain_rows": int(part["both_win"].sum()),
                "any_metric_harm_rows": int(part["any_harm"].sum()),
                "mean_mse_gain_pct": float(part["mse_gain_pct"].mean()),
                "mse_gain_ci95": f"[{mse_ci[0]:.3f}, {mse_ci[1]:.3f}]",
                "mean_mae_gain_pct": float(part["mae_gain_pct"].mean()),
                "mae_gain_ci95": f"[{mae_ci[0]:.3f}, {mae_ci[1]:.3f}]",
                "mse_sign_p": sign_test_pvalue(int((part["mse_gain_pct"] > 0).sum()), int((part["mse_gain_pct"] < 0).sum())),
                "mae_sign_p": sign_test_pvalue(int((part["mae_gain_pct"] > 0).sum()), int((part["mae_gain_pct"] < 0).sum())),
            }
        )
    return pd.DataFrame(records).sort_values(["mean_mse_gain_pct", "mean_mae_gain_pct"], ascending=False)


def write_outputs(df: pd.DataFrame, out_dir: Path, mse_guard: float, mae_guard: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = {name: [] for name in ["main", "control", "external", "ablation", "operator", "runtime"]}
    group_cols = ["dataset", "horizon", "seed", "backbone"]
    for _, group in df.groupby(group_cols, dropna=False):
        summary = summarize_group(group, mse_guard, mae_guard)
        for key, rows in summary.items():
            buckets[key].extend(rows)

    main = pd.DataFrame(buckets["main"])
    control = pd.DataFrame(buckets["control"])
    external = pd.DataFrame(buckets["external"])
    ablation = pd.DataFrame(buckets["ablation"])
    operator = pd.DataFrame(buckets["operator"])
    runtime = pd.DataFrame(buckets["runtime"])

    main.to_csv(out_dir / "main_backbone_vs_full_raspf_rows.csv", index=False)
    control.to_csv(out_dir / "accuracy_risk_control_rows.csv", index=False)
    external.to_csv(out_dir / "external_posthoc_baseline_rows.csv", index=False)
    ablation.to_csv(out_dir / "component_ablation_rows.csv", index=False)
    operator.to_csv(out_dir / "operator_profile_rows.csv", index=False)
    runtime.to_csv(out_dir / "runtime_memory_cost_rows.csv", index=False)
    df.to_csv(out_dir / "all_candidate_rows.csv", index=False)

    if not main.empty:
        full = main[main["method"] == "Full RASPF"]
        by_dataset = (
            full.groupby(["backbone", "dataset"], dropna=False)
            .agg(
                rows=("horizon", "count"),
                both_metric_wins=("both_win", "sum"),
                non_both_win_rows=("both_win", lambda x: int(len(x) - x.sum())),
                base_mse=("base_mse", "mean"),
                base_mae=("base_mae", "mean"),
                raspf_mse=("mse", "mean"),
                raspf_mae=("mae", "mean"),
                mean_mse_gain_pct=("mse_gain_pct", "mean"),
                mean_mae_gain_pct=("mae_gain_pct", "mean"),
            )
            .reset_index()
        )
        by_dataset.to_csv(out_dir / "main_by_backbone_dataset.csv", index=False)
        aggregate_method_table(full.rename(columns={"method": "policy"}), "policy").to_csv(out_dir / "full_raspf_statistical_summary.csv", index=False)
    aggregate_method_table(control, "policy").to_csv(out_dir / "accuracy_risk_control_summary.csv", index=False)
    external_summary = aggregate_method_table(external, "baseline")
    external_summary.to_csv(out_dir / "external_posthoc_baseline_summary.csv", index=False)
    if not external_summary.empty:
        compact_names = {
            "RASPF governor",
            "Best external post-hoc",
            "residual_mean_recenter",
            "residual_median_recenter",
            "tuned_seasonal_residual_recenter",
            "tuned_residual_ewma_recenter",
            "tuned_residual_ridge_recenter",
            "tuned_local_analog_residual_recenter",
            "tuned_global_ridge_stack",
            "tuned_validation_weighted_stack",
        }
        external_summary[external_summary["baseline"].isin(compact_names)].to_csv(
            out_dir / "external_posthoc_baseline_compact.csv", index=False
        )
    aggregate_method_table(ablation, "mode").to_csv(out_dir / "component_ablation_summary.csv", index=False)
    aggregate_method_table(operator, "operator").to_csv(out_dir / "operator_profile_summary.csv", index=False)
    if not runtime.empty:
        runtime.groupby("backbone", dropna=False).agg(
            train_eval_seconds=("train_eval_seconds", "mean"),
            refresh_seconds=("refresh_seconds", "mean"),
            total_seconds=("total_seconds", "mean"),
            refresh_overhead_pct=("refresh_overhead_pct", "mean"),
            cuda_peak_memory_mb=("cuda_peak_memory_mb", "max"),
            mean_mse_gain_pct=("full_mse_gain_pct", "mean"),
            mean_mae_gain_pct=("full_mae_gain_pct", "mean"),
        ).reset_index().to_csv(out_dir / "runtime_memory_cost_summary.csv", index=False)

    readme = [
        "# Generated EAAI Service Experiment Tables",
        "",
        "- `main_backbone_vs_full_raspf_rows.csv`: paired Backbone-only and Full RASPF rows.",
        "- `main_by_backbone_dataset.csv`: compact manuscript-style aggregation.",
        "- `accuracy_risk_control_summary.csv`: accuracy-only selector vs full governor.",
        "- `external_posthoc_baseline_summary.csv`: fair frozen-checkpoint post-hoc baselines.",
        "- `external_posthoc_baseline_compact.csv`: compact main-text external baseline view.",
        "- `component_ablation_summary.csv`: input/output/gate/fallback/service-mode ablations.",
        "- `operator_profile_summary.csv`: output proposal behavior.",
        "- `runtime_memory_cost_summary.csv`: refresh overhead and memory profile.",
        "",
        f"Rows read from `{df['source_file'].nunique() if 'source_file' in df.columns else 0}` result files.",
    ]
    (out_dir / "README_generated_tables.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[summarize] wrote {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EAAI RASPF service-layer experiments.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mse-guard", type=float, default=1.15)
    parser.add_argument("--mae-guard", type=float, default=1.08)
    args = parser.parse_args()
    df = read_rows(args.run_root)
    if df.empty:
        raise SystemExit(f"No dual_interface_ablation_rows.csv found under {args.run_root}")
    out_dir = args.out_dir or args.run_root / "summary"
    write_outputs(df, out_dir, args.mse_guard, args.mae_guard)


if __name__ == "__main__":
    main()
