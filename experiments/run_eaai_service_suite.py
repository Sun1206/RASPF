from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT = Path(os.environ.get("RASPF_PROJECT", Path(__file__).resolve().parents[1]))
EXP = PROJECT / "experiments"
PY = os.environ.get("PYTHON", sys.executable)

DEFAULT_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic"]
DEFAULT_BACKBONES = ["xlinear", "dtaf", "phaseformer"]
DEFAULT_HORIZONS = [96, 192, 336, 720]
WIDE_DATASETS = {"Electricity", "Traffic"}


@dataclass(frozen=True)
class Job:
    name: str
    dataset: str
    horizon: int
    backbone: str
    seed: int
    cmd: list[str]
    out_dir: Path


def horizons_for(dataset: str) -> list[int]:
    return DEFAULT_HORIZONS


def input_len_for(dataset: str, horizon: int) -> int:
    if dataset in WIDE_DATASETS:
        return 96
    return 512


def batch_size_for(dataset: str, backbone: str, smoke: bool) -> int:
    if smoke:
        return 64
    if dataset == "Traffic":
        return 16
    if dataset == "Electricity":
        return 32 if backbone in {"dtaf", "phaseformer"} else 64
    if dataset == "Weather":
        return 256
    if backbone == "dtaf":
        return 384
    return 512


def epochs_for(backbone: str, dataset: str, smoke: bool) -> int:
    if smoke:
        return 1
    if dataset in WIDE_DATASETS:
        return 8
    if backbone in {"dtaf", "phaseformer"}:
        return 16
    return 20


def num_workers_for(dataset: str, smoke: bool) -> int:
    if smoke or dataset in {"Weather", "Traffic"}:
        return 0
    return 4


def use_pin_memory(dataset: str, smoke: bool) -> bool:
    return (not smoke) and dataset not in {"Weather", "Traffic"}


def window_limit_args(dataset: str, smoke: bool) -> list[str]:
    if smoke:
        return [
            "--max-train-windows",
            "128",
            "--max-val-windows",
            "64",
            "--max-test-windows",
            "64",
        ]
    if dataset == "Electricity":
        return [
            "--max-train-windows",
            "4096",
            "--max-val-windows",
            "768",
            "--max-test-windows",
            "768",
            "--eval-stride",
            "4",
        ]
    if dataset == "Traffic":
        return [
            "--max-train-windows",
            "2048",
            "--max-val-windows",
            "256",
            "--max-test-windows",
            "256",
            "--eval-stride",
            "8",
        ]
    return []


def input_actions(smoke: bool) -> str:
    if smoke:
        return "identity,smooth5,revin_mean_05"
    return ",".join(
        [
            "identity",
            "winsor_iqr",
            "smooth5",
            "robust_scale",
            "revin_mean_1",
            "revin_mean_05",
            "revin_mean_025",
            "revin_median_1",
            "revin_median_05",
            "revin_ema_1",
            "revin_ema_05",
        ]
    )


def structures_for(dataset: str, smoke: bool) -> str:
    if smoke:
        return "phase,winsor_phase"
    if dataset in WIDE_DATASETS:
        return "phase,robust_phase,winsor_phase,shrink_phase"
    return "phase,robust_phase,winsor_phase,shrink_phase,corr_phase"


def fixed_weights(smoke: bool) -> str:
    return "0.2" if smoke else "0.2,0.4,0.6"


def model_args(backbone: str) -> list[str]:
    if backbone == "xlinear":
        return ["--d-model", "128", "--depth", "3"]
    if backbone == "dtaf":
        return [
            "--d-model",
            "32",
            "--depth",
            "1",
            "--dtaf-layers",
            "1",
            "--dtaf-heads",
            "2",
            "--dtaf-expert-num",
            "2",
        ]
    if backbone == "phaseformer":
        return [
            "--d-model",
            "32",
            "--depth",
            "1",
            "--phaseformer-latent-dim",
            "32",
            "--phaseformer-layers",
            "1",
            "--phaseformer-heads",
            "4",
            "--phaseformer-routers",
            "8",
        ]
    raise ValueError(f"Unsupported EAAI backbone: {backbone}")


def build_command(
    dataset: str,
    horizon: int,
    backbone: str,
    seed: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        PY,
        "-u",
        str(EXP / "run_dual_interface_ablation.py"),
        "--model",
        backbone,
        "--datasets",
        dataset,
        "--pred-lens",
        str(horizon),
        "--seeds",
        str(seed),
        "--data-dir",
        str(EXP / "data"),
        "--out-dir",
        str(out_dir),
        "--input-len",
        str(input_len_for(dataset, horizon)),
        "--epochs",
        str(epochs_for(backbone, dataset, args.smoke)),
        "--patience",
        "2" if args.smoke else "5",
        "--batch-size",
        str(batch_size_for(dataset, backbone, args.smoke)),
        "--num-workers",
        str(num_workers_for(dataset, args.smoke)),
        "--input-actions",
        input_actions(args.smoke),
        "--structures",
        structures_for(dataset, args.smoke),
        "--fixed-weights",
        fixed_weights(args.smoke),
        "--lambda-max",
        "1.0",
        "--emit-external-baselines",
        "--allow-tf32",
        "--cuda-benchmark",
        "--device",
        args.device,
    ]
    if use_pin_memory(dataset, args.smoke):
        cmd.append("--pin-memory")
    if args.external_baselines_on_cpu:
        cmd.append("--external-baselines-on-cpu")
    cmd.extend(window_limit_args(dataset, args.smoke))
    cmd.extend(model_args(backbone))
    return cmd


def build_jobs(args: argparse.Namespace) -> list[Job]:
    datasets = args.datasets or (["ETTh1"] if args.smoke else DEFAULT_DATASETS)
    backbones = args.backbones or (["xlinear"] if args.smoke else DEFAULT_BACKBONES)
    horizons = args.horizons or ([96] if args.smoke else DEFAULT_HORIZONS)
    seeds = args.seeds or [2026]
    jobs: list[Job] = []
    for dataset in datasets:
        for horizon in horizons:
            for backbone in backbones:
                for seed in seeds:
                    name = f"{backbone}_{dataset}_h{horizon}_seed{seed}"
                    out_dir = args.run_root / name
                    jobs.append(
                        Job(
                            name=name,
                            dataset=dataset,
                            horizon=horizon,
                            backbone=backbone,
                            seed=seed,
                            cmd=build_command(dataset, horizon, backbone, seed, out_dir, args),
                            out_dir=out_dir,
                        )
                    )
    return jobs


def result_exists(job: Job) -> bool:
    rows = job.out_dir / "dual_interface_ablation_rows.csv"
    summary = job.out_dir / "dual_interface_ablation_summary.csv"
    return rows.exists() and rows.stat().st_size > 256 and summary.exists()


def write_manifest(jobs: list[Job], run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "job_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "dataset", "horizon", "backbone", "seed", "out_dir", "command"],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "name": job.name,
                    "dataset": job.dataset,
                    "horizon": job.horizon,
                    "backbone": job.backbone,
                    "seed": job.seed,
                    "out_dir": str(job.out_dir),
                    "command": " ".join(quote_cmd(job.cmd)),
                }
            )
    sh_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    ps_lines = ["$ErrorActionPreference = 'Stop'"]
    for job in jobs:
        sh_lines.append(" ".join(quote_cmd(job.cmd)))
        ps_lines.append(" ".join(quote_cmd(job.cmd, powershell=True)))
    (run_root / "commands.sh").write_text("\n".join(sh_lines) + "\n", encoding="utf-8")
    (run_root / "commands.ps1").write_text("\n".join(ps_lines) + "\n", encoding="utf-8")


def quote_cmd(cmd: list[str], powershell: bool = False) -> list[str]:
    quoted = []
    for item in cmd:
        s = str(item)
        if not s or any(ch.isspace() for ch in s) or any(ch in s for ch in ['"', "'", "(", ")"]):
            if powershell:
                quoted.append("'" + s.replace("'", "''") + "'")
            else:
                quoted.append("'" + s.replace("'", "'\"'\"'") + "'")
        else:
            quoted.append(s)
    return quoted


def gpu_snapshot() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return out.strip().replace("\n", " | ")
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def log(run_root: Path, msg: str) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%F %T')}] {msg}"
    with (run_root / "master.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def summarize(run_root: Path) -> None:
    cmd = [
        PY,
        "-u",
        str(EXP / "summarize_eaai_service_suite.py"),
        "--run-root",
        str(run_root),
        "--out-dir",
        str(run_root / "summary"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{EXP}{os.pathsep}{PROJECT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, cwd=str(PROJECT), env=env, check=False)


def run_jobs(jobs: list[Job], args: argparse.Namespace) -> int:
    log_dir = args.run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not result_exists(job)]
    running: list[tuple[Job, subprocess.Popen]] = []
    failed = 0
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{EXP}{os.pathsep}{PROJECT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("OMP_NUM_THREADS", str(args.omp_threads))
    env.setdefault("MKL_NUM_THREADS", str(args.omp_threads))
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    log(args.run_root, f"start pending={len(pending)}/{len(jobs)} max_parallel={args.max_parallel} gpu={gpu_snapshot()}")
    last_summary = time.time()
    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            stdout = (log_dir / f"{job.name}.log").open("a", encoding="utf-8")
            stderr = (log_dir / f"{job.name}.err").open("a", encoding="utf-8")
            log(args.run_root, f"START {job.name}")
            proc = subprocess.Popen(job.cmd, cwd=str(PROJECT), env=env, stdout=stdout, stderr=stderr, text=True)
            running.append((job, proc))
            time.sleep(args.launch_gap_seconds)
        time.sleep(args.heartbeat_seconds)
        still: list[tuple[Job, subprocess.Popen]] = []
        for job, proc in running:
            rc = proc.poll()
            if rc is None:
                still.append((job, proc))
            else:
                log(args.run_root, f"END {job.name} rc={rc}")
                if rc != 0:
                    failed += 1
        running = still
        done = sum(1 for job in jobs if result_exists(job))
        log(
            args.run_root,
            f"heartbeat done={done}/{len(jobs)} running={[job.name for job, _ in running]} "
            f"pending={len(pending)} failed={failed} gpu={gpu_snapshot()}",
        )
        if time.time() - last_summary >= args.summary_interval_seconds:
            summarize(args.run_root)
            last_summary = time.time()
    summarize(args.run_root)
    log(args.run_root, f"finished failed={failed} gpu={gpu_snapshot()}")
    return failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EAAI fixed-checkpoint RASPF service-layer suite.")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=None)
    parser.add_argument("--backbones", nargs="+", choices=DEFAULT_BACKBONES, default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--omp-threads", type=int, default=4)
    parser.add_argument("--launch-gap-seconds", type=int, default=10)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--summary-interval-seconds", type=int, default=900)
    parser.add_argument("--external-baselines-on-cpu", action="store_true")
    args = parser.parse_args()
    if args.run_root is None:
        args.run_root = EXP / "eaai_service_runs" / ("smoke" if args.smoke else "full")
    if not args.plan_only and not args.run:
        args.plan_only = True
    return args


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    write_manifest(jobs, args.run_root)
    print(f"[eaai-suite] jobs={len(jobs)} manifest={args.run_root / 'job_manifest.csv'}", flush=True)
    if args.plan_only and not args.run:
        return
    raise SystemExit(run_jobs(jobs, args))


if __name__ == "__main__":
    main()

