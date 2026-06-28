"""Dataset loading utilities for the RASPF EAAI reproducibility package.

The experiment scripts in this package share a small data interface: a loaded
dataset has a name plus train/validation/test NumPy arrays.  This module keeps
that interface self-contained so the released scripts can run without relying
on files outside the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


DATASET_FILES = {
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "Weather": "Weather.csv",
    "Electricity": "Electricity.csv",
    "Traffic": "Traffic.csv",
}

DATASET_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}


def _candidate_paths(name: str, data_dir: Path) -> list[Path]:
    filename = DATASET_FILES[name]
    return [
        data_dir / filename,
        data_dir / "raw" / filename,
        Path("data") / filename,
        Path("data") / "raw" / filename,
    ]


def _read_numeric_csv(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError(f"No numeric forecasting columns found in {path}.")
    return numeric.to_numpy(dtype=np.float32)


def _ett_splits(name: str) -> tuple[int, int]:
    if name.startswith("ETTh"):
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
    elif name.startswith("ETTm"):
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
    else:
        raise ValueError(name)
    return train_end, val_end


def _standard_splits(name: str, n: int) -> tuple[int, int]:
    if name.startswith("ETT"):
        train_end, val_end = _ett_splits(name)
        if val_end < n:
            return train_end, val_end
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    return train_end, val_end


def load_dataset(name: str, data_dir: str | Path) -> DatasetBundle:
    if name not in DATASET_FILES:
        raise KeyError(f"Unknown dataset {name!r}. Expected one of {sorted(DATASET_FILES)}.")
    data_dir = Path(data_dir)
    found = next((p for p in _candidate_paths(name, data_dir) if p.exists()), None)
    if found is None:
        searched = "\n  - ".join(str(p) for p in _candidate_paths(name, data_dir))
        raise FileNotFoundError(
            f"Missing {DATASET_FILES[name]} for {name}. Place the raw CSV under one of:\n"
            f"  - {searched}\n"
            "Run `python experiments/run_sdrc_experiments.py --download --data-dir data/raw` "
            "to fetch the ETT files. Weather, Electricity, and Traffic should be obtained "
            "from the standard long-horizon forecasting benchmark release and named as in "
            "`data/DATA_MANIFEST.csv`."
        )
    arr = _read_numeric_csv(found)
    train_end, val_end = _standard_splits(name, len(arr))
    if not (0 < train_end < val_end < len(arr)):
        raise ValueError(f"Invalid split for {name}: n={len(arr)} train_end={train_end} val_end={val_end}.")
    return DatasetBundle(name=name, train=arr[:train_end], val=arr[train_end:val_end], test=arr[val_end:])


def download_datasets(data_dir: str | Path) -> None:
    """Download the small ETT CSV files; large benchmark files are documented."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in DATASET_URLS.items():
        out = data_dir / DATASET_FILES[name]
        if out.exists():
            print(f"[data] exists: {out}")
            continue
        print(f"[data] downloading {name} -> {out}")
        urllib.request.urlretrieve(url, out)
    print(
        "[data] ETT files are ready. Weather, Electricity, and Traffic are larger "
        "benchmark files; place them under data/raw/ using the names in DATA_MANIFEST.csv."
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.download:
        download_datasets(args.data_dir)
    if args.check:
        for name in DATASET_FILES:
            try:
                bundle = load_dataset(name, args.data_dir)
                print(
                    f"{name}: train={bundle.train.shape} val={bundle.val.shape} "
                    f"test={bundle.test.shape}"
                )
            except Exception as exc:  # noqa: BLE001 - command-line diagnostic.
                print(f"{name}: {exc}")


if __name__ == "__main__":
    main()
