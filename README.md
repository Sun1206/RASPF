# RASPF
Reliability-Aware Service Policy for Fixed-Checkpoint Time-Series Forecasting

This package contains the code, fixed result cache, table data, and figure assets for the manuscript **Reliability-Aware Service Policy for Fixed-Checkpoint Time-Series Forecasting**.

## What is included

- `experiments/`: minimal experiment scripts used for the fixed-checkpoint RASPF service-layer study.
- `scripts/`: lightweight scripts for rebuilding paper tables from the fixed 84-row result cache.
- `results/paper_tables/`: final synchronized table cache used by the manuscript.
- `results/raw_cache/`: retained server-run cache for audit and rerun reference; not the authoritative final table source.
- `results/paper_tables/`: CSV files corresponding to the manuscript tables.
- `figures/`: Figure 1 PNG and draw.io source.
- `data/`: dataset manifest and preparation notes.

## Quick check

```bash
python scripts/smoke_check_results.py
```

## Rebuild paper tables

```bash
python scripts/build_paper_tables.py
```

This copies the final synchronized manuscript-table cache to `results/rebuilt_tables/`.

## Figure and table mapping

| Manuscript item | Source artifact | Rebuild command |
|---|---|---|
| Figure 1 | `figures/Figure_1_RASPF_policy_layer.drawio`, `figures/Figure_1_RASPF_policy_layer.png` | edit/export with draw.io |
| Table 1 output-action dictionary | manuscript LaTeX source | fixed definitions in Section 4 |
| Table 2 service modes | manuscript LaTeX source | fixed service-mode definitions |
| Main grouped result table | `results/paper_tables/table4_main_backbone_dataset_results.csv` | `python scripts/build_paper_tables.py` |
| Row-level statistics | `results/paper_tables/table_row_statistics.csv` | `python scripts/build_paper_tables.py` |
| DTAF ablation table | `results/paper_tables/table5_dtaf_ablation_by_dataset.csv` | from `adaptive_revin_study.csv` cache |
| Output-proposal profile | `results/paper_tables/table6_operator_profile.csv` | from `main_backbone_vs_full_raspf.csv` cache |
| Runtime/memory table | `results/paper_tables/table7_runtime_memory.csv` | from `runtime_memory_profile.csv` cache |
| Supplementary row/statistics tables | `results/paper_tables/*.csv` | `python scripts/build_paper_tables.py` |

## Full training/replay campaign

The final table cache in `results/paper_tables/` is sufficient for manuscript table reproduction. The retained `results/raw_cache/` directory is an audit/rerun reference from the server campaign and is not the authoritative source for the final manuscript counts. To rerun the service-layer experiments, place the raw datasets listed in `data/DATA_MANIFEST.csv` under `data/raw/` and use the experiment entry points:

```bash
python experiments/run_eaai_service_suite.py --help
python experiments/run_generic_backbone_raspf_protocol.py --help
python experiments/run_dual_interface_ablation.py --help
python experiments/summarize_eaai_service_suite.py --help
```

The original server campaign used a single seed (`2026`) and fixed train/validation/test splits for all reported rows.

## Notes for GitHub upload

`Traffic.csv` exceeds GitHub's normal 100 MB file limit. Use Git LFS or keep raw datasets outside the repository and record their checksums using `data/DATA_MANIFEST.csv`.
