# Paper benchmarks

This directory is a self-contained copy of the adopted distance-ratio benchmark, with relative data paths and a frozen v1.4.3 matching backend. The experiment driver has its own version 1.0.0; that is not the CosmoMatcher version.

From this directory run:

```bash
python -m unittest discover -s tests -v
python cosmomatcher_benchmark.py --config benchmark_config.json --output baseline_rerun
python cosmomatcher_benchmark.py --config stress_config.json --output stress_rerun
python reproduce_observed_gls.py --output observed_rerun
python make_paper_figures.py --results stress_rerun --output figures_rerun
```

`sample_results/`, `stress_results/`, `observed_results/` and `paper_figures/` are frozen paper references. Use the separate rerun output directories above to preserve them. For quick checking use `--n-sim 100` and a separate output directory; that is a smoke check and does not reproduce the paper's 10,000-realization precision. `VALIDATION.md` records earlier numerical validation, not a new claim of GUI or cosmological-posterior validation. Tests for the primary backend are in the repository root `tests/`.
