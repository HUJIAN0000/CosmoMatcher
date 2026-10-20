# CosmoMatcher v1.4.3

Constrained cosmological tracer assignment with covariance preservation.

This release accompanies the manuscript **CosmoMatcher: constrained cosmological tracer assignment with covariance preservation** by Jian Hu, Yi Liu, Chen Wang, Jian-Ping Hu, Jia Zhao and Liangqing Zhu. The manuscript is prepared for RAA; this repository does not assert journal acceptance or an assigned paper DOI.

## What it does

For a design requiring one non-reused physical SN per target anchor, two MILP stages first maximize complete target coverage and then minimize squared distance mismatch. Physical-ID resource constraints apply across both lens/source roles. A catalogue-order rule resolves equal-cost assignments within a declared numerical tolerance. Retained parent indices support auditable covariance extraction.

Allocation optimality is conditional on the declared catalogue, constraints and successful optimization. It does not guarantee optimal cosmological precision. The distance window uses a fiducial flat LCDM background and is not itself cosmology-independent. Covariance sub-matrix indexing is standard; the software provides ordering, identity and propagation safeguards.

## Install and run

Python 3.12.12 on CPU/float64 was used for this packaged validation. Install the minimum dependencies from `requirements.txt`; `requirements-tested.txt` records the numerical versions used, not a promise that every other version has been tested. A desktop Python with Tk support is required for the GUI. No executable binary is included.

```bash
python -m pip install -r requirements.txt
python CosmoMatcher.py
python CosmoMatcher.py run --config examples/demo_sgl.json
```

The CLI and scientific API can be used without opening the GUI. Synthetic examples cover paired anchors, single anchors and JLA covariance components.

## Repository layout

| Path | Purpose |
| --- | --- |
| `CosmoMatcher.py` | v1.4.3 GUI, CLI and backend |
| `examples/`, `tests/` | Synthetic examples and 46 backend regression tests |
| `data/` | Exact 161-lens and Pantheon+ input snapshots |
| `reproduce.py`, `reproduce.json` | Rerun the paper's 112-system reference matching |
| `verify_reference.py` | Independent optimization/index/covariance checks |
| `reference_results/` | Frozen reference outputs; historical paths in manifests describe their origin |
| `paper_figures/` | Eight adopted manuscript PDF figures |
| `benchmarks/` | Observed GLS comparison and known-truth Monte Carlo study |
| `applications/cddr143_application/` | Adopted cddv301 two-case CDDR example and saved chains |
| `docs/` | Data attribution, manuscript correspondence, release and limitations notes |

## Reproduce the paper

From the repository root:

```bash
python -m unittest discover -s tests -v
python verify_reference.py
python reproduce.py
```

The reference run uses 161 lenses and 1701 Pantheon+ light-curve records, yielding 112 systems and 224 distinct physical SN identifiers. `reproduce.py` writes `rerun_results/` and compares the output with the frozen reference.

For simulations:

```bash
cd benchmarks
python -m unittest discover -s tests -v
python cosmomatcher_benchmark.py --config benchmark_config.json --output baseline_rerun
python cosmomatcher_benchmark.py --config stress_config.json --output stress_rerun
python reproduce_observed_gls.py --output observed_rerun
python make_paper_figures.py --results stress_rerun --output figures_rerun
```

There are 10,000 paired realizations per configuration and nine tolerance/truth-background configurations. At 5% tolerance linear GLS reduces distance-ratio RMSE with near-nominal coverage on the shared targets; constant GLS undercovers. At 3%, poorly supported linear extrapolation can have a heavy-tailed ratio distribution. These are distance-ratio simulations, not cosmological-parameter coverage tests. The benchmark's own version `1.0.0` identifies the experiment driver; its matching backend is v1.4.3.

For the CDDR application (from the repository root):

```bash
python -m pip install -r applications/cddr143_application/requirements.txt
python applications/cddr143_application/reproduce.py
python applications/cddr143_application/reproduce_matching.py
```

The first command after installation verifies frozen inputs and covariance indices; the second reruns matching and compares the 99-system result. To rerun inference, first copy the application directory to a working location, then run `python reproduce.py --sample` there: this replaces the copied `runs/` and `output/` products with new results. The archived reference chains are already supplied.

## Keep these samples distinct

| Purpose | Inputs | Matched systems |
| --- | --- | --- |
| Main software reference | 161 lenses; 1701 Pantheon+ records | 112 |
| Current CDDR revisit | 143 SS3 lenses from the final 204-lens compilation; 1048 original Pantheon SNe | 99 |
| Zhou 2021 historical analysis | 205-lens arXiv-era compilation; 1048 original Pantheon SNe; greedy matching | 120 |
| Separate historical companion sample | 130 selected lenses | 107; not the main reference in this release |

The current CDDR example revisits a published analysis to incorporate previously omitted SN systematic covariance and globally constrained allocation. Lens preselection also changed, so 120 versus 99 is not a fixed-input greedy/MILP performance comparison.

## Covariance and CDDR limitations

Pantheon+ `STAT+SYS` is a total statistical-plus-systematic covariance; do not add its statistical diagonal again. Original Pantheon C11 is systematic-only. The CDDR example adds the mB statistical variance once and retains LL, SS, LS and transpose systematic blocks.

The manuscript adopts cddv301, with identical 99 systems, host-mass errors (including error=99), priors and 140 walkers/1000 steps in both cases. It compares only inclusion/omission of C11 systematic covariance. Within-SN SALT2 statistical cross-covariances are omitted; the original parameter-dependent chi-square weighting has no covariance-determinant factor. Both chains fail the original 50-autocorrelation-time length criterion. This is a reproducible workflow example, not a convergence-validated precision CDDR measurement. cddv302 and other exploratory models are not included.

## Accepted companion use

The manuscript also cites the RAA-accepted lens--quasar--cluster application, arXiv:2610.26097. See `docs/ACCEPTED_APPLICATION.md` for its relation to the allocation design and the scope of validation.

## Cite, data and rights

Use `CITATION.cff` for the software and cite the original data papers listed in `docs/DATA_SOURCES.md`. No paper DOI or new Zenodo DOI is invented here. Existing v1.0/v1.1 archives describe their own samples and cannot automatically be cited as immutable copies of this release.

No software license grant has been invented during packaging. See `docs/LICENSING_STATUS.md`; the copyright holder should choose a software license before describing this as a licensed open-source release. Third-party data and the journal class retain their respective rights.
