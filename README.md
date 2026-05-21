# CosmoMatcher v1.1

**A Global Optimization Pipeline for Multi-probe Cosmological Data Matching and Covariance Extraction**

[![Python 3.10](https://img.shields.io/badge/python-3.10.19-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OS: Windows](https://img.shields.io/badge/OS-Windows-green.svg)](https://www.microsoft.com/windows)
[![Latest: v1.1](https://img.shields.io/badge/release-v1.1-brightgreen.svg)](https://github.com/HUJIAN0000/CosmoMatcher/releases)

**CosmoMatcher** is a professional, GUI-based scientific software designed to resolve the redshift-mismatch bottleneck in model-independent cosmology. It leverages Mixed-Integer Linear Programming (MILP) to identify the globally optimal matching between disparate cosmological probes (e.g., Strong Gravitational Lenses, Type Ia Supernovae, and Galaxy Clusters), and automatically extracts the corresponding full systematic covariance sub-matrices.

---

## 🌟 Key Features

* **Global Optimization (MILP):** Guarantees the theoretical maximum data yield, overcoming the localized "greedy traps" of traditional algorithms.
* **Ablation Study Engine ($2 \times 2$):** Supports seamless switching between two algorithms (**MILP** vs. **Greedy**) and two tolerance criteria (**Comoving Distance $\Delta d/d$** vs. **Static Redshift $\Delta z$**).
* **Automated Covariance Extraction:** Automatically extracts full non-diagonal systematic covariance matrices and cross-covariance terms for matched subsamples.
* **Publication-grade Diagnostics (new in v1.1):** Automatically renders the MILP constraint-matrix sparsity pattern and the reconstructed covariance sub-matrix heatmaps to publication-ready PDF/PNG at 600 dpi.
* **Multi-Probe Support:** Compatible with Pantheon+, JLA, SGLS, and Galaxy Cluster datasets.

---

## 📦 What's New in v1.1

* **🐛 Robust Pantheon+ covariance loader.** `load_pantheon_matrix` has been reverted to the original robust implementation that explicitly reads the first-line dimension integer (`n = 1701`). This eliminates the `2893402 element (+1)` size mismatch that could occur in v1.0 on freshly-unzipped Pantheon+ covariance files.
* **📊 Automatic MILP sparsity-pattern plot.** When SGL triplet matching is run under MILP, the pipeline now emits `matrix_A_sparsity_milp_<tol>_<sn>.{pdf,png}` — a publication-grade rendering of the constraint matrix $A$ with the SGL/SNe-Ia constraint blocks annotated. This is the source figure used as Figure 1 in the companion paper.
* **🎨 Automatic covariance-heatmap rendering.** After covariance extraction, every reconstructed sub-matrix is automatically rendered as a heatmap (`plot_cov_<mode>_<algo>_<tol>_<sn>_<suffix>_<key>.{pdf,png}`) with the colorbar labelled $C_{ij}$, suitable for direct inclusion in manuscripts.
* **🗂️ Disambiguated output filenames.** Output `file_tag` now includes the SN-sample type (`pantheon` / `jla`) so that running the same ablation mode under different SN backends no longer overwrites previous results.
* **💻 GUI version bump.** Window title now reads `CosmoMatcher V1.1 (Full Version with Ablation Engine)`.

> 📌 The matching algorithm, the MILP formulation, and the covariance algebra are **unchanged** from v1.0. The reproducible matched dataset archived on Zenodo (DOI: `10.5281/zenodo.19695657`, generated with v1.0) is therefore bit-identical to a v1.1 re-run on the same inputs.

---

## 📁 Repository Contents

* `CosmoMatcher_v1.1.py` — Core Python source code with full English GUI (latest version).
* `CosmoMatcher_v1.0.py` — Previous release, kept for archival reproducibility.
* `161sgls.csv` — Target catalog of 161 strong gravitational lensing systems from Chen et al. (2019).
* `Pantheon+SH0ES.dat` — Tracer catalog of Pantheon+ Type Ia supernovae.
* `Pantheon+SH0ES_STAT+SYS.zip` — Full systematic covariance matrix for Pantheon+ (compressed).

> **⚠️ NOTE:** You must unzip `Pantheon+SH0ES_STAT+SYS.zip` to extract the `.cov` file before loading it into the software.

---

## 🚀 Installation and Compilation (Windows)

To ensure stability and reproducibility on Windows, we recommend using a **Conda** environment with the specific versions used during development.

### 1. Environment Setup

Open your **Anaconda Prompt** and run the following commands:

```bash
conda create -n CosmoMatcher_Env python=3.10.19
conda activate CosmoMatcher_Env
```

### 2. Install Required Packages

Install the strictly versioned dependencies required for calculation, GUI, plotting, and compilation:

```bash
pip install numpy==2.2.6 pandas==2.3.3 scipy==1.15.3 matplotlib==3.10.0 pyinstaller==6.17.0
```

> 📌 `matplotlib` is a **new** required dependency in v1.1 (used by the automatic diagnostic plots). If you are upgrading from a v1.0 environment, you only need to add `matplotlib==3.10.0`.

### 3. Compile to Standalone Executable (.exe)

To generate a single `.exe` file for Windows distribution:

```bash
pyinstaller --onefile --windowed --name CosmoMatcher_v1.1 CosmoMatcher_v1.1.py
```

The compiled file `CosmoMatcher_v1.1.exe` will be located in the `dist` folder.

---

## 🖥️ User Guide for Windows

1. **Prepare Data**: Unzip the `Pantheon+SH0ES_STAT+SYS.zip` file into your working folder.
2. **Launch**: Double-click `CosmoMatcher_v1.1.exe` (or run `python CosmoMatcher_v1.1.py` in your terminal).
3. **Configurations**:
   - **Algorithm**: Select `MILP (Optimal)` for global optimization results.
   - **Criterion**: Select `Δd/d (Comoving Distance)` for physical consistency.
4. **Data Mapping**:
   - Load `161sgls.csv` as the Target and `Pantheon+SH0ES.dat` as the Tracer.
   - Map redshift columns (e.g., `zHD` for SN, `zl` and `zs` for SGL).
5. **Run**: Click **"Start Matching and Extract Covariances"**. The results will be exported to your working directory:
   - Matched catalog: `matched_<mode>_<algo>_<tol>.csv`
   - Sub-covariance matrices: `cov_<mode>_<algo>_<tol>_<sn>_<suffix>_<key>.txt`
   - **(v1.1)** MILP sparsity pattern: `matrix_A_sparsity_milp_<tol>_<sn>.{pdf,png}`
   - **(v1.1)** Covariance heatmaps: `plot_cov_<mode>_<algo>_<tol>_<sn>_<suffix>_<key>.{pdf,png}`

---

## 📜 Version History

| Version | Date | Key Changes |
|--------:|------|-------------|
| **v1.1** | 2026-05 | Robust Pantheon+ loader; automatic MILP sparsity and covariance heatmap plots; disambiguated output filenames; `matplotlib` added as a required dependency. |
| **v1.0** | 2026-05 | Initial public release. 2 × 2 ablation engine (MILP/Greedy × Δd/d/Δz), automated covariance extraction with full JLA 6-component handling, smart CSV reading, custom output-column selection. |

---

## 📖 Citation

If you use **CosmoMatcher** in your research, please cite the companion paper:

```bibtex
@article{Hu2026CosmoMatcher,
  title  = {CosmoMatcher: A Global Optimization Pipeline for Multi-probe
            Cosmological Data Matching and Covariance Extraction},
  author = {Hu, Jian},
  journal= {Publications of the Astronomical Society of the Pacific},
  year   = {2026},
  note   = {Submitted}
}
```

The reproducible research-ready dataset (130-system SGL input following the Chen et al. 2019 exclusions, yielding 107 matched SGL–SN Ia triplets; full $214 \times 214$ Pantheon+ sub-covariance for the 214 unique SN Ia tracers; and the corresponding JLA sub-matrices) is permanently archived on Zenodo:

> **DOI:** [10.5281/zenodo.19695657](https://doi.org/10.5281/zenodo.19695657)

---

## ✉️ Contact

**Jian Hu** — [dg1626002@smail.nju.edu.cn](mailto:dg1626002@smail.nju.edu.cn)
*Institute of Astronomy and Information, Dali University*
