# CosmoMatcher v1.0

**A Global Optimization Pipeline for Multi-probe Cosmological Data Matching and Covariance Extraction**

[![Python 3.10](https://img.shields.io/badge/python-3.10.19-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OS: Windows](https://img.shields.io/badge/OS-Windows-green.svg)](https://www.microsoft.com/windows)

**CosmoMatcher** is a professional, GUI-based scientific software designed to resolve the redshift-mismatch bottleneck in model-independent cosmology. It leverages Mixed-Integer Linear Programming (MILP) to identify the globally optimal matching between disparate cosmological probes (e.g., Strong Gravitational Lenses, Type Ia Supernovae, and Galaxy Clusters).

---

## 🌟 Key Features
* **Global Optimization (MILP):** Guarantees the theoretical maximum data yield, overcoming the localized "greedy traps" of traditional algorithms.
* **Ablation Study Engine ($2 \times 2$):** Supports seamless switching between two algorithms (**MILP** vs. **Greedy**) and two tolerance criteria (**Comoving Distance $\Delta d/d$** vs. **Static Redshift $\Delta z$**).
* **Automated Covariance Extraction:** Automatically extracts full non-diagonal systematic covariance matrices and cross-covariance terms for matched subsamples.
* **Multi-Probe Support:** Compatible with Pantheon+, JLA, SGLS, and Galaxy Cluster datasets.

---

## 📁 Repository Contents
* `CosmoMatcher_v1.0.py`: The core Python source code with a full English GUI (Based on v6.2).
* `161sgls.csv`: Target catalog of 161 strong gravitational lensing systems from Chen et al. (2019).
* `Pantheon+SH0ES.dat`: Tracer catalog of Pantheon+ Type Ia supernovae.
* `Pantheon+SH0ES_STAT+SYS.zip`: The full systematic covariance matrix for Pantheon+ (Compressed).

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
Install the strictly versioned dependencies required for calculation, GUI, and compilation:
```bash
pip install numpy==2.2.6 pandas==2.3.3 scipy==1.15.3 pyinstaller==6.17.0
```

### 3. Compile to Standalone Executable (.exe)
To generate a single `.exe` file for Windows distribution:
```bash
pyinstaller --onefile --windowed --name CosmoMatcher_v1.0 CosmoMatcher_v1.0.py
```
The compiled file `CosmoMatcher_v1.0.exe` will be located in the `dist` folder.

---

## 🖥️ User Guide for Windows

1. **Prepare Data**: Unzip the `Pantheon+SH0ES_STAT+SYS.zip` file into your working folder.
2. **Launch**: Double-click `CosmoMatcher_v1.0.exe` (or run `python CosmoMatcher_v1.0.py` in your terminal).
3. **Configurations**:
   - **Algorithm**: Select `MILP (Optimal)` for global optimization results.
   - **Criterion**: Select `Δd/d (Comoving Distance)` for physical consistency.
4. **Data Mapping**: 
   - Load `161sgls.csv` as the Target and `Pantheon+SH0ES.dat` as the Tracer.
   - Map redshift columns (e.g., `zHD` for SN, `zl` and `zs` for SGL).
5. **Run**: Click **"Start Matching and Extract Covariances"**. The results (matched catalog and sub-covariance matrices) will be exported to your directory.

---

## 📖 Citation

If you use **CosmoMatcher** in your research, please cite the following paper:

```bibtex
@article{Hu2026CosmoMatcher,
  title={CosmoMatcher: A Global Optimization Pipeline for Multi-probe Cosmological Data Matching and Covariance Extraction},
  author={Hu, Jian and Liu, Yi and Hu, Jian-Ping and Li, Zhongmu},
  journal={The Astrophysical Journal Supplement Series},
  year={2026},
  note={Submitted}
}
```

---

## ✉️ Contact
**Jian Hu** - [dg1626002@smail.nju.edu.cn]  
*Institute of Astronomy and Information, Dali University*

