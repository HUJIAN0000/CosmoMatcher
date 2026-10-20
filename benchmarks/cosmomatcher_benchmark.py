#!/usr/bin/env python3
"""Paired Monte Carlo benchmark on Pantheon+ sampling and full covariance.

Run: python cosmomatcher_benchmark.py --config benchmark_config.json
Synthetic magnitudes replace the observed magnitudes; redshifts, CID, covariance
and lens redshifts are retained. No cosmological parameters are fitted.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

VERSION = "1.0.0"
KAPPA = np.log(10.0) / 5.0
PROBABILITIES = (0.68, 0.95)
METHODS = ("single", "gls_constant", "gls_linear")
LABELS = {"single": "Single SN", "gls_constant": "Constant GLS", "gls_linear": "Linear GLS"}


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv_auto(path, **kwargs):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}: use UTF-8 or GB18030 CSV.")


def read_covariance(path, n, symmetry_tolerance=1e-6):
    """Square or row-major flat text, optionally headed by n or n*n."""
    a = np.loadtxt(path, dtype=np.float64)
    if a.ndim == 2 and a.shape == (n, n):
        form = "square"
    else:
        a = a.ravel()
        if a.size == n * n + 1:
            if a[0] not in (n, n * n):
                raise ValueError(f"Covariance header {a[0]} disagrees with {n} catalogue rows.")
            form = "flat_dimension_header" if a[0] == n else "flat_element_count_header"
            a = a[1:]
        elif a.size == n * n:
            form = "flat_no_header"
        else:
            raise ValueError(f"Expected {n*n} covariance values (+ optional header); got {a.size}.")
        a = a.reshape(n, n)
    if not np.isfinite(a).all() or np.any(np.diag(a) <= 0):
        raise ValueError("Covariance must have finite entries and positive variances.")
    asymmetry = float(np.max(np.abs(a - a.T)))
    if asymmetry > symmetry_tolerance:
        raise ValueError(f"Covariance asymmetry {asymmetry:g} exceeds {symmetry_tolerance:g}.")
    a = (a + a.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(a)
    tolerance = 1e-12 * float(np.max(np.diag(a)))
    if eigenvalues[0] < -tolerance:
        raise ValueError(f"Parent covariance is not PSD: min eigenvalue {eigenvalues[0]:g}.")
    return a, {"format": form, "n": n, "max_asymmetry_before_symmetrizing": asymmetry,
               "minimum_eigenvalue": float(eigenvalues[0]),
               "roundoff_symmetrized": bool(asymmetry), "psd_tolerance": tolerance}


def comoving_distance(z, omega_m=0.3, w=-1.0):
    """Dimensionless distance in spatially flat, constant-w cosmology."""
    z = np.asarray(z, dtype=float)
    if not np.isfinite(z).all() or np.any(z < 0):
        raise ValueError("Redshifts must be finite and non-negative.")
    if not 0 < omega_m < 1 or not np.isfinite(w):
        raise ValueError("Truth/matching requires 0 < omega_m < 1 and finite w.")
    unique, inverse = np.unique(z.ravel(), return_inverse=True)
    values = np.array([quad(lambda x: 1.0 / np.sqrt(
        omega_m * (1+x)**3 + (1-omega_m) * (1+x)**(3*(1+w))),
        0.0, float(v), epsabs=2e-12, epsrel=2e-12)[0] for v in unique])
    return values[inverse].reshape(z.shape)


def true_magnitudes(z, omega_m, w, h0, absolute_magnitude):
    dc = comoving_distance(z, omega_m, w)
    dl_mpc = 299792.458 / h0 * (1 + np.asarray(z)) * dc
    return absolute_magnitude + 5*np.log10(dl_mpc) + 25.0


def load_inputs(config):
    cols = config["columns"]
    sn = pd.read_csv(config["sn_file"], sep=r"\s+", dtype={cols["sn_id"]: str})
    lenses = read_csv_auto(config["lens_file"])
    z = pd.to_numeric(sn[cols["sn_z"]], errors="raise").to_numpy(float)
    zl = pd.to_numeric(lenses[cols["lens_z"]], errors="raise").to_numpy(float)
    zs = pd.to_numeric(lenses[cols["source_z"]], errors="raise").to_numpy(float)
    ids = sn[cols["sn_id"]].astype(str).to_numpy()
    if sn[cols["sn_id"]].isna().any() or np.any(ids == ""):
        raise ValueError("Missing physical SN identifier.")
    if not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError("Invalid SN redshift; rows are not silently dropped.")
    if not np.isfinite(zl).all() or not np.isfinite(zs).all() or np.any(zl <= 0) or np.any(zs <= zl):
        raise ValueError("Require finite 0 < zl < zs for every lens.")
    covariance, diagnostics = read_covariance(config["covariance_file"], len(sn))
    return sn, lenses, z, ids, zl, zs, covariance, diagnostics


def validate_selection(table, z, ids, zl, zs, omega_m, tolerance):
    required = ("obj_index", "sn_lens_original_index", "sn_source_original_index")
    if any(c not in table for c in required):
        raise ValueError("Selection CSV needs obj_index, sn_lens_original_index, sn_source_original_index.")
    if table.empty:
        raise ValueError("No selected systems.")
    arrays = []
    for col in required:
        raw = pd.to_numeric(table[col], errors="raise").to_numpy(float)
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError(f"Non-integer catalogue/target index in {col}.")
        arrays.append(raw.astype(int))
    target, jl, js = arrays
    if np.any(target < 0) or np.any(target >= len(zl)) or len(np.unique(target)) != len(target):
        raise ValueError("Invalid or duplicated selected target.")
    if np.any(jl < 0) or np.any(js < 0) or np.any(jl >= len(z)) or np.any(js >= len(z)):
        raise ValueError("Selected row index outside the original SN catalogue.")
    if len(set(ids[np.r_[jl, js]])) != 2*len(target):
        raise ValueError("A physical CID is reused across selected roles/targets.")
    for column, rows in (("sn_lens_CID", jl), ("sn_source_CID", js)):
        if column in table and not np.array_equal(table[column].astype(str).to_numpy(), ids[rows]):
            raise ValueError(f"{column} disagrees with the original catalogue row mapping.")
    if np.any(z[jl] >= z[js]):
        raise ValueError("Selected SN redshifts violate lens/source ordering.")
    d = comoving_distance(z, omega_m)
    dl = comoving_distance(zl[target], omega_m)
    ds = comoving_distance(zs[target], omega_m)
    el, es = np.abs(d[jl]-dl)/dl, np.abs(d[js]-ds)/ds
    if np.any(el > tolerance+1e-11) or np.any(es > tolerance+1e-11):
        raise ValueError("External/cached selection violates this distance window.")
    order = np.argsort(target)
    return target[order], jl[order], js[order], el[order], es[order]


def get_selection(config, case_dir, inputs):
    sn, lenses, z, ids, zl, zs, _, _ = inputs
    fingerprint = {"inputs": {k: sha256(config[k]) for k in
                    ("sn_file", "lens_file", "covariance_file")},
                   "omega_m": config["matching_omega_m"], "tolerance": config["tolerance"],
                   "columns": config["columns"]}
    backend_file = Path(__file__).parent / "backend" / "CosmoMatcher.py"
    fingerprint["backend_sha256"] = sha256(backend_file) if backend_file.exists() else None
    external = config.get("selection_file")
    selection_path = case_dir / "selection.csv"
    meta_path = case_dir / "selection_provenance.json"
    if external:
        selection = read_csv_auto(external)
        metadata = {"fingerprint": fingerprint, "source": "external CSV",
                    "selection_sha256": sha256(external),
                    "optimality": "constraints validated; optimality/canonicality not certified by this benchmark"}
    elif selection_path.exists() and meta_path.exists() and json.loads(meta_path.read_text())["fingerprint"] == fingerprint:
        selection = read_csv_auto(selection_path)
        metadata = json.loads(meta_path.read_text())
        print("Reusing verified selection cache.", flush=True)
    else:
        from backend import CosmoMatcher as cm
        columns = config["columns"]
        cfg = {"mode": "sgls", "sn_type": "pantheon", "algo": "milp", "tol_type": "dd",
               "file_paths": {"tracer": config["sn_file"], "obj": config["lens_file"]},
               "col_cfg": {"tracer_z_col": columns["sn_z"], "tracer_id_col": columns["sn_id"],
                           "z1_col": columns["lens_z"], "z2_col": columns["source_z"]},
               "params": {"wm": config["matching_omega_m"], "tol": config["tolerance"],
                          "unique_by": "auto", "formulation": "compact", "plots": False,
                          "require_covariance": False, "tie_break": "catalogue_order",
                          "tie_cost_tolerance": 1e-12, "invalid_rows": "error",
                          "output_dir": str(case_dir / "matching")}}
        result = cm.run_config(cfg)
        selection = read_csv_auto(result.output_dir / "matched_sgls_milp_dd_pantheon.csv")
        metadata = {"fingerprint": fingerprint, "source": "two-stage MILP plus catalogue-order rule",
                    "backend_version": cm.__version__, "matching_summary": result.summary,
                    "backend_sha256": sha256(Path(cm.__file__))}
    validate_selection(selection, z, ids, zl, zs, config["matching_omega_m"], config["tolerance"])
    selection.to_csv(selection_path, index=False)
    dump_json(meta_path, metadata)
    return selection, metadata


def local_weights(z, covariance, anchors, distance, anchor_distance, tolerance, linear):
    """Local GLS; no independent-window approximation or silent regularization."""
    weights = np.zeros((len(anchors), len(z)))
    counts = np.zeros(len(anchors), int)
    reasons = []
    for i, anchor in enumerate(anchors):
        rows = np.flatnonzero(np.abs(distance-anchor_distance[i])/anchor_distance[i] <= tolerance)
        counts[i] = len(rows)
        if len(rows) < (2 if linear else 1):
            reasons.append("too_few_candidates")
            continue
        x = z[rows] - anchor
        if linear and np.ptp(x) == 0:
            reasons.append("unidentified_slope")
            continue
        # Scaling the predictor changes slope units, not the fitted intercept.
        xscale = np.max(np.abs(x)) if linear else 1.0
        X = np.column_stack((np.ones(len(rows)), x/xscale)) if linear else np.ones((len(rows), 1))
        try:
            factor = cho_factor(covariance[np.ix_(rows, rows)], lower=True, check_finite=False)
            ci_x = cho_solve(factor, X, check_finite=False)
            normal = X.T @ ci_x
            if not np.isfinite(normal).all() or np.linalg.cond(normal) > 1e12:
                reasons.append("ill_conditioned_gls")
                continue
            weights[i, rows] = np.linalg.solve(normal, ci_x.T)[0]
        except np.linalg.LinAlgError:
            reasons.append("non_positive_definite_local_covariance")
            continue
        if not np.isclose(weights[i].sum(), 1, atol=1e-9, rtol=0):
            raise ValueError("GLS intercept does not preserve a constant magnitude offset.")
        reasons.append("ok")
    return weights, counts, np.array(reasons)


@dataclass
class Design:
    name: str
    targets: np.ndarray
    W: np.ndarray
    z_l: np.ndarray
    z_s: np.ndarray
    contrast: np.ndarray
    covariance_anchor: np.ndarray
    covariance_logq: np.ndarray
    stack_slice: slice | None = None


def make_design(name, targets, wl, ws, zl_used, zs_used, covariance):
    W = np.vstack((wl, ws))
    C = W @ covariance @ W.T
    contrast = KAPPA * (wl - ws)
    ct = contrast @ covariance @ contrast.T
    return Design(name, targets, W, zl_used, zs_used, contrast, C, (ct+ct.T)/2)


def build_designs(config, inputs, selection, case_dir):
    sn, lenses, z, ids, zl, zs, C, _ = inputs
    tol, om = config["tolerance"], config["matching_omega_m"]
    targets, jl, js, el, es = validate_selection(selection, z, ids, zl, zs, om, tol)
    wl = np.zeros((len(targets), len(z))); ws = np.zeros_like(wl)
    wl[np.arange(len(targets)), jl] = 1; ws[np.arange(len(targets)), js] = 1
    # The single-SN protocol uses the selected SN redshifts, as in the paper.
    designs = {"single": make_design("single", targets, wl, ws, z[jl], z[js], C)}
    availability = pd.DataFrame({"target_index": np.arange(len(zl)), "zl": zl, "zs": zs})
    namecol = config["columns"].get("lens_name")
    if namecol and namecol in lenses:
        availability["lens_name"] = lenses[namecol].astype(str)
    availability["single_available"] = np.isin(np.arange(len(zl)), targets)
    availability["selected_lens_row"] = -1; availability["selected_source_row"] = -1
    availability.loc[targets, "selected_lens_row"] = jl
    availability.loc[targets, "selected_source_row"] = js
    availability["relative_mismatch_lens"] = np.nan; availability["relative_mismatch_source"] = np.nan
    availability.loc[targets, "relative_mismatch_lens"] = el
    availability.loc[targets, "relative_mismatch_source"] = es
    d = comoving_distance(z, om)
    for name in config["methods"]:
        if name == "single":
            continue
        anchors = np.r_[zl, zs]
        W, counts, reasons = local_weights(z, C, anchors, d, comoving_distance(anchors, om), tol,
                                           linear=name == "gls_linear")
        n = len(zl); valid = (reasons[:n] == "ok") & (reasons[n:] == "ok")
        tt = np.flatnonzero(valid)
        availability[name+"_available"] = valid
        availability[name+"_lens_candidates"] = counts[:n]
        availability[name+"_source_candidates"] = counts[n:]
        availability[name+"_lens_status"] = reasons[:n]
        availability[name+"_source_status"] = reasons[n:]
        if len(tt):
            designs[name] = make_design(name, tt, W[tt], W[n+tt], zl[tt], zs[tt], C)
    common = set(targets)
    for name in config["methods"]:
        common &= set(designs[name].targets) if name in designs else set()
    availability["common_all_methods"] = np.isin(np.arange(len(zl)), sorted(common))
    availability.to_csv(case_dir / "availability.csv", index=False)
    cohorts = {}
    for name in config["methods"]:
        if name != "single" and name in designs:
            tt = np.intersect1d(targets, designs[name].targets)
            if len(tt):cohorts["single_vs_"+name] = {"targets": tt, "methods": ["single", name]}
    if common:
        cohorts["common_all"] = {"targets": np.array(sorted(common), int), "methods": list(config["methods"])}
    if not cohorts:
        raise ValueError("No common targets for single-SN and GLS comparison.")
    for name, design in designs.items():
        np.savez_compressed(case_dir / f"design_{name}.npz", target_index=design.targets,
                            weights=design.W, redshift_lens_used=design.z_l, redshift_source_used=design.z_s,
                            covariance_anchor=design.covariance_anchor,
                            covariance_logq=design.covariance_logq)
    return designs, cohorts, availability


def gaussian_factor(C):
    """PSD factor; retain positive eigenvalues, clip negative roundoff only."""
    eig, vectors = np.linalg.eigh((C+C.T)/2)
    minimum = float(eig[0])
    tolerance = 1e-12 * float(np.max(np.diag(C)))
    if eig[0] < -tolerance:
        raise ValueError(f"Joint propagated covariance is not PSD: {eig[0]:g}.")
    clipped = int(np.sum(eig < 0)); eig = np.maximum(eig, 0)
    positive = eig > 0
    factor = vectors[:, positive] * np.sqrt(eig[positive])
    return factor, {"minimum_eigenvalue_before_roundoff_clipping": minimum,
                    "negative_roundoff_eigenvalues_clipped": clipped,
                    "negative_eigenvalue_tolerance": tolerance,
                    "numerical_rank": int(np.linalg.matrix_rank(C)),
                    "retained_factor_columns": int(positive.sum())}


def analytic_statistics(t0, logq_truth, variance):
    """Independent Gaussian/lognormal reference for MC validation."""
    s = np.sqrt(variance)
    r_true = -np.expm1(logq_truth)
    r0 = -np.expm1(t0)
    mean = -np.expm1(t0 + variance/2)
    variance_r = np.exp(2*t0 + variance) * np.expm1(variance)
    bias = mean - r_true
    result = {"R_true": r_true, "R_noiseless": r0, "transfer_bias": r0-r_true,
              "nonlinear_noise_bias": mean-r0, "bias_analytic": bias,
              "rmse_analytic": np.sqrt(variance_r+bias*bias),
              "variance_R_analytic": variance_r, "sigma_logq": s}
    delta = t0-logq_truth
    positive = s > 0
    for p in PROBABILITIES:
        label = str(int(100*p)); quantile = norm.ppf((1+p)/2)
        cover = np.equal(t0, logq_truth).astype(float)
        cover[positive] = norm.cdf(quantile-delta[positive]/s[positive]) - norm.cdf(-quantile-delta[positive]/s[positive])
        result["coverage"+label+"_log_analytic"] = cover
        u = quantile*s
        lo = np.full_like(s, -np.inf); hi = np.full_like(s, np.inf)
        lo[positive] = (-np.log1p(u[positive])-delta[positive])/s[positive]
        bounded = positive & (u < 1)
        hi[bounded] = (-np.log1p(-u[bounded])-delta[bounded])/s[bounded]
        cv = norm.cdf(hi)-norm.cdf(lo)
        cv[~positive] = np.equal(t0[~positive], logq_truth[~positive])
        result["coverage"+label+"_delta_analytic"] = cv
    return result


def summary_from_replicates(values):
    """MCSE uses independent realizations, never targets as independent trials."""
    return float(values.mean()), float(values.std(ddof=1)/np.sqrt(len(values)))


def simulate_case(config, inputs, designs, cohorts, case_dir):
    sn, lenses, z, ids, zl, zs, C, _ = inputs
    order = list(designs)
    D = np.vstack([designs[m].contrast for m in order])
    joint = D @ C @ D.T
    joint = (joint+joint.T)/2
    factor, diagnostics = gaussian_factor(joint)
    np.savez_compressed(case_dir / "joint_logq_covariance.npz", covariance=joint,
                        contrast=D, factor=factor, method_order=np.array(order))
    start = 0
    for name in order:
        size = len(designs[name].targets)
        designs[name].stack_slice = slice(start, start+size); start += size
    diagnostics["cross_method_noise"] = "one joint draw with covariance D C_parent D^T; exact marginal of the same parent SN noise"
    dump_json(case_dir / "sampling_diagnostics.json", diagnostics)
    summary_rows, target_rows, paired_rows = [], [], []
    n_sim = config["n_sim"]
    for truth_number, truth in enumerate(config["truth_models"]):
        truth_id = truth.get("name", f"truth_{truth_number}")
        print(f"Truth {truth_id}: {n_sim} paired realizations", flush=True)
        omega, w = float(truth["omega_m"]), float(truth.get("w", -1))
        mags = true_magnitudes(z, omega, w, config["h0"], config["absolute_magnitude"])
        logq_true = np.log(comoving_distance(zl, omega, w)/comoving_distance(zs, omega, w))
        state = {}
        for name, design in designs.items():
            tt = design.targets
            t0 = design.contrast @ mags + np.log((1+design.z_s)/(1+design.z_l))
            v = np.diag(design.covariance_logq).copy()
            if np.any(v < -1e-12):raise ValueError("Negative log-ratio variance.")
            v = np.maximum(v, 0)
            analytic = analytic_statistics(t0, logq_true[tt], v)
            state[name] = {"t0": t0, "v": v, "analytic": analytic,
                           "sum_e": np.zeros(len(tt)), "sum_e2": np.zeros(len(tt)),
                           "sum_e4": np.zeros(len(tt)), "nonphysical": np.zeros(len(tt)),
                           **{key: np.zeros(len(tt)) for key in
                              ("coverage68_log", "coverage95_log", "coverage68_delta", "coverage95_delta")}}
        entries = []
        for cohort_name, cohort in cohorts.items():
            for method in cohort["methods"]:
                indices = np.searchsorted(designs[method].targets, cohort["targets"])
                entries.append((cohort_name, method, indices))
        # Columns: mean error, mean squared error, four coverage fractions.
        per_rep = np.empty((n_sim, len(entries), 6))
        rng = np.random.Generator(np.random.PCG64(config["seed"]))
        for first in range(0, n_sim, config["batch_size"]):
            last = min(first+config["batch_size"], n_sim)
            noise = rng.standard_normal((last-first, factor.shape[1])) @ factor.T
            batches = {}
            for name, design in designs.items():
                st = state[name]
                t = st["t0"][None, :] + noise[:, design.stack_slice]
                q = np.exp(t); ratio = -np.expm1(t)
                if not np.isfinite(ratio).all():raise ValueError("Ratio overflow: inspect local weights/noise model.")
                error = ratio-st["analytic"]["R_true"][None, :]
                sd = np.sqrt(st["v"])
                batches[name] = {"e": error, "e2": error*error}
                st["sum_e"] += error.sum(axis=0); st["sum_e2"] += (error**2).sum(axis=0)
                st["sum_e4"] += (error**4).sum(axis=0)
                st["nonphysical"] += ((ratio < 0) | (ratio > 1)).sum(axis=0)
                for p in PROBABILITIES:
                    label = str(int(100*p)); zz = norm.ppf((1+p)/2)
                    flag_log = np.abs(t-logq_true[design.targets]) <= zz*sd
                    flag_delta = np.abs(error) <= zz*q*sd
                    for suffix, flag in (("log", flag_log), ("delta", flag_delta)):
                        key = "coverage"+label+"_"+suffix
                        batches[name][key] = flag
                        st[key] += flag.sum(axis=0)
            for j, (_, name, indices) in enumerate(entries):
                b = batches[name]
                for k, key in enumerate(("e", "e2", "coverage68_log", "coverage95_log", "coverage68_delta", "coverage95_delta")):
                    per_rep[first:last, j, k] = b[key][:, indices].mean(axis=1)
            if first == 0 or last == n_sim or last % max(config["batch_size"], n_sim//5) < config["batch_size"]:
                print(f"  {last}/{n_sim}", flush=True)
        for name, design in designs.items():
            st = state[name]; aa = st["analytic"]
            bias = st["sum_e"]/n_sim; mse = st["sum_e2"]/n_sim
            var_e = np.maximum((st["sum_e2"]-st["sum_e"]**2/n_sim)/(n_sim-1), 0)
            for i, target in enumerate(design.targets):
                row = {"truth_model": truth_id, "tolerance": config["tolerance"], "method": name,
                       "target_index": int(target), "zl": zl[target], "zs": zs[target],
                       "single_selected": bool(target in set(designs["single"].targets)),
                       "bias": bias[i], "bias_mcse": np.sqrt(var_e[i]/n_sim), "rmse": np.sqrt(mse[i]),
                       "nonphysical_fraction": st["nonphysical"][i]/n_sim,
                       **{key: float(value[i]) for key, value in aa.items()}}
                for key in ("coverage68_log", "coverage95_log", "coverage68_delta", "coverage95_delta"):
                    cv = st[key][i]/n_sim
                    row[key] = cv; row[key+"_mcse"] = np.sqrt(cv*(1-cv)/n_sim)
                target_rows.append(row)
        entry_index = {}
        for j, (cohort_name, name, indices) in enumerate(entries):
            entry_index[cohort_name, name] = j
            aa = state[name]["analytic"]
            vals = per_rep[:, j]
            bias, bias_se = summary_from_replicates(vals[:, 0])
            mse, mse_se = summary_from_replicates(vals[:, 1])
            rmse = np.sqrt(mse)
            row = {"truth_model": truth_id, "tolerance": config["tolerance"], "cohort": cohort_name,
                   "method": name, "n_targets": len(indices), "n_sim": n_sim,
                   "bias": bias, "bias_mcse": bias_se, "rmse": rmse,
                   "rmse_mcse_delta_method": mse_se/(2*rmse) if rmse else 0,
                   "mean_absolute_target_bias": float(np.abs(state[name]["sum_e"][indices]/n_sim).mean()),
                   "bias_analytic": float(aa["bias_analytic"][indices].mean()),
                   "rmse_analytic": float(np.sqrt(np.mean(aa["rmse_analytic"][indices]**2))),
                   "mean_absolute_transfer_bias": float(np.abs(aa["transfer_bias"][indices]).mean()),
                   "nonphysical_fraction": float(np.mean(state[name]["nonphysical"][indices]/n_sim))}
            for k, key in enumerate(("coverage68_log", "coverage95_log", "coverage68_delta", "coverage95_delta"), 2):
                row[key], row[key+"_mcse"] = summary_from_replicates(vals[:, k])
                row[key+"_analytic"] = float(aa[key+"_analytic"][indices].mean())
            summary_rows.append(row)
        for cohort_name, cohort in cohorts.items():
            j0 = entry_index[cohort_name, "single"]
            for name in cohort["methods"]:
                if name == "single":continue
                j1 = entry_index[cohort_name, name]
                difference = per_rep[:, j1]-per_rep[:, j0]
                row = {"truth_model": truth_id, "tolerance": config["tolerance"], "cohort": cohort_name,
                       "comparison": name+" minus single", "n_targets": len(cohort["targets"])}
                for k, label in enumerate(("bias_difference", "mse_difference", "coverage68_log_difference",
                                           "coverage95_log_difference", "coverage68_delta_difference", "coverage95_delta_difference")):
                    row[label], row[label+"_mcse"] = summary_from_replicates(difference[:, k])
                paired_rows.append(row)
        if config.get("save_replicates", False):
            np.savez_compressed(case_dir / f"replicates_{truth_number}.npz", statistics=per_rep,
                                entries=np.array([(a,b) for a,b,_ in entries]),
                                columns=np.array(["mean_error", "mean_squared_error", "coverage68_log", "coverage95_log", "coverage68_delta", "coverage95_delta"]))
    return pd.DataFrame(summary_rows), pd.DataFrame(target_rows), pd.DataFrame(paired_rows), diagnostics


def plot_results(summary, per_target, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output / "benchmark_figures.pdf") as pdf:
        for (truth, tol), case in summary.groupby(["truth_model", "tolerance"], sort=False):
            cohort = "common_all" if "common_all" in case.cohort.values else case.cohort.iloc[0]
            subset = case[case.cohort == cohort]
            names = subset.method.tolist(); positions = np.arange(len(names))
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
            axes[0].bar(positions, subset.bias, yerr=1.96*subset.bias_mcse, capsize=4)
            axes[0].axhline(0, color="black", lw=.7); axes[0].set_ylabel("Mean signed bias in R")
            axes[1].bar(positions, subset.rmse, yerr=1.96*subset.rmse_mcse_delta_method, capsize=4)
            axes[1].set_ylabel("Pooled RMSE in R")
            for shift, key, nominal in ((-.16, "coverage68_log", .68), (.16, "coverage95_log", .95)):
                axes[2].bar(positions+shift, subset[key], width=.3,
                            yerr=1.96*subset[key+"_mcse"], capsize=3, label=f"{int(nominal*100)}%")
                axes[2].axhline(nominal, color="gray", ls="--", lw=.7)
            axes[2].set_ylim(0, 1.08); axes[2].set_ylabel("Coverage (log-q intervals)"); axes[2].legend()
            for ax in axes:
                ax.set_xticks(positions, [LABELS[n] for n in names], rotation=15); ax.grid(axis="y", alpha=.2)
            fig.suptitle(f"{truth}; tolerance={tol:g}; {cohort}; N={int(subset.n_targets.iloc[0])}")
            pdf.savefig(fig); fig.savefig(output / f"summary_{truth}_{str(tol).replace('.', 'p')}.png", dpi=180)
            plt.close(fig)
            # Compare precisely the same lens targets, not each method's own sample.
            tt = per_target[(per_target.truth_model == truth) & (per_target.tolerance == tol)]
            common_ids = set.intersection(*(set(tt[tt.method == n].target_index) for n in names))
            fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
            for name in names:
                data = tt[(tt.method == name) & tt.target_index.isin(common_ids)].sort_values("zs")
                for ax, key in zip(axes.ravel(), ("bias", "rmse", "coverage68_log", "coverage95_log")):
                    ax.plot(data.zs, data[key], ".", ms=3, label=LABELS[name], alpha=.7)
            for ax, ylabel in zip(axes.ravel(), ("Bias in R", "RMSE in R", "68% coverage", "95% coverage")):
                ax.set_xlabel("Source redshift"); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
            axes[0,0].axhline(0, color="black", lw=.7)
            axes[1,0].axhline(.68, color="gray", ls="--"); axes[1,1].axhline(.95, color="gray", ls="--")
            axes[1,0].set_ylim(0,1.02); axes[1,1].set_ylim(0,1.02); axes[0,1].legend()
            fig.suptitle(f"{truth}; tolerance={tol:g}; common target diagnostics")
            pdf.savefig(fig); plt.close(fig)


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    defaults = {"n_sim": 10000, "batch_size": 512, "seed": 20261014,
                "matching_omega_m": .3, "tolerances": [.05], "h0": 70.0,
                "absolute_magnitude": -19.3, "methods": list(METHODS),
                "truth_models": [{"name": "flat_LCDM_0p30", "omega_m": .3, "w": -1}],
                "blas_threads": 1, "plots": True, "save_replicates": False,
                "selection_file": None, "output_dir": "results",
                "columns": {"sn_z": "zHD", "sn_id": "CID", "lens_z": "zl",
                            "source_z": "zs", "lens_name": "Lens name"}}
    unknown = set(config)-set(defaults)-{"sn_file", "lens_file", "covariance_file"}
    if unknown:raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if "columns" in config:
        if not isinstance(config["columns"], dict) or set(config["columns"])-set(defaults["columns"]):
            raise ValueError("Unknown or invalid column configuration.")
        config["columns"] = defaults["columns"] | config["columns"]
    for key, value in defaults.items():config.setdefault(key, value)
    for key in ("sn_file", "lens_file", "covariance_file", "selection_file", "output_dir"):
        if config.get(key):config[key] = str((path.parent / config[key]).resolve())
    for key in ("n_sim", "batch_size", "seed", "blas_threads"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):raise ValueError(f"{key} must be an integer.")
    if config["n_sim"] < 2 or config["batch_size"] < 1 or config["seed"] < 0 or config["blas_threads"] < 1:
        raise ValueError("Invalid simulation count, batch size, seed or thread count.")
    if "single" not in config["methods"] or len(config["methods"]) < 2 or len(set(config["methods"])) != len(config["methods"]) or set(config["methods"])-set(METHODS):
        raise ValueError("Choose single and at least one GLS method, with no duplicate methods.")
    if not config["truth_models"] or not np.isfinite(config["h0"]) or config["h0"] <= 0 or not np.isfinite(config["absolute_magnitude"]):
        raise ValueError("Invalid truth configuration.")
    names = [truth.get("name", f"truth_{i}") for i, truth in enumerate(config["truth_models"])]
    if len(set(names)) != len(names) or any(not n or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in n) for n in names):
        raise ValueError("Truth names must be unique, nonempty, and filename-safe.")
    if not config["tolerances"] or any(not np.isfinite(t) or not 0 < t < 1 for t in config["tolerances"]):
        raise ValueError("Tolerances must lie strictly between 0 and 1.")
    if config["selection_file"] and len(config["tolerances"]) != 1:
        raise ValueError("An external selection CSV can be used with only one tolerance.")
    return config


def run(config, overwrite=False):
    start = time.perf_counter(); output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("configuration") != config and not overwrite:
            raise ValueError("Output contains a different configuration. Choose another output_dir or add --overwrite.")
    print("Loading the original catalogue order and full STAT+SYS covariance ...", flush=True)
    inputs = load_inputs(config)
    summaries, targets, pairs, case_records = [], [], [], []
    for tolerance in config["tolerances"]:
        case = dict(config, tolerance=float(tolerance))
        directory = output / ("tol_"+format(tolerance, ".8g").replace(".", "p"))
        directory.mkdir(exist_ok=True)
        print(f"Preparing tolerance {tolerance:g}", flush=True)
        selection, provenance = get_selection(case, directory, inputs)
        designs, cohorts, availability = build_designs(case, inputs, selection, directory)
        summary, target, paired, diagnostics = simulate_case(case, inputs, designs, cohorts, directory)
        summaries.append(summary); targets.append(target); pairs.append(paired)
        case_records.append({"tolerance": tolerance, "available_systems": {name: len(d.targets) for name,d in designs.items()},
                             "cohorts": {name: {"target_indices": v["targets"].tolist(), "methods": v["methods"]} for name,v in cohorts.items()},
                             "selection_provenance": provenance, "sampler": diagnostics})
    summary = pd.concat(summaries, ignore_index=True)
    target = pd.concat(targets, ignore_index=True)
    paired = pd.concat(pairs, ignore_index=True)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    target.to_csv(output / "metrics_per_target.csv", index=False)
    paired.to_csv(output / "paired_comparison.csv", index=False)
    if config["plots"]:plot_results(summary, target, output)
    manifest = {"benchmark_version": VERSION, "configuration": config,
                "inputs": {key: {"path": config[key], "sha256": sha256(config[key])} for key in ("sn_file", "covariance_file", "lens_file")},
                "script_sha256": sha256(__file__), "n_sn_records": len(inputs[0]),
                "n_physical_CIDs": len(set(inputs[3])), "n_lenses": len(inputs[1]),
                "covariance_validation": inputs[7], "cases": case_records,
                "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
                "elapsed_seconds": time.perf_counter()-start,
                "truth_model": "flat constant-w background; common fixed absolute magnitude; exact catalogue redshifts",
                "magnitude_coordinate": config["columns"]["sn_z"],
                "noise_model": "joint Gaussian with the supplied STAT+SYS covariance; no extra diagonal scatter added",
                "selection": "fixed across noise realizations; determined using only redshifts, CID and catalogue order",
                "intervals": "equal-tailed Gaussian interval for log q, monotonically transformed to R; also symmetric delta-method intervals; no clipping",
                "scope": "conditional fixed-design simulation, not a cosmological parameter fit or a proof of universal estimator superiority",
                "mcse": "aggregate MCSE uses per-realization statistics, preserving correlations across targets",
                "coverage_nominal": {"68": .68, "95": .95}}
    dump_json(manifest_path, manifest)
    print(summary[["truth_model", "cohort", "method", "n_targets", "bias", "rmse", "coverage68_log", "coverage95_log"]].to_string(index=False))
    print(f"Results: {output}\nElapsed: {manifest['elapsed_seconds']:.1f} s", flush=True)
    return summary, target, paired


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("benchmark_config.json")))
    parser.add_argument("--n-sim", type=int, help="Override number of realizations")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument("--overwrite", action="store_true", help="Allow a different configuration in an existing output directory")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.n_sim is not None:
        if args.n_sim < 2:parser.error("--n-sim must be at least 2")
        config["n_sim"] = args.n_sim
    if args.output:config["output_dir"] = str(Path(args.output).resolve())
    try:
        from threadpoolctl import threadpool_limits
        limits = threadpool_limits(limits=config["blas_threads"])
    except ImportError:
        limits = contextlib.nullcontext()
    with limits:run(config, overwrite=args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
