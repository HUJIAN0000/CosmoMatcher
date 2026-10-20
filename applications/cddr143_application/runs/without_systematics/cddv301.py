# -*- coding: utf-8 -*-
"""
Cosmic Distance Duality Relation (CDDR) Constraint - Version 301 (Covariance-Enabled)
=============================================================================
Pairs Type Ia Supernova (SNe Ia) data with Strong Gravitational Lensing (SGL)
data to constrain possible deviations from the standard CDDR.

Updated with the full Pantheon+ sub-covariance matrix extracted by CosmoMatcher:
    C_total = C_RA + C_pRL
where:
    C_RA  = diag(dra^2)
    C_pRL = W @ C_delta_mu @ W  + diag(pfac^2 * var_nuisance)
    with W = diag(p * R_L * ln(10)/5) and C_delta_mu from Pantheon+ sub-matrix.
Chi^2 evaluation:
    chi2 = residual.T @ C_total_inv @ residual  (via robust Cholesky solve)
"""

from __future__ import division

import time
from pathlib import Path
from multiprocessing import Pool, freeze_support

import numpy as np
from numpy import loadtxt
import scipy.linalg as la
from scipy.special import expit
import scipy.optimize as op
import emcee
from getdist import plots, MCSamples
import corner

# ── Physical constant ─────────────────────────────────────────────────────────
C_LIGHT = 299792.458          # Speed of light [km/s]

# ── Data and Covariance loading ───────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent
data_file   = _script_dir / "result_final_min_select_handle"
npz_file    = _script_dir / "cov_sgls_milp_dd_pantheon.npz"

(zl, zs, q, sg, dsg,
 mbl, dmbl, xl, dxl, cl, dcl,
 mbs, dmbs, xs, dxs, cs, dcs,
 M, dM, M_s, dM_s) = loadtxt(data_file, unpack=True)

# Load component covariance matrices:
#   1. C_lens:    99x99 covariance among lens SNe
#   2. C_source:  99x99 covariance among source SNe
#   3. C_cross:   99x99 cross-covariance between lens and source SNe
#   4. C_cross.T: 99x99 cross-covariance between source and lens SNe
if npz_file.exists():
    _npz = np.load(npz_file)
    _C_LENS   = _npz['lens']
    _C_SOURCE = _npz['source']
    _C_CROSS  = _npz['cross']
elif (_script_dir / "cov_sgls_milp_dd_pantheon_lens.txt").exists():
    _C_LENS   = np.loadtxt(_script_dir / "cov_sgls_milp_dd_pantheon_lens.txt")
    _C_SOURCE = np.loadtxt(_script_dir / "cov_sgls_milp_dd_pantheon_source.txt")
    _C_CROSS  = np.loadtxt(_script_dir / "cov_sgls_milp_dd_pantheon_cross.txt")
else:
    _C_LENS, _C_SOURCE, _C_CROSS = None, None, None

if _C_LENS is not None:
    # 完整的系统误差对比阵：显式包含两项交叉协方差 C_cross 与 C_cross.T
    _C_SYS_DELTA = _C_LENS + _C_SOURCE - _C_CROSS - _C_CROSS.T
    # 叠加超新星视星等统计方差，构建完整的 99x99 Delta_mu 协方差矩阵
    _C_DELTA_MU = np.diag(dmbl**2 + dmbs**2) + _C_SYS_DELTA
else:
    cov_npy = _script_dir / "cov_delta_mu_submatrix.npy"
    cov_txt = _script_dir / "cov_delta_mu_submatrix.txt"
    if cov_npy.exists():
        _C_DELTA_MU = np.load(cov_npy)
    elif cov_txt.exists():
        _C_DELTA_MU = np.loadtxt(cov_txt)
    else:
        _C_DELTA_MU = None

_C_DELTA_MU = None  # Only comparison switch: omit all C11 systematics.

# Pre-bundle data for clean argument passing
_DATA = (zl, zs, q, sg, dsg,
         mbl, dmbl, xl, dxl, cl, dcl,
         mbs, dmbs, xs, dxs, cs, dcs,
         M, dM, M_s, dM_s,
         _C_DELTA_MU)

# ── Sigmoid step function and uncertainty ─────────────────────────────────────

def sigmoid(M_arr, m, tau):
    """
    Smooth Heaviside step function: S(M) = expit((m - M) / tau).
    Vectorized and numerically stable via scipy.special.expit.
    """
    return expit((m - M_arr) / tau)


def dsigmoid(M_arr, m, tau, dM_arr):
    """Propagated uncertainty of sigmoid."""
    s = sigmoid(M_arr, m, tau)
    return s * (1.0 - s) * dM_arr / tau


# ── Model functions ───────────────────────────────────────────────────────────

def RA(q, fe, sg):
    """
    Angular-diameter distance ratio from SGL:
        R_A = c^2 * q / (4 * pi * f_e^2 * sigma^2)
    """
    return C_LIGHT**2 * q / (4.0 * np.pi * fe**2 * sg**2)


def dRA(q, fe, sg, dsg):
    """Propagated uncertainty of R_A."""
    return RA(q, fe, sg) * np.sqrt(4.0 * (dsg / sg)**2 + 0.05**2)


def p_factor(h0, zl, zs):
    """Redshift-dependent CDDR correction factor."""
    return (1.0 + zs) * (1.0 + h0 * zl) / ((1.0 + zl) * (1.0 + h0 * zs))


def RL(mbl, mbs, xl, xs, cl, cs, M, M_s, a, b, gama, m, tau):
    """Luminosity-distance ratio from standardized SNe Ia distance moduli."""
    exponent = (mbl - mbs
                + a * (xl - xs)
                - b * (cl - cs)
                + gama * sigmoid(M,   m, tau)
                - gama * sigmoid(M_s, m, tau)) / 5.0
    return 10.0**exponent


def dRL(mbl, mbs, xl, xs, cl, cs,
        dmbl, dxl, dcl, dmbs, dxs, dcs,
        M, dM, M_s, dM_s,
        a, b, gama, m, tau):
    """
    Propagated uncertainty of R_L.
    Explicit parameter passing avoids global state dependencies.
    """
    rl = RL(mbl, mbs, xl, xs, cl, cs, M, M_s, a, b, gama, m, tau)
    variance = (dmbl**2
                + a**2 * dxl**2
                + b**2 * dcl**2
                + dmbs**2
                + a**2 * dxs**2
                + b**2 * dcs**2
                + gama**2 * dsigmoid(M,   m, tau, dM)**2
                + gama**2 * dsigmoid(M_s, m, tau, dM_s)**2)
    return (np.log(10.0) / 5.0) * rl * np.sqrt(variance)


# ── Log-likelihood ────────────────────────────────────────────────────────────

def lnlike(theta,
           zl, zs, q, sg, dsg,
           mbl, dmbl, xl, dxl, cl, dcl,
           mbs, dmbs, xs, dxs, cs, dcs,
           M, dM, M_s, dM_s,
           c_delta_mu=None):
    """
    Gaussian log-likelihood for CDDR residuals:
        residual = R_A - (1 - p * R_L)

    Full covariance propagation with the Pantheon+ sub-covariance matrix:
        C_total = diag(dra^2 + pfac^2 * var_nuisance) + W @ C_delta_mu @ W
        chi2 = residual.T @ C_total_inv @ residual
    Evaluated using Cholesky factorisation for numerical stability and float64 precision.
    """
    a, b, fe, h0, gama, m, tau = theta

    ra   = RA(q, fe, sg)
    dra  = dRA(q, fe, sg, dsg)
    pfac = p_factor(h0, zl, zs)
    rl   = RL(mbl, mbs, xl, xs, cl, cs, M, M_s, a, b, gama, m, tau)

    residual = ra - (1.0 - pfac * rl)

    if c_delta_mu is not None:
        # First-order Taylor propagation weight: w_i = p_i * R_{L,i} * ln(10)/5
        w = pfac * rl * (np.log(10.0) / 5.0)

        # Intra-lightcurve parameter variances (x1, c, M)
        var_nuisance = (np.log(10.0) / 5.0)**2 * rl**2 * (
            a**2 * dxl**2 + b**2 * dcl**2
            + a**2 * dxs**2 + b**2 * dcs**2
            + gama**2 * dsigmoid(M,   m, tau, dM)**2
            + gama**2 * dsigmoid(M_s, m, tau, dM_s)**2
        )
        cov_total = np.diag(dra**2 + pfac**2 * var_nuisance) + (w[:, None] * c_delta_mu * w[None, :])

        try:
            cho = la.cho_factor(cov_total, lower=True)
            chi2 = np.dot(residual, la.cho_solve(cho, residual))
        except (np.linalg.LinAlgError, la.LinAlgError):
            return -np.inf
    else:
        drl  = dRL(mbl, mbs, xl, xs, cl, cs,
                   dmbl, dxl, dcl, dmbs, dxs, dcs,
                   M, dM, M_s, dM_s,
                   a, b, gama, m, tau)
        sigma2 = dra**2 + pfac**2 * drl**2
        inv_cov = np.diag(1.0 / sigma2)
        chi2 = np.dot(residual, np.dot(inv_cov, residual))

    return -0.5 * chi2


# ── Prior ─────────────────────────────────────────────────────────────────────
# Prior bounds inherited from cddv205:
#   -0.2 < a < 0.2
#   2.0 < b < 6.0
#   0.5 < fe < 1.5
#   -0.5 < h0 < 0.5
#   0 < gama < 0.3
#   5 < m < 15
#   0.05 < tau < 0.20
PARAM_BOUNDS = {
    'alpha':  (-0.2,   0.7),
    'beta':   ( 2.0,   6.0),
    'f_e':    ( 0.5,   1.5),
    'eta0':   (-0.5,   0.5),
    'gamma':  ( 0.0,   0.3),
    'm_step': ( 5.0,  15.0),
    'tau':    ( 0.05,  0.20),
}

_bounds_lo = np.array([v[0] for v in PARAM_BOUNDS.values()])
_bounds_hi = np.array([v[1] for v in PARAM_BOUNDS.values()])

def lnprior(theta):
    """Uniform flat prior within bounds."""
    theta = np.asarray(theta)
    if np.all(theta > _bounds_lo) and np.all(theta < _bounds_hi):
        return 0.0
    return -np.inf


# ── Log-posterior ─────────────────────────────────────────────────────────────

def lnprob(theta, *data):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnlike(theta, *data)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# Initial guess and labels
THETA0 = np.array([0.00, 5.00, 1.06, 0.00, 0.16, 10.0, 0.10])
PARAM_NAMES_PLAIN = ['alpha', 'beta', 'f_e', 'eta0', 'gamma', 'm_step', 'tau']
PARAM_NAMES_LATEX = [r'\alpha', r'\beta', r'f_e',
                     r'\eta_0', r'\gamma', r'm_{\rm step}', r'\tau']


# ══════════════════════════════════════════════════════════════════════════════
# Main execution block
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    freeze_support()

    print(f"Loaded {len(zl)} matched SGL-SN pairs from: {data_file}")
    if _C_DELTA_MU is not None:
        if _C_LENS is not None:
            print(f"Loaded component covariance blocks from: {npz_file.name}")
            print(f"  -> C_lens   (透镜端系统协方差)    : {_C_LENS.shape}")
            print(f"  -> C_source (源端系统协方差)      : {_C_SOURCE.shape}")
            print(f"  -> C_cross  (透镜-源交叉协方差)   : {_C_CROSS.shape}")
            print(f"  -> C_cross.T(源-透镜交叉协方差)   : {_C_CROSS.T.shape}")
            print(f"  -> C_delta_mu (合并总协方差阵)    : {_C_DELTA_MU.shape}")
        else:
            print(f"Loaded sub-covariance matrix: {_C_DELTA_MU.shape}")
    else:
        print("Note: Running without sub-covariance matrix (diagonal mode).")

    # ── Step 1: Maximum Likelihood ───────────────────────────────────────────
    print("=" * 60)
    print("Step 1 — Maximum-likelihood optimisation (Nelder-Mead with bounds) …")
    neg_lnlike = lambda theta, *args: -lnlike(theta, *args)
    bounds = list(PARAM_BOUNDS.values())
    ml_result = op.minimize(
        neg_lnlike, THETA0, method='Nelder-Mead',
        bounds=bounds,
        args=_DATA,
        options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    theta_ml = ml_result.x
    print(f"  Converged : {ml_result.success}  |  nfev = {ml_result.nfev}")
    for name, val in zip(PARAM_NAMES_PLAIN, theta_ml):
        print(f"    {name:>8s} = {val:+.5f}")
    print(f"  Min chi2  = {2.0 * ml_result.fun:.3f} (for {len(zl)} pairs)")

    # ── Step 2: MCMC Configuration ───────────────────────────────────────────
    NDIM     = 7
    NWALKERS = 140     # From cddv205 configuration
    N_STEPS  = 1000    # Production steps
    NTHREADS = 8       # Multi-process workers

    # Initialise walkers strictly within the prior bounds
    rng = np.random.default_rng(seed=42)
    lo = _bounds_lo + 1e-3
    hi = _bounds_hi - 1e-3
    center = np.clip(theta_ml, lo, hi)
    pos = center + 1e-4 * rng.standard_normal((NWALKERS, NDIM))
    pos = np.clip(pos, _bounds_lo + 1e-5, _bounds_hi - 1e-5)

    # ── Step 3: Run emcee ────────────────────────────────────────────────────
    print(f"\nStep 2 — emcee sampling "
          f"({NWALKERS} walkers × {N_STEPS} steps, {NTHREADS} workers) …")

    t0 = time.perf_counter()
    with Pool(processes=NTHREADS) as pool:
        sampler = emcee.EnsembleSampler(
            NWALKERS, NDIM, lnprob,
            args=_DATA,
            pool=pool
        )
        sampler.run_mcmc(pos, N_STEPS, progress=True)
    elapsed = time.perf_counter() - t0
    print(f"  Sampling finished in {elapsed:.1f} s")

    # ── Step 4: Convergence & Acceptance ─────────────────────────────────────
    acc = np.mean(sampler.acceptance_fraction)
    print(f"  Mean acceptance fraction: {acc:.3f} "
          f"({'OK' if 0.2 <= acc <= 0.5 else 'WARNING: outside 0.2–0.5'})")

    print("\nStep 3 — Convergence diagnostics …")
    try:
        tau_int = sampler.get_autocorr_time(quiet=True)
        if np.any(np.isnan(tau_int)) or not np.all(np.isfinite(tau_int)):
            raise emcee.autocorr.AutocorrError("Autocorrelation time returned non-finite values.")
        burn_est = int(2 * np.max(tau_int))
        thin_est = max(1, int(0.5 * np.min(tau_int)))
        print(f"  Integrated autocorrelation times:")
        for name, t in zip(PARAM_NAMES_PLAIN, tau_int):
            print(f"    {name:>8s}: tau = {t:.1f}")
        print(f"  -> Suggested burn-in  >= {burn_est} steps")
        print(f"  -> Suggested thinning >= {thin_est}")
        if N_STEPS < 50 * np.max(tau_int):
            print(f"  ! WARNING: N_STEPS={N_STEPS} may be short "
                  f"(recommend >= {int(50*np.max(tau_int))})")
    except (emcee.autocorr.AutocorrError, ValueError) as exc:
        print(f"  ! Autocorrelation estimate unreliable: {exc}")
        burn_est = 150
        thin_est = 1

    # ── Step 5: Flatten chain and summaries ──────────────────────────────────
    flat_samples = sampler.get_chain(discard=burn_est, thin=thin_est, flat=True)
    print(f"\n  Flat sample shape: {flat_samples.shape}")

    print("\nStep 4 — Posterior summary (median +/- 1 sigma):")
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    for i, name in enumerate(PARAM_NAMES_PLAIN):
        lo, mid, hi = percentiles[:, i]
        print(f"  {name:>8s} = {mid:+.4f}  +{hi - mid:.4f}  -{mid - lo:.4f}")

    # ── Step 6: Generate plots ───────────────────────────────────────────────
    print("\nStep 5 — Generating plots …")
    out_dir = _script_dir

    # GetDist triangle plot
    mc_samples = MCSamples(
        samples=flat_samples,
        names=PARAM_NAMES_LATEX,
        labels=PARAM_NAMES_LATEX
    )
    g = plots.getSubplotPlotter()
    g.triangle_plot(mc_samples, filled=True)
    getdist_out = str(out_dir / "cddv301_getdist.pdf")
    g.export(getdist_out)
    print(f"  GetDist plot -> {getdist_out}")

    # Corner plot
    fig = corner.corner(
        flat_samples,
        bins=30,
        color="blue",
        levels=[0.6526, 0.9544],
        quantiles=[0.160, 0.500, 0.840],
        smooth=True,
        smooth1d=True,
        labels=PARAM_NAMES_LATEX,
        label_kwargs={"fontsize": 20},
        plot_contours=True,
        fill_contours=True,
        no_fill_contours=False,
    )
    corner_out = str(out_dir / "cddv301_corner.pdf")
    fig.savefig(corner_out)
    print(f"  Corner plot  -> {corner_out}")

    print("\nDone.")
