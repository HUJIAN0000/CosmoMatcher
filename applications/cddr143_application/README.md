# CDDR application: adopted cddv301 protocol

This example revisits Zhou et al. (2021), ApJ 909, 118. The original 1048-SN Pantheon release is retained to address a specific historical omission: SN systematic covariance was not propagated in that analysis. The old greedy implementation used an unfiltered 205-lens arXiv-era catalogue and reported 120 pairs. Here the final Amante et al. (2020) 204-lens catalogue is represented by its independently specified 143-system SS3 subsample; MILP yields 99 systems with 198 distinct SNe. The changed sample and catalogue mean 120 versus 99 is not an isolated algorithm comparison.

The full case uses the original cddv301.py unchanged. The second case inserts only `_C_DELTA_MU = None` before `_DATA` is bundled. Both retain host mass error=99 without filtering or zeroing, the same priors, Nelder-Mead initialization, 140 walkers, 1000 steps, eight processes and the default emcee move. C11 systematic LL, SS, LS and transpose blocks are included or omitted together. The wrapper records deterministic seeds and actual plotting samples.

## Run

From anywhere, after installing this directory's requirements:

```bash
python reproduce.py
python reproduce_matching.py
```

Use the full script path when outside this directory. The first checks input hashes, SN identities and exact parent-matrix blocks. The second reads `matching_config.json`, reruns the 99-system assignment to `matching_rerun/`, and compares its table and covariance with the frozen reference.

`prepare.py` prepares the two inference working copies from the packaged `inputs/` only. It does not regenerate the 21-column input from an external local author folder. That table is frozen, with its values checked against the matched CSV by `verify_inputs.py`.

To regenerate inference and Figures 7–8, first copy this entire directory to a new working directory and run `python reproduce.py --sample` there. It replaces `runs/` and `output/` in that copy. The saved reference chains are already present; rerunning is unnecessary to inspect the adopted results. Multi-process creation must be permitted on the host.

## Limits of interpretation

Both cases omit within-SN SALT2 statistical cross-covariances. The original parameter-dependent chi-square weighting excludes a covariance determinant. The code convention is `DA*(1+z)^2/DL = 1+eta0*z`, reciprocal to the printed Zhou 2021 convention; finite-redshift linear forms cannot be converted by a sign change alone. Both saved chains fail the original 50-autocorrelation-time length check. Burn-in and thinning do not establish convergence. The tables illustrate this fixed workflow and are not a new convergence-validated precision CDDR measurement.

`output/cddr143_triangle.pdf` is the adopted Figure 7 overlay generated with `cosmo_tools.py`; Figure 8 and the parameter table use the same two samples. No cddv302, alternate normalization, host-error zeroing or alternative sampler analysis is included. Historical absolute paths in manifests identify source provenance; executable entry points use packaged relative paths.
