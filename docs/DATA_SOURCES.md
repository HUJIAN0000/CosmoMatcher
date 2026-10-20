# Data provenance and covariance conventions

- `data/161sgls.csv`: author-supplied 161-lens software benchmark catalogue associated with Chen et al. (2019). It is not the SS3 sample and is not claimed to be a new lens compilation.
- `data/Pantheon+SH0ES.dat` and `Pantheon+SH0ES_STAT+SYS.cov`: author-supplied Pantheon+ release snapshots, 1701 light-curve records. Upstream: https://github.com/PantheonPlusSH0ES/DataRelease . The matrix is STAT+SYS, not systematic-only.
- `applications/cddr143_application/inputs/Ancillary_C11.FITRES`, converted `Ancillary_C11_jla.csv`, and `sys_full_long_C11.txt`: original 1048-SN Pantheon C11 inputs associated with Scolnic et al. (2018), ApJ 859, 101. Upstream: https://github.com/dscolnic/Pantheon . The C11 matrix is systematic-only. The converted CSV includes mB-unit covariance fields; do not apply the x0-to-mB conversion twice.
- `sgls_143_fiduciary_ss3.csv`: author-prepared table checked against Amante et al. (2020), MNRAS 498, 6013, using their SS3 criterion 0.5 <= Dobs <= 1. The final parent has 204 systems, comprising 143 SS3 systems, 32 with Dobs>1 and 29 with Dobs<0.5. Only the actually used 143-system input is needed by the current matching run.
- Zhou et al. (2021), ApJ 909, 118 supplies the historical motivation (205-lens input, 120 pairs, statistical-only SN treatment), not the adopted matching reference or an identical modern inference protocol.

The journal PDFs themselves and unrelated unpublished manuscripts are not redistributed in this repository. The data snapshots and derived data retain their original attribution and any upstream terms. The authors should confirm redistribution terms; this document does not relicense third-party content.

`PACKAGE_SHA256.json` records package checksums; historical manifests preserve original machine paths as provenance only. Portable entry points use paths within this repository. Matching backends frozen in the benchmark and CDDR application are byte-identical to root `CosmoMatcher.py` at packaging.
