"""Rerun the 143-lens/1048-SN assignment, without replacing reference results."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
R = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('matching_backend', R/'inputs/CosmoMatcherv1.43.py')
cm = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = cm
spec.loader.exec_module(cm)

def main():
    cfg = cm.read_config(R/'matching_config.json')
    result = cm.run_config(cfg)
    assert result.summary['matched_targets'] == 99
    filename = 'matched_sgls_milp_dd_pantheon.csv'
    old = pd.read_csv(R/'reference_matching'/filename)
    new = pd.read_csv(result.output_dir/filename)
    pd.testing.assert_frame_equal(old, new, check_exact=False, rtol=1e-12, atol=1e-12)
    with np.load(R/'inputs/cov_sgls_milp_dd_pantheon.npz') as ref:
        for key in ref.files:
            np.testing.assert_allclose(result.matrices[key], ref[key], rtol=1e-12, atol=1e-12)
    print('PASS: 99 matched systems and all covariance blocks reproduce the frozen reference.')

if __name__ == '__main__':
    main()
