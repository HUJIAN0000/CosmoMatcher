"""Run beside reproduce.json with the three original inputs in data/."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import CosmoMatcher as cm
root=Path(__file__).resolve().parent
config=cm.read_config(root/'reproduce.json')
reference=json.loads((root/'reference_results/run_summary.json').read_text())
for name,path in config['file_paths'].items():
 digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 if digest!=reference['inputs'][name]['sha256']:raise ValueError(f'{name}: input bytes do not match reference')
result=cm.run_config(config)
old=pd.read_csv(root/'reference_results/matched_sgls_milp_dd_pantheon.csv')
new=pd.read_csv(result.output_dir/'matched_sgls_milp_dd_pantheon.csv')
pd.testing.assert_frame_equal(old,new,check_exact=False,rtol=1e-12,atol=1e-12)
with np.load(root/'reference_results/cov_sgls_milp_dd_pantheon.npz') as cov:
 for key in cov.files:np.testing.assert_allclose(result.matrices[key],cov[key],rtol=1e-12,atol=1e-12)
print('Reference matching and covariance reproduced within stated tolerances.')
