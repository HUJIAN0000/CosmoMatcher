"""Generate explicitly synthetic examples; these are not publication data."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import CosmoMatcher as cm


def main():
    folder = Path(__file__).resolve().parent
    z = np.array([.100, .105, .300, .315, .500, .800])
    mu = 5*np.log10(299792.458/70*(1+z)*cm.comoving_distance(z)) + 25
    x1 = np.array([.1, -.2, .3, .1, -.1, 0.])
    colour = np.array([.01, -.01, .02, .0, -.02, .01])
    mag = mu-19.3
    alpha, beta = .14, 3.1
    sn = pd.DataFrame({"CID": [f"SYNTHETIC_{i:03d}" for i in range(len(z))],
                       "z": z, "m_b_corr": mag, "mu": mu, "mb": mag-alpha*x1+beta*colour,
                       "x1": x1, "colour": colour, "dmb": .1, "dx1": .1, "dcolour": .02,
                       "cov_m_s": 0., "cov_m_c": 0., "cov_s_c": 0.})
    sn.to_csv(folder / "synthetic_tracers.csv", index=False)
    pd.DataFrame({"Name": ["SYNTHETIC_LENS_A", "SYNTHETIC_LENS_B"], "zl": [.101, .106],
                  "zs": [.300, .316]}).to_csv(folder / "synthetic_sgl.csv", index=False)
    pd.DataFrame({"Name": ["SYNTHETIC_CLUSTER_A", "SYNTHETIC_CLUSTER_B", "SYNTHETIC_CLUSTER_C"],
                  "z": [.101, .106, .315]}).to_csv(folder / "synthetic_clusters.csv", index=False)
    c = np.eye(len(z))*.01 + .001
    with (folder / "synthetic_covariance.cov").open("w") as handle:
        handle.write(f"{len(z)}\n")
        np.savetxt(handle, c.ravel(), fmt="%.16e")
    np.save(folder / "synthetic_target_covariance.npy", np.eye(2)*.0004)
    np.save(folder / "synthetic_target_sn_cross.npy", np.full((2, len(z)), .00001))
    coeffs = {"v0": .0005, "va": .0002, "vb": .0001, "v0a": .00001, "v0b": .00001, "vab": .000005}
    for key, coefficient in coeffs.items():
        np.save(folder / f"synthetic_jla_{key}.npy", np.eye(len(z))*coefficient)
    base = {"mode": "sgls", "sn_type": "pantheon", "algo": "milp", "tol_type": "dd",
            "file_paths": {"tracer": "synthetic_tracers.csv", "obj": "synthetic_sgl.csv", "covariance": "synthetic_covariance.cov"},
            "col_cfg": {"tracer_z_col": "z", "z1_col": "zl", "z2_col": "zs", "tracer_id_col": "CID",
                        "magnitude_col": "m_b_corr", "mu_col": "mu"},
            "params": {"wm": .3, "tol": .05, "unique_by": "object", "formulation": "compact",
                       "plots": True, "require_covariance": True, "output_dir": "../demo_results/sgl"}}
    (folder / "demo_sgl.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    cluster = json.loads(json.dumps(base)); cluster["mode"] = "cluster"
    cluster["file_paths"]["obj"] = "synthetic_clusters.csv"
    cluster["col_cfg"]["z1_col"] = "z"; cluster["col_cfg"].pop("z2_col")
    cluster["params"]["output_dir"] = "../demo_results/cluster"
    (folder / "demo_cluster.json").write_text(json.dumps(cluster, indent=2), encoding="utf-8")
    jla = json.loads(json.dumps(base)); jla["sn_type"] = "jla"
    jla["file_paths"].pop("covariance")
    for key in cm.JLA_KEYS:
        jla["file_paths"][f"cov_jla_{key}"] = f"synthetic_jla_{key}.npy"
    jla["col_cfg"]["magnitude_col"] = "mb"; jla["col_cfg"].pop("mu_col")
    jla["params"].update(alpha=alpha, beta=beta, jla_stat_mode="auto", absolute_magnitude=-19.3,
                         output_dir="../demo_results/jla")
    (folder / "demo_jla.json").write_text(json.dumps(jla, indent=2), encoding="utf-8")
    print("Synthetic catalogues, covariance matrices and three demo configurations created.")


if __name__ == "__main__":
    main()
