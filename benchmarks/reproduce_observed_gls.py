#!/usr/bin/env python3
"""Descriptive comparison using observed m_b_corr; no bias/coverage claims."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import cosmomatcher_benchmark as b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--selection', type=Path,
                        default=root/'sample_results/tol_0p05/selection.csv')
    parser.add_argument('--output', type=Path, default=root/'observed_results')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {'sn_file': str(root/'data/Pantheon+SH0ES.dat'),
              'covariance_file': str(root/'data/Pantheon+SH0ES_STAT+SYS.cov'),
              'lens_file': str(root/'data/161sgls.csv'), 'tolerance': .05,
              'matching_omega_m': .3, 'methods': list(b.METHODS),
              'columns': {'sn_z': 'zHD', 'sn_id': 'CID', 'lens_z': 'zl',
                          'source_z': 'zs', 'lens_name': 'Lens name'}}
    with threadpool_limits(limits=1):
        inputs = b.load_inputs(config)
        selection = b.read_csv_auto(args.selection, dtype={'sn_lens_CID': str,
                                                          'sn_source_CID': str})
        designs, cohorts, _ = b.build_designs(config, inputs, selection, output)
        sn, lenses, z, ids, zl, zs, covariance, diagnostics = inputs
        magnitudes = pd.to_numeric(sn['m_b_corr'], errors='raise').to_numpy(float)
        if not np.isfinite(magnitudes).all():
            raise ValueError('Nonfinite observed corrected magnitude.')
        ratios, jacobians, frames, summaries = {}, {}, [], []
        for method, design in designs.items():
            t = design.contrast @ magnitudes + np.log((1+design.z_s)/(1+design.z_l))
            q = np.exp(t)
            ratios[method] = -np.expm1(t)
            jacobians[method] = -q[:, None]*design.contrast
            cr = jacobians[method] @ covariance @ jacobians[method].T
            summaries.append({'method': method, 'n_available': len(design.targets),
                              'anchor_covariance_rank': int(np.linalg.matrix_rank(design.covariance_anchor)),
                              'ratio_covariance_rank': int(np.linalg.matrix_rank(cr))})
            frames.append(pd.DataFrame({'method': method, 'target_index': design.targets,
                                        'zl': zl[design.targets], 'zs': zs[design.targets],
                                        'R_observed': ratios[method],
                                        'sigma_R_delta': np.sqrt(np.maximum(np.diag(cr), 0))}))
        pd.concat(frames, ignore_index=True).to_csv(output/'observed_per_target.csv', index=False)
        comparison = []
        for cohort_name, cohort in cohorts.items():
            targets = cohort['targets']
            base_idx = np.searchsorted(designs['single'].targets, targets)
            rb = ratios['single'][base_idx]
            jb = jacobians['single'][base_idx]
            vb = np.diag(jb @ covariance @ jb.T)
            for method in cohort['methods']:
                if method == 'single':
                    continue
                idx = np.searchsorted(designs[method].targets, targets)
                r = ratios[method][idx]
                j = jacobians[method][idx]
                v = np.diag(j @ covariance @ j.T)
                difference_j = j-jb
                difference_cov = difference_j @ covariance @ difference_j.T
                difference = r-rb
                np.savez_compressed(output/f'paired_{cohort_name}_{method}.npz',
                                    target_index=targets, R_single=rb, R_gls=r,
                                    R_difference=difference,
                                    covariance_difference_delta=difference_cov,
                                    cross_method_covariance_delta=jb @ covariance @ j.T)
                comparison.append({'cohort': cohort_name, 'method': method,
                                   'n_targets': len(targets),
                                   'median_abs_R_difference': float(np.median(abs(difference))),
                                   'max_abs_R_difference': float(np.max(abs(difference))),
                                   'median_sigma_ratio_gls_to_single': float(np.median(np.sqrt(v/vb)))})
        pd.DataFrame(comparison).to_csv(output/'observed_comparison.csv', index=False)
        b.dump_json(output/'observed_manifest.json', {
            'inputs': {k: {'sha256': b.sha256(config[k])} for k in
                       ('sn_file', 'covariance_file', 'lens_file')},
            'selection_sha256': b.sha256(args.selection),
            'benchmark_sha256': b.sha256(root/'cosmomatcher_benchmark.py'),
            'script_sha256': b.sha256(Path(__file__)),
            'configuration': config, 'covariance_validation': diagnostics,
            'availability': summaries, 'cohort_comparisons': comparison,
            'scope': 'Observed magnitudes, exact supplied redshifts, fixed selection; descriptive estimator differences, not truth-based bias or interval coverage.'})
        print(pd.DataFrame(comparison).to_string(index=False))
        print(f'Observed-data outputs: {output}')


if __name__ == '__main__':
    main()
