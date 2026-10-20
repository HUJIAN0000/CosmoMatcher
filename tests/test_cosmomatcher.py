"""Scientific and end-to-end regression checks; all catalogues here are synthetic."""
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import CosmoMatcher as cm


def quiet(_):
    pass


def brute(z1, zt, z2, tol, criterion, groups):
    """Independent small-instance enumeration, not rounded MILP output."""
    tr = cm.comoving_distance(zt) if criterion == "dd" else zt
    a = cm.comoving_distance(z1) if criterion == "dd" else z1
    b = None if z2 is None else cm.comoving_distance(z2) if criterion == "dd" else z2
    choices = []
    for i in range(len(z1)):
        options = []
        for j in range(len(zt)):
            el = abs(tr[j]-a[i])/(a[i] if criterion == "dd" else 1)
            if el > tol:
                continue
            if z2 is None:
                options.append(((groups[j],), el**2))
            else:
                for k in range(len(zt)):
                    es = abs(tr[k]-b[i])/(b[i] if criterion == "dd" else 1)
                    if es <= tol and groups[j] != groups[k] and zt[j] < zt[k]:
                        options.append(((groups[j], groups[k]), el**2+es**2))
        choices.append(options)
    best = (0, 0.0)
    def visit(i, used, count, cost):
        nonlocal best
        if i == len(choices):
            if count > best[0] or (count == best[0] and cost < best[1]):
                best = count, cost
            return
        visit(i+1, used, count, cost)
        for resources, extra in choices[i]:
            if not (set(resources) & used):
                visit(i+1, used | set(resources), count+1, cost+extra)
    visit(0, set(), 0, 0.0)
    return best


class Algorithms(unittest.TestCase):
    def test_comoving_distance_analytic_matter_only(self):
        z = np.array([0., .01, .1, 1., 4.])
        np.testing.assert_allclose(cm.comoving_distance(z, 1), 2*(1-1/np.sqrt(1+z)), atol=1e-11)

    def test_cluster_counterexample(self):
        p = cm.build_problem([.10, .14], [.11, .08], tol=.031, tol_type="dz")
        selection, info = cm.solve_problem(p, log_func=quiet)
        self.assertEqual(info["maximum_cardinality"], 2)
        np.testing.assert_array_equal(selection[:, 1], [1, 0])
        greedy, _ = cm.greedy_match(np.array([.10, .14]), np.array([.11, .08]),
                                   groups=np.arange(2), tol=.031, tol_type="dz", omega_m=.3)
        self.assertEqual(len(greedy), 1)

    def test_secondary_objective_tiny_cost(self):
        for formulation in ("triplet", "compact"):
            p = cm.build_problem([.3], [.311, .3, .625, .6], [.6],
                                 tol=.1, groups=np.arange(4), formulation=formulation)
            chosen, report = cm.solve_problem(p, log_func=quiet)
            np.testing.assert_array_equal(chosen, [[0, 1, 3]])
            self.assertEqual(report["secondary_cost"], 0.)

    def test_one_candidate_and_zero_cost(self):
        p = cm.build_problem([.1], [.1], tol=0)
        selected, report = cm.solve_problem(p, log_func=quiet)
        self.assertEqual(len(selected), 1)
        self.assertEqual(report["secondary_cost"], 0.)

    def test_empty_candidate_set(self):
        for form in ("triplet", "compact"):
            p = cm.build_problem([.1], [.6, .7], [.3], tol=.001, formulation=form)
            selected, report = cm.solve_problem(p, log_func=quiet)
            self.assertEqual(len(selected), 0)
            self.assertEqual(report["maximum_cardinality"], 0)

    def test_same_physical_object_cannot_fill_two_roles(self):
        for form in ("triplet", "compact"):
            p = cm.build_problem([.1], [.1, .3], [.3], groups=["SN_A", "SN_A"], tol=0, formulation=form)
            selected, _ = cm.solve_problem(p, log_func=quiet)
            self.assertEqual(len(selected), 0)

    def test_shared_pool_across_lens_source_roles(self):
        for form in ("triplet", "compact"):
            p = cm.build_problem([.1, .2], [.1, .2, .3], [.2, .3], tol=0, formulation=form)
            selected, _ = cm.solve_problem(p, log_func=quiet)
            self.assertEqual(len(selected), 1)

    def test_compact_overlapping_windows_order_constraint(self):
        z1, z2, zt = np.array([.40, .41]), np.array([.45, .46]), np.array([.39, .43, .44, .47])
        expected = brute(z1, zt, z2, .1, "dz", np.arange(4))
        for form in ("triplet", "compact"):
            p = cm.build_problem(z1, zt, z2, tol=.1, tol_type="dz", formulation=form)
            selected, info = cm.solve_problem(p, log_func=quiet)
            self.assertEqual(len(selected), expected[0])
            self.assertAlmostEqual(info["secondary_cost"], expected[1], places=10)
            self.assertTrue(np.all(zt[selected[:, 1]] < zt[selected[:, 2]]))

    def test_random_instances_against_brute_force(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            zt = np.sort(rng.uniform(.08, .6, 6))
            z1 = rng.uniform(.1, .35, 3)
            z2 = z1+rng.uniform(.05, .25, 3)
            groups = np.array([0, 1, 2, 3, 4, 4]) if seed % 3 == 0 else np.arange(6)
            for criterion, tol in (("dz", .13), ("dd", .3)):
                for sources in (None, z2):
                    expected = brute(z1, zt, sources, tol, criterion, groups)
                    for form in ("triplet", "compact"):
                        with self.subTest(seed=seed, criterion=criterion, sgl=sources is not None, formulation=form):
                            p = cm.build_problem(z1, zt, sources, tol=tol, tol_type=criterion,
                                                 groups=groups, formulation=form)
                            chosen, report = cm.solve_problem(p, log_func=quiet)
                            self.assertEqual(len(chosen), expected[0])
                            self.assertAlmostEqual(report["secondary_cost"], expected[1], places=8)

    def test_invalid_geometry_and_tolerance(self):
        for kwargs in ({"tol": -1}, {"tol": float("nan")}, {"tol_type": "unknown"}):
            with self.assertRaises(cm.CosmoMatcherError):
                cm.build_problem([.1], [.1], **kwargs)
        with self.assertRaises(cm.CosmoMatcherError):
            cm.build_problem([.3], [.2], [.1])

    def test_uncertified_solver_output_is_rejected(self):
        p = cm.build_problem([.1], [.1], tol=0)
        failure = SimpleNamespace(status=1, message="Time limit reached", x=np.array([1.]), fun=-1.)
        with patch.object(cm, "milp", return_value=failure):
            with self.assertRaisesRegex(cm.CosmoMatcherError, "certified optimum"):
                cm.solve_problem(p, log_func=quiet)

    def test_extra_linear_constraints(self):
        p = cm.build_problem([.1], [.1, .11], tol=.02, tol_type="dz")
        constraint = cm.LinearConstraint(np.array([[1., 0.]]), 0., 0.)
        selection, info = cm.solve_problem(p, extra_constraints=constraint, log_func=quiet)
        self.assertEqual(info["maximum_cardinality"], 1)
        self.assertEqual(selection[0, 1], 1)

    def test_empty_model_with_infeasible_extra_constraint(self):
        p = cm.build_problem([.1], [.3], tol=0)
        constraint = cm.LinearConstraint(np.empty((1, 0)), 1., 1.)
        with self.assertRaises(cm.CosmoMatcherError):
            cm.solve_problem(p, extra_constraints=constraint, log_func=quiet)


class Covariances(unittest.TestCase):
    def test_jla_signs_and_statistical_variance(self):
        components = {key: np.eye(2)*(i+1) for i, key in enumerate(cm.JLA_KEYS)}
        a, b = .14, 3.1
        expected = 1+a*a*2+b*b*3+2*a*4-2*b*5-2*a*b*6+.04
        actual = cm.assemble_jla(components, a, b, np.full(2, .04))
        np.testing.assert_allclose(actual, np.eye(2)*expected)

    def test_jla_diagonal_no_double_counting(self):
        table = pd.DataFrame({"dmb": [.1], "dx1": [.2], "dcolour": [.03],
                              "cov_m_s": [.001], "cov_m_c": [.0002], "cov_s_c": [.0003]})
        a, b = .14, 3.1
        expected = .1**2+a*a*.2**2+b*b*.03**2+2*a*.001-2*b*.0002-2*a*b*.0003
        np.testing.assert_allclose(cm.jla_statistical_variance(table, a, b), [expected])

    def test_covariance_psd_validation(self):
        for c in (np.array([[1., 2.], [2., 1.]]), np.array([[1., .2], [.3, 1.]]),
                  np.array([[np.nan]])):
            with self.assertRaises(cm.CosmoMatcherError):
                cm.validate_covariance(c)
        cm.validate_covariance(np.zeros((2, 2)))
        cm.validate_covariance(np.ones((2, 2)))

    def test_text_roundoff_symmetrization_is_bounded_and_recorded(self):
        raw = np.array([[.04, .00300001], [.003, .04]])
        original = raw.copy()
        normalized, report = cm.normalize_covariance(raw)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(normalized, normalized.T)
        self.assertTrue(report['roundoff_correction_applied'])
        self.assertAlmostEqual(report['asymmetry_max_abs'], 1e-8)
        with self.assertRaisesRegex(cm.CosmoMatcherError, 'Max'):
            cm.normalize_covariance(np.array([[.04, .005], [.003, .04]]))

    def test_complete_role_blocks(self):
        c = np.arange(25).reshape(5, 5)
        indices = [4, 1, 3, 0]
        blocks = cm._matrix_blocks(c, indices, 2, True)
        np.testing.assert_array_equal(blocks["full"], c[np.ix_(indices, indices)])
        np.testing.assert_array_equal(blocks["cross"], c[np.ix_([4, 1], [3, 0])])

    def test_lens_ratio_jacobian_finite_difference(self):
        m = np.array([20., 21., 22., 23.])
        zl, zs = np.array([.1, .2]), np.array([.4, .6])
        _, _, j = cm.flat_lens_distance_ratio(m[:2], m[2:], zl, zs)
        numerical = np.empty((2, 4))
        for k in range(4):
            delta = np.zeros(4); delta[k] = 1e-5
            plus = cm.flat_lens_distance_ratio((m+delta)[:2], (m+delta)[2:], zl, zs)[0]
            minus = cm.flat_lens_distance_ratio((m-delta)[:2], (m-delta)[2:], zl, zs)[0]
            numerical[:, k] = (plus-minus)/2e-5
        np.testing.assert_allclose(j, numerical, rtol=1e-8, atol=1e-9)

    def test_cross_covariance_changes_ratio_variance(self):
        c = np.array([[.04, .03], [.03, .04]])
        _, propagated, j = cm.flat_lens_distance_ratio([20], [23], [.3], [.6], c)
        np.testing.assert_allclose(propagated, j @ c @ j.T)
        _, diagonal, _ = cm.flat_lens_distance_ratio([20], [23], [.3], [.6], np.diag(np.diag(c)))
        self.assertLess(propagated[0, 0], diagonal[0, 0])

    def test_luminosity_distance_known_value(self):
        d, c = cm.luminosity_distance([35], np.array([[.04]]))
        np.testing.assert_allclose(d, [100.])
        np.testing.assert_allclose(c, [[(np.log(10)/5*100)**2*.04]])

    def test_all_candidate_gls_preserves_overlap_and_local_trend(self):
        z = np.array([.09, .10, .11, .12])
        c = np.eye(4)*.01 + .001
        result = cm.window_gls(z, 20+10*z, c, [.1, .11], tolerance=.3)
        np.testing.assert_allclose(result["estimates"], [21., 21.1], atol=1e-9)
        np.testing.assert_allclose(result["covariance"], result["weights"] @ c @ result["weights"].T)
        self.assertNotEqual(result["covariance"][0, 1], 0)
        constant = cm.window_gls(z, 20+10*z, c, [.1, .11], tolerance=.3, linear=False)
        self.assertEqual(constant["covariance_rank"], 1)


class FilesAndWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        pd.DataFrame({"CID": ["001", "002", "003"], "z": [.1, 0., .3],
                      "m_b_corr": [20., 21., 23.]}).to_csv(self.path / "sn.csv", index=False)
        pd.DataFrame({"Name": ["lens"], "zl": [.1], "zs": [.3]}).to_csv(self.path / "sgl.csv", index=False)
        self.c = np.array([[.04, .001, .01], [.001, .04, .001], [.01, .001, .04]])
        np.save(self.path / "cov.npy", self.c)
        self.config = {"file_paths": {"tracer": str(self.path / "sn.csv"), "obj": str(self.path / "sgl.csv"), "covariance": str(self.path / "cov.npy")},
                       "col_cfg": {"tracer_z_col": "z", "z1_col": "zl", "z2_col": "zs", "magnitude_col": "m_b_corr"},
                       "params": {"tol": 0., "invalid_rows": "drop", "plots": False, "output_dir": str(self.path / "out")}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_filtering_preserves_parent_covariance_positions(self):
        result = cm.run_config(self.config, quiet)
        self.assertEqual(len(result.matched), 1)
        np.testing.assert_array_equal(result.index_map.covariance_row, [0, 2])
        np.testing.assert_allclose(result.matrices["full"], self.c[np.ix_([0, 2], [0, 2])])
        self.assertIn("ratio", result.matrices)
        self.assertTrue((result.output_dir / "run_summary.json").exists())
        self.assertEqual(result.summary["distinct_physical_ids"], 2)
        self.assertEqual(result.matched.sn_lens_CID.iloc[0], "001")

    def test_explicit_covariance_index_for_reordered_catalogue(self):
        df = pd.read_csv(self.path / "sn.csv").iloc[[2, 0, 1]].copy()
        df["cov_row"] = [2, 0, 1]
        df.to_csv(self.path / "sn.csv", index=False)
        self.config["col_cfg"]["cov_index_col"] = "cov_row"
        result = cm.run_config(self.config, quiet)
        np.testing.assert_allclose(result.matrices["full"], self.c[np.ix_([0, 2], [0, 2])])

    def test_strict_redshifts_by_default(self):
        del self.config["params"]["invalid_rows"]
        with self.assertRaisesRegex(cm.CosmoMatcherError, "Invalid redshifts"):
            cm.run_config(self.config, quiet)
        self.assertFalse((self.path / "out").exists())

    def test_wrong_covariance_dimension_is_rejected(self):
        np.save(self.path / "cov.npy", np.eye(2))
        with self.assertRaises(cm.CosmoMatcherError):
            cm.run_config(self.config, quiet)

    def test_no_matches_exports_empty_catalogue_and_report(self):
        df = pd.read_csv(self.path / "sgl.csv"); df.zl = .2; df.zs = .4
        df.to_csv(self.path / "sgl.csv", index=False)
        result = cm.run_config(self.config, quiet)
        self.assertEqual(result.summary["matched_targets"], 0)
        self.assertEqual(result.matrices["full"].shape, (0, 0))

    def test_real_cross_probe_block_requires_supplied_data(self):
        np.save(self.path / "target.npy", [[.04]])
        np.save(self.path / "cross.npy", [[.002, .001, .003]])
        self.config["file_paths"].update(target_covariance=str(self.path / "target.npy"), target_sn_cross=str(self.path / "cross.npy"))
        result = cm.run_config(self.config, quiet)
        np.testing.assert_allclose(result.matrices["joint"][0, 1:], [.002, .003])
        self.assertEqual(result.summary["cross_probe_assumption"], "supplied")

    def test_incomplete_jla_files_fail(self):
        self.config.update(sn_type="jla")
        self.config["file_paths"]["cov_jla_v0"] = str(self.path / "cov.npy")
        with self.assertRaisesRegex(cm.CosmoMatcherError, "six"):
            cm.run_config(self.config, quiet)

    def test_jla_end_to_end(self):
        sn = pd.read_csv(self.path / "sn.csv")
        for col, value in {"dmb": .1, "dx1": .1, "dcolour": .02, "cov_m_s": 0., "cov_m_c": 0., "cov_s_c": 0.}.items():
            sn[col] = value
        sn.to_csv(self.path / "sn.csv", index=False)
        self.config["sn_type"] = "jla"
        self.config["params"].update(alpha=.14, beta=3.1)
        for key in cm.JLA_KEYS:
            matrix = np.eye(3)*.001 if key == "v0" else np.zeros((3, 3))
            np.save(self.path / f"{key}.npy", matrix)
            self.config["file_paths"][f"cov_jla_{key}"] = str(self.path / f"{key}.npy")
        result = cm.run_config(self.config, quiet)
        expected = .001+.1**2+.14**2*.1**2+3.1**2*.02**2
        np.testing.assert_allclose(result.matrices["full"], np.eye(2)*expected)
        self.assertEqual(result.summary["covariance_kind"], "total")
        self.assertIn("jla_vab_full", result.matrices)

    def test_commented_header_and_malformed_rows(self):
        p = self.path / "jla.dat"
        p.write_text("# metadata\n#name zcmb mb\nSN_A .1 20\nSN_B .2 21\n")
        df = cm.smart_read_csv(p)
        self.assertEqual(list(df.columns), ["name", "zcmb", "mb"])
        p.write_text("CID,z\nA,.1\nB,.2,EXTRA\n")
        with self.assertRaisesRegex(cm.CosmoMatcherError, "No records were skipped"):
            cm.smart_read_csv(p)

    def test_custom_identifier_keeps_leading_zeros(self):
        p = self.path / "custom.csv"
        p.write_text("custom_id,z\n001,.1\n1,.2\n")
        self.assertEqual(cm.smart_read_csv(p).custom_id.tolist(), ["001", "1"])

    def test_id_whitespace_does_not_create_extra_objects(self):
        df = pd.read_csv(self.path / "sn.csv", dtype=str)
        df.loc[0, "CID"] = "SN_A"; df.loc[2, "CID"] = " SN_A "
        df.to_csv(self.path / "sn.csv", index=False)
        result = cm.run_config(self.config, quiet)
        self.assertEqual(result.summary["matched_targets"], 0)

    def test_matrix_formats_and_declared_dimension(self):
        p = self.path / "matrix.cov"
        for text in ("2\n1\n0\n0\n1\n", "4\n1\n0\n0\n1\n", "2\n1 0\n0 1\n", "1.0 0\n0 1\n", "1.0,0\n0,1\n"):
            p.write_text(text)
            np.testing.assert_array_equal(cm.load_matrix(p, 2), np.eye(2))
        p.write_text("3\n1 0\n0 1\n")
        with self.assertRaisesRegex(cm.CosmoMatcherError, "declared"):
            cm.load_matrix(p)
        p.write_text(".1 .2 .3\n")
        self.assertEqual(cm.load_matrix(p, square=False).shape, (1, 3))

    def test_large_single_column_both_header_conventions(self):
        n = 1701
        matrix = np.eye(n)*.04+.0001
        for header in (n, n*n):
            path = self.path/f'parent_{header}.cov'
            with path.open('w') as handle:
                handle.write(str(header)+'\n')
                np.savetxt(handle,matrix.ravel(),fmt='%.8e')
            actual = cm.load_matrix(path, expected_n=n)
            np.testing.assert_allclose(actual,matrix,rtol=1e-14,atol=1e-15)

    def test_all_four_ablation_combinations(self):
        table = cm.run_ablation(self.config, log_func=quiet)
        self.assertEqual(len(table), 4)
        self.assertEqual(set(zip(table.algorithm, table.criterion)), set(itertools.product(["greedy", "milp"], ["dd", "dz"])))
        self.assertTrue((self.path / "out" / "ablation_summary.csv").exists())

    def test_sensitivity_reports_membership(self):
        table = cm.run_sensitivity(self.config, [.01, .05], [.3], quiet)
        self.assertEqual(len(table), 3)
        self.assertEqual(table.target_jaccard.iloc[0], 1.)
        self.assertEqual(table.assignment_jaccard.iloc[0], 1.)

    def test_cli_success_and_failure_exit_codes(self):
        p = self.path / "config.json"; p.write_text(json.dumps(self.config))
        run = subprocess.run([sys.executable, str(ROOT / "CosmoMatcher.py"), "run", "--config", str(p)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        del self.config["file_paths"]["covariance"]
        self.config["params"]["require_covariance"] = True
        p.write_text(json.dumps(self.config))
        run = subprocess.run([sys.executable, str(ROOT / "CosmoMatcher.py"), "run", "--config", str(p)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("requires a covariance", run.stderr)

    def test_headless_import_does_not_need_tkinter(self):
        command = "import sys; sys.modules['tkinter']=None; import CosmoMatcher; print(CosmoMatcher.__version__)"
        run = subprocess.run([sys.executable, "-c", command], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


class CanonicalMatchingTests(unittest.TestCase):
    def test_equal_redshift_choices_have_one_catalogue_order_assignment(self):
        expected = np.array([[0, 0, 2], [1, 1, 3]])
        for form in ("compact", "triplet"):
            p = cm.build_problem([.1, .1], [.1, .1, .2, .2], [.2, .2], tol=0, formulation=form)
            for repeat in range(2):
                selected, info = cm.solve_problem(p, log_func=quiet)
                np.testing.assert_array_equal(selected, expected)
                self.assertEqual(info["secondary_cost"], 0)
                self.assertEqual(info["canonicalization"]["rule"], "catalogue_order")

    def test_canonical_order_cannot_trade_cardinality_for_early_target(self):
        p = cm.build_problem([.1, .2, .2], [.1, .2], [.2, .3, .3], tol=.11, tol_type="dz")
        selected, info = cm.solve_problem(p, log_func=quiet)
        self.assertEqual(len(selected), info["maximum_cardinality"])
        self.assertLessEqual(abs(info["canonicalization"]["cost_change"]), 1e-12)

    def test_single_tracer_equal_choices_and_invalid_rule(self):
        p = cm.build_problem([.1, .1], [.1, .1], tol=0)
        selected, _ = cm.solve_problem(p, log_func=quiet)
        np.testing.assert_array_equal(selected[:, :2], [[0, 0], [1, 1]])
        with self.assertRaises(cm.CosmoMatcherError):
            cm.solve_problem(p, tie_break="random", log_func=quiet)


class GUIProcessTests(unittest.TestCase):
    def test_process_matches_direct_backend_with_relative_input_paths(self):
        config = cm.read_config(ROOT / "examples/demo_sgl.json")
        with tempfile.TemporaryDirectory() as folder:
            config["params"]["output_dir"] = str(Path(folder) / "direct")
            direct = cm.run_config(config, quiet)
            config["params"]["output_dir"] = str(Path(folder) / "process")
            config["file_paths"] = {k: os.path.relpath(v) for k, v in config["file_paths"].items()}
            original = json.dumps(config)
            job = cm._start_gui_job(config)
            temp_path = Path(job.temporary.name)
            try:
                self.assertNotEqual(job.process.pid, os.getpid())
                self.assertEqual(job.process.wait(timeout=60), 0)
                lines = cm._read_gui_job(job)
                self.assertIn("Report:", "\n".join(lines))
                self.assertEqual(cm._read_gui_job(job), [])
                actual = pd.read_csv(job.output_dir / "matched_sgls_milp_dd_pantheon.csv")
                expected = pd.read_csv(direct.output_dir / "matched_sgls_milp_dd_pantheon.csv")
                pd.testing.assert_frame_equal(actual, expected)
                np.testing.assert_allclose(np.load(job.output_dir / "cov_sgls_milp_dd_pantheon.npz")["full"], direct.matrices["full"])
                self.assertEqual(json.dumps(config), original)
            finally:
                cm._cleanup_gui_job(job)
            self.assertFalse(temp_path.exists())

    def test_worker_error_is_logged_and_cleanup_succeeds(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"file_paths": {"tracer": str(Path(folder)/"missing.csv"),
                                     "obj": str(Path(folder)/"missing_target.csv")},
                      "params": {"output_dir": str(Path(folder)/"results")}}
            job = cm._start_gui_job(config)
            try:
                self.assertEqual(job.process.wait(timeout=30), 2)
                self.assertTrue(cm._read_gui_job(job))
            finally:
                cm._cleanup_gui_job(job)

    def test_close_reaps_running_child_before_removing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            job = cm._start_gui_job({"params": {"output_dir": folder}})
            temp_path = Path(job.temporary.name)
            cm._cleanup_gui_job(job)
            self.assertIsNotNone(job.process.poll())
            self.assertFalse(temp_path.exists())

    def test_nonplain_configuration_and_nonmain_gui_rejected(self):
        with self.assertRaises(TypeError):
            cm._start_gui_job({"widget": object()})
        with patch.object(cm.threading, "current_thread", return_value=object()):
            with self.assertRaisesRegex(cm.CosmoMatcherError, "main thread"):
                cm.launch_gui()


if __name__ == "__main__":
    unittest.main(verbosity=2)
