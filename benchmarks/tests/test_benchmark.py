"""Numerical/statistical verification, with independent Gaussian references."""
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cosmomatcher_benchmark as b


class InputAndDesignTests(unittest.TestCase):
    def test_both_flat_headers_and_square(self):
        C = np.array([[.04, -.003], [-.003, .09]])
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"c.txt"
            for header in (2, 4):
                np.savetxt(p, np.r_[header, C.ravel()])
                loaded, info = b.read_covariance(p, 2)
                np.testing.assert_array_equal(loaded, C)
            np.savetxt(p, C)
            loaded, _ = b.read_covariance(p, 2)
            np.testing.assert_array_equal(loaded, C)
            np.savetxt(p, np.r_[3, C.ravel()])
            with self.assertRaises(ValueError): b.read_covariance(p, 2)

    def test_invalid_covariance_is_not_silently_repaired(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"c.txt"
            np.savetxt(p, [[1., 2.], [2., 1.]])
            with self.assertRaises(ValueError): b.read_covariance(p, 2)
            np.savetxt(p, [[1., .02], [.0, 1.]])
            with self.assertRaises(ValueError): b.read_covariance(p, 2)

    def test_gls_intercept_and_overlap_covariance(self):
        z = np.array([.1, .14, .17, .21, .24, .3])
        anchors = np.array([.16, .2])
        C = np.diag(np.linspace(.015, .04, 6)) + .005*np.ones((6,6))
        dc = b.comoving_distance(z); da = b.comoving_distance(anchors)
        W, counts, reasons = b.local_weights(z, C, anchors, dc, da, .5, True)
        self.assertTrue(np.all(reasons == "ok"))
        np.testing.assert_allclose(W @ (18.+3*z), 18.+3*anchors, atol=1e-12)
        np.testing.assert_allclose(W.sum(axis=1), 1, atol=1e-12)
        propagated = W @ C @ W.T
        self.assertGreater(abs(propagated[0,1]), 1e-5)
        # Compare the preserved data model against the existing backend API.
        from backend import CosmoMatcher
        result = CosmoMatcher.window_gls(z, 18.+3*z, C, anchors, tolerance=.5, linear=True)
        np.testing.assert_allclose(result["weights"], W, atol=1e-11)
        np.testing.assert_allclose(result["covariance"], propagated, atol=1e-12)

    def test_unidentified_slope_is_excluded(self):
        z = np.array([.15, .15])
        W, counts, reason = b.local_weights(z, np.eye(2)*.02, np.array([.15]),
                                           b.comoving_distance(z), b.comoving_distance([.15]), .05, True)
        self.assertEqual(reason[0], "unidentified_slope")
        self.assertEqual(W.sum(), 0)

    def test_physical_id_resource_is_not_catalogue_row_resource(self):
        z = np.array([.1, .3, .1, .3])
        ids = np.array(["same", "s1", "same", "s2"])
        selection = pd.DataFrame({"obj_index": [0,1], "sn_lens_original_index": [0,2],
                                  "sn_source_original_index": [1,3]})
        with self.assertRaises(ValueError):
            b.validate_selection(selection, z, ids, np.array([.1,.1]), np.array([.3,.3]), .3, .05)

    def test_parent_and_joint_noise_are_the_same_statistical_model(self):
        C = np.array([[.04, .006, .003], [.006, .05, -.004], [.003, -.004, .03]])
        D = np.array([[1.,-1.,0.], [0.,1.,-1.], [1.,0.,-1.]])
        joint = D @ C @ D.T
        factor, diag = b.gaussian_factor(joint)
        np.testing.assert_allclose(factor @ factor.T, joint, atol=1e-13)
        draws = np.random.default_rng(914).standard_normal((100000, factor.shape[1])) @ factor.T
        empirical = np.cov(draws, rowvar=False)
        np.testing.assert_allclose(empirical, joint, rtol=.025, atol=8e-4)
        # Square-rooting a positive roundoff eigenvalue can create ~1e-8
        # residuals in an exactly dependent direction; assess their variance.
        self.assertLess(np.var(draws[:,0]+draws[:,1]-draws[:,2]), 1e-12*np.trace(joint))


class StatisticalReferenceTests(unittest.TestCase):
    def test_moments_and_both_coverages_against_direct_simulation(self):
        t_true = np.array([-.7]); t0 = np.array([-.66]); variance = np.array([.12**2])
        analytic = b.analytic_statistics(t0, t_true, variance)
        rng = np.random.default_rng(410)
        t = t0[0] + np.sqrt(variance[0])*rng.standard_normal(250000)
        ratio = 1-np.exp(t); true_ratio = 1-np.exp(t_true[0])
        error = ratio-true_ratio
        mcse = error.std(ddof=1)/np.sqrt(len(t))
        self.assertLess(abs(error.mean()-analytic["bias_analytic"][0]), 5*mcse)
        self.assertLess(abs(np.sqrt(np.mean(error**2))-analytic["rmse_analytic"][0]), .001)
        for p in (.68,.95):
            zz = norm.ppf((1+p)/2); label = str(int(100*p))
            # Construct endpoints directly in R rather than using the benchmark's log test.
            lower = 1-np.exp(t+zz*np.sqrt(variance[0]))
            upper = 1-np.exp(t-zz*np.sqrt(variance[0]))
            log_coverage = np.mean((lower <= true_ratio) & (true_ratio <= upper))
            delta_sd = np.exp(t)*np.sqrt(variance[0])
            delta_coverage = np.mean((ratio-zz*delta_sd <= true_ratio) & (true_ratio <= ratio+zz*delta_sd))
            self.assertLess(abs(log_coverage-analytic["coverage"+label+"_log_analytic"][0]), .004)
            self.assertLess(abs(delta_coverage-analytic["coverage"+label+"_delta_analytic"][0]), .004)

    def test_nominal_exact_coverage_when_log_target_is_unbiased(self):
        t = np.array([-.2, -1.4])
        result = b.analytic_statistics(t, t, np.array([.01,.04]))
        np.testing.assert_allclose(result["coverage68_log_analytic"], .68, atol=1e-15)
        np.testing.assert_allclose(result["coverage95_log_analytic"], .95, atol=1e-15)
        self.assertTrue(np.all(result["nonlinear_noise_bias"] < 0))

    def test_zero_noise_reference(self):
        truth = np.array([-.4,-.4]); mean = np.array([-.4,-.39])
        result = b.analytic_statistics(mean, truth, np.zeros(2))
        np.testing.assert_array_equal(result["coverage95_log_analytic"], [1,0])
        np.testing.assert_allclose(result["rmse_analytic"], np.abs(result["transfer_bias"]))

    def test_aggregate_mcse_respects_perfect_target_correlation(self):
        rep = np.random.default_rng(941).binomial(1,.68,2000).astype(float)
        targets = np.repeat(rep[:,None],100,axis=1)
        _, mcse = b.summary_from_replicates(targets.mean(axis=1))
        self.assertAlmostEqual(mcse, rep.std(ddof=1)/np.sqrt(len(rep)), places=14)
        self.assertGreater(mcse, np.sqrt(.68*.32/(2000*100))*8)

    def test_ratio_geometry_and_common_absolute_magnitude(self):
        z = np.array([.2,.8])
        dc = b.comoving_distance(z)
        truth = 1-dc[0]/dc[1]
        for M in (-19.3,-18.1):
            m = b.true_magnitudes(z,.3,-1,70,M)
            logq = b.KAPPA*(m[0]-m[1])+np.log((1+z[1])/(1+z[0]))
            self.assertAlmostEqual(1-np.exp(logq), truth, places=13)


if __name__ == "__main__":unittest.main()
