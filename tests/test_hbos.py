import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hbos import _dynamic_bins, _dynamic_bins_per_bin, _static_bins, default_bins, hbos_scores  # noqa: E402
import data  # noqa: E402


class TestBins(unittest.TestCase):
    def test_default_bins(self):
        self.assertEqual([default_bins(n) for n in (367, 809, 6724)], [19, 28, 82])

    def test_static_equal_width(self):
        x = np.arange(0.0, 10.0)  # 0..9, k=3 -> width 3
        bins = _static_bins(x, 3)
        self.assertEqual([(b[0], b[1]) for b in bins], [(0, 3), (3, 6), (6, 9)])
        # Boundaries match scoring: [0,3), [3,6), [6,9].
        self.assertEqual([b[2] for b in bins], [3, 3, 4])

    def test_dynamic_equal_counts(self):
        x = np.arange(1.0, 9.0)  # 8 distinct values, k=4 -> 2 per bin
        bins = _dynamic_bins(x, 4)
        self.assertEqual([b[2] for b in bins], [2, 2, 2, 2])
        self.assertEqual(bins[0][:2], [1.0, 3.0])  # a bin ends at the first value of the next bin

    def test_dynamic_repeated_value_gets_own_bin(self):
        # Two trailing values avoid the documented zero-width final-bin merge.
        x = np.array([1, 2, 3, 5, 5, 5, 5, 5, 5, 9, 10], dtype=float)  # floor(11/5) = 2
        bins = _dynamic_bins(x, 5)
        self.assertIn(6, [b[2] for b in bins])  # the six 5s stay together

    def test_bins_cover_all_records(self):
        rng = np.random.default_rng(0)
        x = np.sort(rng.integers(0, 20, 200).astype(float))
        for f in (_dynamic_bins, _static_bins):
            self.assertEqual(sum(b[2] for b in f(x, 14)), 200)

    def test_dynamic_final_singleton_is_counted(self):
        # Java omitted the fifth value; it belongs in the final, merged bin.
        x = np.arange(5.0)
        self.assertEqual(_dynamic_bins(x, 2), [[0.0, 2.0, 2], [2.0, 4.0, 3]])
        expected = [math.log10(1.5), math.log10(1.5), 0, 0, 0]
        np.testing.assert_allclose(hbos_scores(x[:, None], bins=2), expected)

    def test_historical_helper_preserves_java_tail(self):
        self.assertEqual(_dynamic_bins_per_bin(np.arange(5.0), 2),
                         [[0.0, 2.0, 2], [2.0, 4.0, 2]])

    def test_constant_static_one_bin_has_positive_width(self):
        self.assertEqual(_static_bins(np.ones(3), 1), [[1.0, 2.0, 3]])


class TestScore(unittest.TestCase):
    def test_static_boundary_values_use_their_counted_bin(self):
        x = np.arange(10.0)
        score = hbos_scores(x[:, None], bins=3, mode="static")
        np.testing.assert_allclose(score, [math.log10(4 / 3)] * 6 + [0.0] * 4)

    def test_static_boundary_does_not_score_an_empty_bin(self):
        # Java counted the two 1s below their boundary, then looked them up in
        # an empty next bin, assigning infinity despite observing them twice.
        x = np.array([0.0, 1.0, 1.0, 3.0])
        score = hbos_scores(x[:, None], bins=3, mode="static")
        self.assertTrue(np.isfinite(score).all())
        np.testing.assert_allclose(score, [math.log10(2), 0, 0, math.log10(2)])

    def test_hand_computed_static_score(self):
        # values 0,0,0,0,10 with k=2: bins [0,5) holds 4 values, [5,10] holds 1, equal widths
        # relative heights 1.0 and 0.25, so scores are log10(1)=0 and log10(4)
        s = hbos_scores(np.array([[0], [0], [0], [0], [10.0]]), bins=2, mode="static")
        np.testing.assert_allclose(s, [0, 0, 0, 0, math.log10(4)])

    def test_features_add_up(self):
        x = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [10.0, 10.0]])
        one = hbos_scores(x[:, :1], bins=2, mode="static")
        both = hbos_scores(x, bins=2, mode="static")
        np.testing.assert_allclose(both, 2 * one)

    def test_isolated_point_ranks_first(self):
        rng = np.random.default_rng(1)
        X = np.vstack([rng.normal(0, 1, (300, 3)), [[8, 8, 8]]])
        for mode in ("dynamic", "static"):
            self.assertEqual(int(np.argmax(hbos_scores(X, mode=mode))), 300)

    def test_scale_invariance(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(100, 2))
        np.testing.assert_allclose(hbos_scores(X, bins=10), hbos_scores(X * 50 + 7, bins=10), atol=1e-9)

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            hbos_scores(np.zeros((3, 1)), mode="x")

    def test_constant_and_single_record_scores_are_zero(self):
        for mode in ("dynamic", "static"):
            for X in (np.ones((3, 2)), np.array([[2.0, -4.0]])):
                with self.subTest(mode=mode, shape=X.shape):
                    np.testing.assert_array_equal(hbos_scores(X, bins=1, mode=mode),
                                                  np.zeros(len(X)))

    def test_invalid_arrays_are_rejected(self):
        for X in ([], [1, 2], np.empty((0, 2)), np.empty((2, 0)),
                  [[[1.0]]], [[np.nan]], [[np.inf]], [[-np.inf]]):
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    hbos_scores(X)

    def test_invalid_bin_counts_are_rejected(self):
        for k in (0, -1, 1.5, 2.0, True, "2", np.nan, np.inf):
            for mode in ("dynamic", "static"):
                with self.subTest(k=k, mode=mode):
                    with self.assertRaises(ValueError):
                        hbos_scores([[0.0], [1.0]], bins=k, mode=mode)

    def test_numpy_integer_bin_count_is_supported(self):
        np.testing.assert_array_equal(hbos_scores([[0.0], [1.0]], bins=np.int64(1)),
                                      [0.0, 0.0])

    def test_unrepresentable_static_bins_are_rejected(self):
        for X in ([[1.0], [np.nextafter(1.0, np.inf)]], [[-1e308], [1e308]]):
            with self.subTest(X=X):
                with self.assertRaisesRegex(ValueError, "representable"):
                    hbos_scores(X, bins=2, mode="static")


class TestData(unittest.TestCase):
    def test_shapes_and_outlier_counts(self):
        expected = {"breast-cancer": (367, 30, 10), "pen-global": (809, 16, 90), "pen-local": (6724, 16, 10)}
        for name, (n, d, o) in expected.items():
            X, y = data.load(name)
            self.assertEqual((X.shape, int(y.sum())), ((n, d), o))


if __name__ == "__main__":
    unittest.main()
