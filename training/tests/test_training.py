from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from training.aggregate_plants import coverage_weights, distinct_top_mean
from training.make_splits import assign_balanced_folds, choose_holdout
from training.model import coral_targets, ordinal_probabilities
from training.train_plant_model import ConstantProbability


class OrdinalTests(unittest.TestCase):
    def test_coral_targets(self):
        result = coral_targets(torch.tensor([0, 1, 2]))
        self.assertTrue(torch.equal(result, torch.tensor([[0, 0], [1, 0], [1, 1]]).float()))

    def test_probabilities_sum_to_one_and_are_nonnegative(self):
        result = ordinal_probabilities(torch.tensor([[0.0, 2.0], [3.0, -1.0]]))
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.allclose(result.sum(dim=1), torch.ones(2)))


class AggregationTests(unittest.TestCase):
    def test_nonoverlap_weights_equal_tile_area(self):
        rows = pd.DataFrame({"x": [0, 512], "y": [0, 0]})
        result = coverage_weights(rows)
        self.assertTrue(np.allclose(result, 512 * 512))

    def test_overlap_does_not_double_count_total_area(self):
        rows = pd.DataFrame({"x": [0, 384], "y": [0, 0]})
        result = coverage_weights(rows)
        self.assertAlmostEqual(float(result.sum()), float((512 + 384) * 512), places=2)

    def test_distinct_top_regions(self):
        rows = pd.DataFrame({"x": [0, 10, 500], "y": [0, 10, 0]})
        value = distinct_top_mean(rows, np.array([1.0, 0.9, 0.8]), count=2)
        self.assertAlmostEqual(value, 0.9)


class SplitTests(unittest.TestCase):
    def test_holdout_keeps_two_per_score(self):
        rows = []
        for score in range(1, 6):
            for i in range(8):
                rows.append({"human_score": score, "collection": f"c{i % 2}"})
        plants = pd.DataFrame(rows)
        selected = choose_holdout(plants, 10, 7)
        remaining = plants.drop(index=selected).human_score.value_counts()
        self.assertTrue((remaining >= 2).all())

    def test_fold_assignment_is_complete_and_balanced(self):
        dev = pd.DataFrame(
            [{"human_score": i % 5 + 1, "collection": f"c{i % 3}"} for i in range(40)]
        )
        folds = assign_balanced_folds(dev, 5, 7)
        self.assertFalse((folds < 0).any())
        self.assertLessEqual(int(folds.value_counts().max() - folds.value_counts().min()), 1)

    def test_constant_probability_shape(self):
        result = ConstantProbability(0.25).predict_proba(pd.DataFrame({"x": [1, 2, 3]}))
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(np.allclose(result[:, 1], 0.25))


if __name__ == "__main__":
    unittest.main()
