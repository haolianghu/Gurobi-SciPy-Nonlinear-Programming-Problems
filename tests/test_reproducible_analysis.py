import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from reproducible_analysis import best_subset, nfl_ratings, portfolio_optimum, production_optimum, run


ROOT = Path(__file__).resolve().parents[1]


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stocks = pd.read_csv(ROOT / "homework4stocks.csv")
        cls.variables = pd.read_csv(ROOT / "variable_selection.csv")
        cls.nfl = pd.read_csv(ROOT / "nflratings.csv", header=None)

    def test_existing_datasets(self):
        result = run(self.stocks, self.variables, self.nfl)
        self.assertAlmostEqual(result["production"]["budget_used"], 100000, places=2)
        weights = list(result["portfolio"]["weights"].values())
        self.assertAlmostEqual(sum(weights), 1, places=6)
        self.assertGreaterEqual(result["portfolio"]["expected_monthly_return"], 0.01 - 1e-6)
        self.assertLessEqual(len(result["best_subset_at_most_two"]["features"]), 2)
        self.assertEqual(len(result["nfl"]["ratings"]), 32)
        self.assertEqual(result["nfl"]["games"], 256)

    def test_portfolio_infeasible_target(self):
        with self.assertRaisesRegex(ValueError, "infeasible"):
            portfolio_optimum(self.stocks, target_return=10)

    def test_invalid_nfl_team(self):
        nfl = self.nfl.copy()
        nfl.iloc[0, 1] = 33
        with self.assertRaisesRegex(ValueError, "Team identifiers"):
            nfl_ratings(nfl)

    def test_subset_rejects_nonfinite(self):
        variables = self.variables.copy()
        variables.loc[0, "x1"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            best_subset(variables)

    def test_production_validation(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            production_optimum(0)


if __name__ == "__main__":
    unittest.main()
