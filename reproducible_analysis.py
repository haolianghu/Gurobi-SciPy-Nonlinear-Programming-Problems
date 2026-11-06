"""Reproduce selected nonlinear, portfolio, subset, and NFL analyses."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def production_optimum(budget=100_000.0, labor_cost=12.0, capital_cost=15.0):
    if not all(np.isfinite(v) and v > 0 for v in (budget, labor_cost, capital_cost)):
        raise ValueError("Budget and input costs must be finite and positive.")
    result = minimize(
        lambda inputs: -0.05 * inputs[0] ** (2 / 3) * inputs[1] ** (1 / 3),
        x0=np.array([budget / (2 * labor_cost), budget / (2 * capital_cost)]),
        bounds=[(0, None), (0, None)],
        constraints=[{"type": "ineq", "fun": lambda x: budget - labor_cost * x[0] - capital_cost * x[1]}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise ValueError(f"Production optimizer failed: {result.message}")
    return {"labor": float(result.x[0]), "capital": float(result.x[1]), "machines": float(-result.fun),
            "budget_used": float(labor_cost * result.x[0] + capital_cost * result.x[1])}


def portfolio_optimum(frame, target_return=0.01):
    returns = frame.drop(columns=["Date"], errors="ignore").apply(pd.to_numeric, errors="raise")
    values = returns.to_numpy(float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("Stock returns must be finite with at least two observations.")
    means = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False)
    if not np.isfinite(target_return):
        raise ValueError("Target return must be finite.")
    if means.max() < target_return - 1e-10:
        raise ValueError("Target return is infeasible for long-only weights.")
    result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        np.full(values.shape[1], 1 / values.shape[1]),
        method="SLSQP",
        bounds=[(0, 1)] * values.shape[1],
        constraints=[
            {"type": "eq", "fun": lambda weights: weights.sum() - 1},
            {"type": "ineq", "fun": lambda weights: means @ weights - target_return},
        ],
        options={"ftol": 1e-12, "maxiter": 3000},
    )
    if not result.success:
        raise ValueError(f"Portfolio optimizer failed: {result.message}")
    return {"expected_monthly_return": float(means @ result.x),
            "target_monthly_return": float(target_return),
            "monthly_volatility": float(np.sqrt(max(0, result.fun))),
            "weights": {str(name): float(weight) for name, weight in zip(returns.columns, result.x)}}


def best_subset(frame):
    required = {"y", "x1", "x2", "x3"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Variable-selection data must include {', '.join(sorted(required))}.")
    data = frame[list(sorted(required, key=lambda x: ["y", "x1", "x2", "x3"].index(x)))].apply(pd.to_numeric, errors="raise")
    if data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Variable-selection data must be finite and complete.")
    y = data["y"].to_numpy()
    predictors = [name for name in ("x1", "x2", "x3")]
    results = []
    for size in (1, 2):
        for selected in itertools.combinations(predictors, size):
            design = np.column_stack([np.ones(len(y)), data[list(selected)].to_numpy()])
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coefficients
            results.append({"features": list(selected), "sse": float(residual @ residual),
                            "coefficients": coefficients.tolist()})
    return min(results, key=lambda result: (result["sse"], result["features"]))


def nfl_ratings(frame):
    if frame.shape[1] != 5:
        raise ValueError("NFL game data must have five columns: week, home, away, home score, away score.")
    games = frame.iloc[:, 1:5].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(games).all():
        raise ValueError("NFL game data must be finite.")
    if not np.equal(games[:, :2], np.floor(games[:, :2])).all() or not np.isin(games[:, :2], np.arange(1, 33)).all():
        raise ValueError("Team identifiers must be integers from 1 to 32.")
    home, away = games[:, 0].astype(int) - 1, games[:, 1].astype(int) - 1
    actual = games[:, 2] - games[:, 3]
    design = np.zeros((len(games), 33))
    design[np.arange(len(games)), home] += 1
    design[np.arange(len(games)), away] -= 1
    design[:, 32] = 1
    constraint = np.r_[np.ones(32), 0.0]
    kkt = np.block([[design.T @ design, constraint[:, None]],
                    [constraint[None, :], np.zeros((1, 1))]])
    solution = np.linalg.lstsq(kkt, np.r_[design.T @ actual, 32 * 85.0], rcond=None)[0][:33]
    predicted = design @ solution
    non_ties = actual != 0
    accuracy = float(np.mean(np.sign(predicted[non_ties]) == np.sign(actual[non_ties]))) if non_ties.any() else None
    return {"games": len(games), "home_advantage_points": float(solution[32]),
            "rmse_points": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "non_tie_winner_accuracy": accuracy, "ratings": solution[:32].tolist()}


def run(stocks, variables, nfl):
    return {
        "production": production_optimum(),
        "portfolio": portfolio_optimum(stocks),
        "best_subset_at_most_two": best_subset(variables),
        "nfl": nfl_ratings(nfl),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stocks", type=Path, default=Path("homework4stocks.csv"))
    parser.add_argument("--variables", type=Path, default=Path("variable_selection.csv"))
    parser.add_argument("--nfl", type=Path, default=Path("nflratings.csv"))
    args = parser.parse_args()
    try:
        result = run(pd.read_csv(args.stocks), pd.read_csv(args.variables),
                     pd.read_csv(args.nfl, header=None))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
