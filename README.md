# Nonlinear programming and applied optimization

The notebook covers production planning, long-only portfolio optimization,
best-subset regression, and NFL point-spread ratings. The repository includes
the stock returns, variable-selection observations, and 256 game records used
by the reproducible analysis; no Excel workbook is included.

## Run the analyses

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproducible_analysis.py
python -m unittest discover -s tests -v
```

The standalone script runs four checks without a Gurobi license: SLSQP
production optimization, a long-only minimum-variance portfolio with a 1%
monthly expected return target, exhaustive best-subset regression using at most
two of `x1`–`x3`, and equality-constrained least-squares team ratings with mean
rating 85. Results are printed as JSON. Alternate CSVs may be supplied using
`--stocks`, `--variables`, and `--nfl`.

The portfolio uses sample means and covariance from 77 historical monthly
returns. This is a retrospective illustration, not investment advice or a
forecast; estimated returns and covariance are uncertain. The NFL rating model
uses the game rows as observations; repeated and incomplete matchups can affect
interpretation. Optimizer success does not establish that the models or input
assumptions are suitable for a real decision. The original notebook's Gurobi
parts are not re-executed by this standalone workflow.
