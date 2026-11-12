# Walk-forward results

Generated 2026-11-12 14:12 EST from `default.yaml`.

These numbers are the expanding-window test blocks only. Hyperparameters were
selected on a time-series split inside each training window. Nothing here was
chosen by looking at the test period.

## Sample

- Signal months: 2012-01-31 to 2026-09-30 (177 months).
- Median names per month: 474.
- Costs: 10.0 bps per side on one-way turnover `0.5 * sum(|w_t - w_t-1|)`, charged from a flat book in the first month.
- Long/short books are +1 in the top decile and -1 in the bottom decile, equal weight.
- The one-month embargo drops the feature month immediately before each test block,
  so a training label cannot end on the first test date.
- Universe: current S&P 500 constituents unless a point-in-time file was supplied.
  That default embeds survivorship bias.

## Long/short, net of 10 bps

| Model | Ann. return | Ann. vol | Sharpe | Sortino | Max DD | Calmar | Turnover | Hit rate | Rank IC | ICIR | IC t | Alpha (ann.) | Alpha t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Momentum + reversal | -1.55% | 13.71% | -0.05 | -0.06 | -51.23% | -0.03 | 1.75 | 46.33% | 0.006 | 0.04 | 0.53 | -3.91% | -1.11 |
| Ridge | 10.47% | 14.73% | 0.75 | 1.41 | -20.71% | 0.51 | 0.96 | 54.24% | 0.024 | 0.18 | 2.34 | 9.70% | 3.35 |
| LightGBM | 10.46% | 11.43% | 0.93 | 1.91 | -12.32% | 0.85 | 1.27 | 58.76% | 0.025 | 0.25 | 3.27 | 8.46% | 3.60 |
| MLP ensemble | 9.82% | 12.50% | 0.81 | 1.50 | -19.02% | 0.52 | 1.05 | 58.19% | 0.015 | 0.12 | 1.66 | 8.88% | 2.96 |

## Long/short, gross of costs

| Model | Ann. return | Ann. vol | Sharpe | Sortino | Max DD | Calmar | Turnover | Hit rate | Rank IC | ICIR | IC t | Alpha (ann.) | Alpha t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Momentum + reversal | 0.54% | 13.71% | 0.11 | 0.16 | -46.59% | 0.01 | 1.75 | 50.85% | 0.006 | 0.04 | 0.53 | — | — |
| Ridge | 11.73% | 14.73% | 0.83 | 1.59 | -19.71% | 0.60 | 0.96 | 58.19% | 0.024 | 0.18 | 2.34 | — | — |
| LightGBM | 12.14% | 11.43% | 1.06 | 2.27 | -11.30% | 1.07 | 1.27 | 61.02% | 0.025 | 0.25 | 3.27 | — | — |
| MLP ensemble | 11.20% | 12.50% | 0.91 | 1.73 | -18.71% | 0.60 | 1.05 | 58.76% | 0.015 | 0.12 | 1.66 | — | — |

## Long only, net of 10 bps

| Model | Ann. return | Ann. vol | Sharpe | Sortino | Max DD | Calmar | Turnover | Hit rate | Rank IC | ICIR | IC t | Alpha (ann.) | Alpha t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Momentum + reversal | 18.31% | 17.67% | 1.05 | 1.83 | -24.18% | 0.76 | 0.88 | 63.84% | 0.006 | 0.04 | 0.53 | 2.64% | 1.28 |
| Ridge | 26.52% | 21.29% | 1.22 | 2.24 | -30.76% | 0.86 | 0.48 | 69.49% | 0.024 | 0.18 | 2.34 | 11.37% | 6.09 |
| LightGBM | 28.30% | 21.18% | 1.29 | 2.37 | -29.17% | 0.97 | 0.59 | 67.80% | 0.025 | 0.25 | 3.27 | 11.67% | 7.10 |
| MLP ensemble | 25.26% | 20.00% | 1.23 | 2.32 | -27.35% | 0.92 | 0.49 | 70.06% | 0.015 | 0.12 | 1.66 | 10.57% | 5.35 |

## Information coefficients

IC is the monthly Pearson correlation of the score with the raw next-month return.
Rank IC is Spearman. ICIR is mean / sample standard deviation. The t-statistic is
`mean / (std / sqrt(n))`. Decile Spearman correlates decile number with mean return.

| Model | IC | IC std | ICIR | IC t | Rank IC | Rank IC std | Rank ICIR | Rank IC t | Decile Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Momentum + reversal | 0.007 | 0.138 | 0.05 | 0.63 | 0.006 | 0.152 | 0.04 | 0.53 | 0.39 |
| Ridge | 0.030 | 0.135 | 0.22 | 2.98 | 0.024 | 0.136 | 0.18 | 2.34 | 0.87 |
| LightGBM | 0.032 | 0.115 | 0.28 | 3.75 | 0.025 | 0.103 | 0.25 | 3.27 | 0.84 |
| MLP ensemble | 0.025 | 0.121 | 0.20 | 2.70 | 0.015 | 0.123 | 0.12 | 1.66 | 0.96 |

## Factor attribution

Net monthly returns are regressed on Ken French FF5 + momentum with Newey-West
standard errors (lags = 3). Long/short alpha
is the intercept of the spread. Long-only alpha uses return minus RF. Annual alpha
compounds the monthly intercept. t-statistics in the tables above are for the
monthly alpha.

## Hyperparameters chosen inside each training window

Inner validation is the last 24 training months, separated from the
inner fit by the same embargo. The score being maximized is mean Spearman IC
against the raw next-month return. The final model is refit on the full training
window through the embargo cutoff; the test block is not used.

| Window | Model | Selection | Inner rank IC |
| --- | --- | --- | --- |
| 2012-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2012-01 | ridge | alpha=0.1 | 0.017 |
| 2012-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20 | 0.023 |
| 2012-01 | mlp | dropout=0.1, rank_lambda=0.2, epochs=8 | 0.030 |
| 2013-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2013-01 | ridge | alpha=100.0 | 0.029 |
| 2013-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20 | 0.040 |
| 2013-01 | mlp | dropout=0.1, rank_lambda=0.2, epochs=4 | 0.033 |
| 2014-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2014-01 | ridge | alpha=1.0 | 0.038 |
| 2014-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50 | 0.028 |
| 2014-01 | mlp | dropout=0.3, rank_lambda=0.2, epochs=8 | 0.024 |
| 2015-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2015-01 | ridge | alpha=10.0 | 0.015 |
| 2015-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20 | 0.012 |
| 2015-01 | mlp | dropout=0.3, rank_lambda=0.2, epochs=5 | 0.007 |
| 2016-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2016-01 | ridge | alpha=0.1 | 0.031 |
| 2016-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20 | 0.032 |
| 2016-01 | mlp | dropout=0.1, rank_lambda=0.2, epochs=6 | 0.031 |
| 2017-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2017-01 | ridge | alpha=0.1 | 0.044 |
| 2017-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20 | 0.034 |
| 2017-01 | mlp | dropout=0.1, rank_lambda=0.0, epochs=7 | 0.036 |
| 2018-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2018-01 | ridge | alpha=1.0 | 0.033 |
| 2018-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50 | 0.034 |
| 2018-01 | mlp | dropout=0.3, rank_lambda=0.2, epochs=8 | 0.049 |
| 2019-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2019-01 | ridge | alpha=0.1 | 0.023 |
| 2019-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50 | 0.031 |
| 2019-01 | mlp | dropout=0.1, rank_lambda=0.2, epochs=6 | 0.036 |
| 2020-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2020-01 | ridge | alpha=0.1 | 0.046 |
| 2020-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50 | 0.048 |
| 2020-01 | mlp | dropout=0.1, rank_lambda=0.2, epochs=6 | 0.051 |
| 2021-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2021-01 | ridge | alpha=100.0 | 0.048 |
| 2021-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50 | 0.056 |
| 2021-01 | mlp | dropout=0.1, rank_lambda=0.2, epochs=6 | 0.078 |
| 2022-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2022-01 | ridge | alpha=100.0 | 0.029 |
| 2022-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20 | 0.025 |
| 2022-01 | mlp | dropout=0.1, rank_lambda=0.2, epochs=8 | 0.033 |
| 2023-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2023-01 | ridge | alpha=100.0 | 0.004 |
| 2023-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20 | 0.012 |
| 2023-01 | mlp | dropout=0.1, rank_lambda=0.0, epochs=4 | 0.018 |
| 2024-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2024-01 | ridge | alpha=100.0 | 0.014 |
| 2024-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50 | 0.020 |
| 2024-01 | mlp | dropout=0.3, rank_lambda=0.2, epochs=7 | 0.021 |
| 2025-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2025-01 | ridge | alpha=100.0 | 0.032 |
| 2025-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50 | 0.028 |
| 2025-01 | mlp | dropout=0.1, rank_lambda=0.2, epochs=5 | 0.027 |
| 2026-01 | momentum | rule=equal-weight mom minus reversal | n/a |
| 2026-01 | ridge | alpha=100.0 | 0.006 |
| 2026-01 | lgbm | n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=20 | 0.013 |
| 2026-01 | mlp | dropout=0.1, rank_lambda=0.2, epochs=8 | 0.016 |

## Figures

- `figures/cumulative_ls_net.png`
- `figures/cumulative_ls_gross.png`
- `figures/rolling_rank_ic.png`
- `figures/decile_bars.png`
- `figures/turnover.png`
- `figures/drawdown.png`
- `figures/feature_importance.png`

Permutation importance shuffles each feature inside the test month and records the
drop in rank IC. It is an explanation of the fitted models, not a feature-selection
step. The feature list was fixed before the backtest.
