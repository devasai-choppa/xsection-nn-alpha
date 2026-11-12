"""Turn walk-forward predictions into metrics, figures, and results.md."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from xsnn.backtest import run_backtest
from xsnn.metrics import (
    decile_monotonicity,
    factor_attribution,
    ic_by_date,
    summarize_ic,
    summarize_returns,
)
from xsnn.plots import MODEL_LABELS, write_all_figures

logger = logging.getLogger(__name__)


def _fmt_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def _fmt_num(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def evaluate_predictions(
    predictions: pd.DataFrame,
    factors: pd.DataFrame,
    importance: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Backtest every model, gross and net, long/short and long-only."""
    cost_bps = float(cfg["backtest"]["cost_bps"])
    n_deciles = int(cfg["backtest"]["n_deciles"])
    min_names = int(cfg["backtest"]["min_names"])
    hac_lags = int(cfg["backtest"].get("hac_lags", 3))
    metric_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    decile_frames: list[pd.DataFrame] = []
    ic_frames: list[pd.DataFrame] = []

    for model, group in predictions.groupby("model", sort=False):
        result = run_backtest(
            group,
            score_col="score",
            return_col="fwd_ret",
            n_deciles=n_deciles,
            cost_bps=cost_bps,
            min_names=min_names,
        )
        pearson = ic_by_date(group["score"], group["fwd_ret"], group["date"], method="pearson")
        spearman = ic_by_date(group["score"], group["fwd_ret"], group["date"], method="spearman")
        ic_frame = pd.DataFrame(
            {
                "signal_date": pearson.index,
                "ic": pearson.to_numpy(),
                "rank_ic": spearman.reindex(pearson.index).to_numpy(),
                "model": model,
            }
        )
        ic_frames.append(ic_frame)
        ic_stats = summarize_ic(ic_frame["ic"])
        rank_stats = summarize_ic(ic_frame["rank_ic"])
        mono = float("nan")
        if not result.deciles.empty:
            means = result.deciles.groupby("decile")["ret"].mean()
            mono = decile_monotonicity(means)
        books = {"long_short": result.long_short, "long_only": result.long_only}
        for book_name, book in books.items():
            if book.empty:
                logger.warning("Model %s book %s produced no months", model, book_name)
                continue
            indexed = book.set_index("signal_date").sort_index()
            realized = book.set_index("realized_date").sort_index()
            monthly_frames.append(book.assign(model=model, book=book_name))
            for kind in ("gross", "net"):
                stats = summarize_returns(indexed[kind], turnover=indexed["turnover"])
                attribution = {
                    "alpha_monthly": float("nan"),
                    "alpha_annual": float("nan"),
                    "alpha_t": float("nan"),
                    "r2": float("nan"),
                }
                if kind == "net":
                    attribution = factor_attribution(
                        realized[kind],
                        factors,
                        excess=(book_name == "long_only"),
                        hac_lags=hac_lags,
                    )
                metric_rows.append(
                    {
                        "model": model,
                        "book": book_name,
                        "cost": kind,
                        **stats,
                        "ic_mean": ic_stats["mean"],
                        "ic_std": ic_stats["std"],
                        "ic_icir": ic_stats["icir"],
                        "ic_t": ic_stats["t_stat"],
                        "rank_ic_mean": rank_stats["mean"],
                        "rank_ic_std": rank_stats["std"],
                        "rank_ic_icir": rank_stats["icir"],
                        "rank_ic_t": rank_stats["t_stat"],
                        "decile_monotonicity": mono,
                        "alpha_monthly": attribution["alpha_monthly"],
                        "alpha_annual": attribution["alpha_annual"],
                        "alpha_t": attribution["alpha_t"],
                        "factor_r2": attribution["r2"],
                    }
                )
        if not result.deciles.empty:
            decile_frames.append(result.deciles.assign(model=model))

    metrics = pd.DataFrame(metric_rows)
    monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    deciles = pd.concat(decile_frames, ignore_index=True) if decile_frames else pd.DataFrame()
    ic = pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame()
    return {
        "metrics": metrics,
        "monthly": monthly,
        "deciles": deciles,
        "ic": ic,
        "importance": importance,
        "cost_bps": cost_bps,
    }


def _headline_table(metrics: pd.DataFrame, book: str, cost: str) -> pd.DataFrame:
    view = metrics.loc[(metrics["book"] == book) & (metrics["cost"] == cost)].copy()
    order = ["momentum", "ridge", "lgbm", "mlp"]
    view["model"] = pd.Categorical(view["model"], categories=order, ordered=True)
    view = view.sort_values("model")
    rows = []
    for row in view.itertuples(index=False):
        rows.append(
            {
                "Model": MODEL_LABELS.get(row.model, row.model),
                "Ann. return": _fmt_pct(row.ann_return),
                "Ann. vol": _fmt_pct(row.ann_vol),
                "Sharpe": _fmt_num(row.sharpe),
                "Sortino": _fmt_num(row.sortino),
                "Max DD": _fmt_pct(row.max_drawdown),
                "Calmar": _fmt_num(row.calmar),
                "Turnover": _fmt_num(row.turnover),
                "Hit rate": _fmt_pct(row.hit_rate),
                "Rank IC": _fmt_num(row.rank_ic_mean, 3),
                "ICIR": _fmt_num(row.rank_ic_icir, 2),
                "IC t": _fmt_num(row.rank_ic_t, 2),
                "Alpha (ann.)": _fmt_pct(row.alpha_annual) if cost == "net" else "—",
                "Alpha t": _fmt_num(row.alpha_t, 2) if cost == "net" else "—",
            }
        )
    return pd.DataFrame(rows)


def _ic_table(metrics: pd.DataFrame) -> pd.DataFrame:
    view = metrics.loc[(metrics["book"] == "long_short") & (metrics["cost"] == "net")].copy()
    rows = []
    for row in view.itertuples(index=False):
        rows.append(
            {
                "Model": MODEL_LABELS.get(row.model, row.model),
                "IC": _fmt_num(row.ic_mean, 3),
                "IC std": _fmt_num(row.ic_std, 3),
                "ICIR": _fmt_num(row.ic_icir, 2),
                "IC t": _fmt_num(row.ic_t, 2),
                "Rank IC": _fmt_num(row.rank_ic_mean, 3),
                "Rank IC std": _fmt_num(row.rank_ic_std, 3),
                "Rank ICIR": _fmt_num(row.rank_ic_icir, 2),
                "Rank IC t": _fmt_num(row.rank_ic_t, 2),
                "Decile Spearman": _fmt_num(row.decile_monotonicity, 2),
            }
        )
    return pd.DataFrame(rows)


def _hyper_lines(hyperparams: list[dict[str, Any]]) -> list[str]:
    if not hyperparams:
        return ["No hyperparameter log was attached to this run."]
    lines = ["| Window | Model | Selection | Inner rank IC |", "| --- | --- | --- | --- |"]
    for row in hyperparams:
        chosen = row.get("chosen", {})
        bits = []
        for key, value in chosen.items():
            if key == "inner_rank_ic":
                continue
            bits.append(f"{key}={value}")
        ic = chosen.get("inner_rank_ic", float("nan"))
        ic_text = _fmt_num(float(ic), 3) if isinstance(ic, int | float) else "n/a"
        lines.append(
            f"| {row.get('window', '')} | {row.get('model', '')} | {', '.join(bits)} | {ic_text} |"
        )
    return lines


def render_results_markdown(
    bundle: dict[str, Any],
    cfg: dict[str, Any],
    hyperparams: list[dict[str, Any]],
    *,
    predictions: pd.DataFrame,
) -> str:
    """Human-readable model comparison. Numbers are the walk-forward result as run."""
    metrics: pd.DataFrame = bundle["metrics"]
    generated = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M %Z")
    dates = pd.to_datetime(predictions["date"])
    n_names = predictions.groupby("date")["ticker"].nunique()
    cost_bps = bundle["cost_bps"]
    sections = [
        "# Walk-forward results",
        "",
        f"Generated {generated} from `{Path(cfg['_config_path']).name}`.",
        "",
        "These numbers are the expanding-window test blocks only. Hyperparameters were",
        "selected on a time-series split inside each training window. Nothing here was",
        "chosen by looking at the test period.",
        "",
        "## Sample",
        "",
        f"- Signal months: {dates.min().date()} to {dates.max().date()} ({dates.nunique()} months).",
        f"- Median names per month: {float(n_names.median()):.0f}.",
        f"- Costs: {cost_bps:.1f} bps per side on one-way turnover "
        "`0.5 * sum(|w_t - w_t-1|)`, charged from a flat book in the first month.",
        "- Long/short books are +1 in the top decile and -1 in the bottom decile, equal weight.",
        "- The one-month embargo drops the feature month immediately before each test block,",
        "  so a training label cannot end on the first test date.",
        "- Universe: current S&P 500 constituents unless a point-in-time file was supplied.",
        "  That default embeds survivorship bias.",
        "",
        f"## Long/short, net of {cost_bps:.0f} bps",
        "",
        _markdown_table(_headline_table(metrics, "long_short", "net")),
        "",
        "## Long/short, gross of costs",
        "",
        _markdown_table(_headline_table(metrics, "long_short", "gross")),
        "",
        f"## Long only, net of {cost_bps:.0f} bps",
        "",
        _markdown_table(_headline_table(metrics, "long_only", "net")),
        "",
        "## Information coefficients",
        "",
        "IC is the monthly Pearson correlation of the score with the raw next-month return.",
        "Rank IC is Spearman. ICIR is mean / sample standard deviation. The t-statistic is",
        "`mean / (std / sqrt(n))`. Decile Spearman correlates decile number with mean return.",
        "",
        _markdown_table(_ic_table(metrics)),
        "",
        "## Factor attribution",
        "",
        "Net monthly returns are regressed on Ken French FF5 + momentum with Newey-West",
        f"standard errors (lags = {int(cfg['backtest'].get('hac_lags', 3))}). Long/short alpha",
        "is the intercept of the spread. Long-only alpha uses return minus RF. Annual alpha",
        "compounds the monthly intercept. t-statistics in the tables above are for the",
        "monthly alpha.",
        "",
        "## Hyperparameters chosen inside each training window",
        "",
        "Inner validation is the last "
        f"{int(cfg['validation']['inner_valid_months'])} training months, separated from the",
        "inner fit by the same embargo. The score being maximized is mean Spearman IC",
        "against the raw next-month return. The final model is refit on the full training",
        "window through the embargo cutoff; the test block is not used.",
        "",
        *_hyper_lines(hyperparams),
        "",
        "## Figures",
        "",
        "- `figures/cumulative_ls_net.png`",
        "- `figures/cumulative_ls_gross.png`",
        "- `figures/rolling_rank_ic.png`",
        "- `figures/decile_bars.png`",
        "- `figures/turnover.png`",
        "- `figures/drawdown.png`",
        "- `figures/feature_importance.png`",
        "",
        "Permutation importance shuffles each feature inside the test month and records the",
        "drop in rank IC. It is an explanation of the fitted models, not a feature-selection",
        "step. The feature list was fixed before the backtest.",
        "",
    ]
    return "\n".join(sections)


def write_report(
    bundle: dict[str, Any],
    cfg: dict[str, Any],
    hyperparams: list[dict[str, Any]],
    predictions: pd.DataFrame,
) -> Path:
    """Write metrics, figures, and `reports/results.md`."""
    reports = Path(cfg["outputs"]["reports"])
    figures = Path(cfg["outputs"]["figures"])
    reports.mkdir(parents=True, exist_ok=True)
    bundle["monthly"].to_parquet(reports / "monthly_returns.parquet", index=False)
    bundle["deciles"].to_parquet(reports / "decile_returns.parquet", index=False)
    bundle["ic"].to_parquet(reports / "ic_series.parquet", index=False)
    bundle["importance"].to_parquet(reports / "permutation_importance.parquet", index=False)
    metrics_records = json.loads(bundle["metrics"].to_json(orient="records"))
    (reports / "metrics.json").write_text(json.dumps(metrics_records, indent=2), encoding="utf-8")
    write_all_figures(bundle["monthly"], bundle["ic"], bundle["deciles"], bundle["importance"], figures)
    text = render_results_markdown(bundle, cfg, hyperparams, predictions=predictions)
    path = reports / "results.md"
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return path
