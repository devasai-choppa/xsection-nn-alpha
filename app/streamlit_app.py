"""Interactive walk-forward viewer for the cross-sectional MLP.

Run with ``make app`` from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xsnn.attribution import integrated_gradients_for_ticker  # noqa: E402
from xsnn.backtest import assign_deciles  # noqa: E402
from xsnn.metrics import summarize_returns  # noqa: E402
from xsnn.viz.loaders import (  # noqa: E402
    activation_dates,
    artifacts,
    decile_returns,
    ic_series,
    load_activation,
    load_window_model,
    metrics_table,
    monthly_returns,
    predictions,
    universe,
    window_meta,
)
from xsnn.viz.network import layer_bundle, plotly_network, pyvis_html  # noqa: E402

MODEL_LABELS = {
    "momentum": "Momentum + reversal",
    "ridge": "Ridge",
    "lgbm": "LightGBM",
    "mlp": "MLP ensemble",
}
COLORS = {"momentum": "#9aa0a6", "ridge": "#4C78A8", "lgbm": "#E09F3E", "mlp": "#E85D4C"}


@st.cache_data(show_spinner=False)
def _cfg() -> dict:
    return artifacts()


@st.cache_data(show_spinner=False)
def _predictions() -> pd.DataFrame:
    return predictions(_cfg())


@st.cache_data(show_spinner=False)
def _monthly() -> pd.DataFrame:
    return monthly_returns(_cfg())


@st.cache_data(show_spinner=False)
def _deciles() -> pd.DataFrame:
    return decile_returns(_cfg())


@st.cache_data(show_spinner=False)
def _ic() -> pd.DataFrame:
    return ic_series(_cfg())


@st.cache_data(show_spinner=False)
def _names() -> pd.DataFrame:
    return universe(_cfg())


@st.cache_resource(show_spinner=False)
def _model(window_id: str):
    return load_window_model(_cfg(), window_id)


def _net_returns(model: str, book: str, cost_bps: float) -> pd.DataFrame:
    frame = _monthly()
    frame = frame.loc[(frame["model"] == model) & (frame["book"] == book)].sort_values("signal_date")
    out = frame.copy()
    out["net"] = out["gross"] - out["turnover"] * (cost_bps / 10_000.0)
    return out


def _wealth_figure(cost_bps: float, book: str) -> go.Figure:
    figure = go.Figure()
    for model in ("momentum", "ridge", "lgbm", "mlp"):
        series = _net_returns(model, book, cost_bps)
        wealth = (1.0 + series["net"]).cumprod()
        figure.add_trace(
            go.Scatter(
                x=series["signal_date"],
                y=wealth,
                name=MODEL_LABELS[model],
                line={"color": COLORS[model], "width": 2.6 if model == "mlp" else 1.6},
            )
        )
    figure.update_layout(
        template="plotly_dark",
        height=460,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        yaxis_title="Growth of $1",
        title=f"{'Long/short' if book == 'long_short' else 'Long only'}, net of {cost_bps:.0f} bps per side",
        legend={"orientation": "h"},
    )
    return figure


def _rank_tab() -> None:
    dates = activation_dates(_cfg())
    if not dates:
        st.error("Activation dumps are missing. Run `make train` first.")
        return
    labels = [date.strftime("%Y-%m-%d") for date in dates]
    chosen = st.select_slider("Signal month", options=labels, value=labels[len(labels) // 2])
    date = pd.Timestamp(chosen)
    blob = load_activation(_cfg(), date)
    tickers = np.asarray(blob["tickers"]).astype(str)
    scores = np.asarray(blob["ensemble_outputs"], dtype=float)
    frame = pd.DataFrame({"ticker": tickers, "score": scores, "features": list(blob["inputs"])})
    # Realized return is joined from the prediction file when this month was scored.
    preds = _predictions()
    realized = preds.loc[(preds["model"] == "mlp") & (preds["date"] == date), ["ticker", "fwd_ret"]]
    frame = frame.merge(realized, on="ticker", how="left")
    frame["decile"] = assign_deciles(frame["score"], 10).to_numpy()
    names = _names()
    if "name" in names.columns:
        frame = frame.merge(names[["ticker", "name", "sector"]], on="ticker", how="left")
    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    longs = frame.loc[frame["decile"] == 10]
    shorts = frame.loc[frame["decile"] == 1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Names", f"{len(frame)}")
    c2.metric("Long book", f"{len(longs)} names")
    c3.metric("Short book", f"{len(shorts)} names")
    left, right = st.columns([1.1, 1.2])
    with left:
        st.subheader("Ranked universe")
        show = frame.drop(columns=["features"]).copy()
        show["score"] = show["score"].map(lambda value: f"{value:+.3f}")
        if "fwd_ret" in show.columns:
            show["fwd_ret"] = show["fwd_ret"].map(lambda value: "" if pd.isna(value) else f"{value:+.2%}")
        st.dataframe(show, height=520, hide_index=True, width="stretch")
    with right:
        st.subheader("Network at the selected name")
        ticker = st.selectbox("Ticker", frame["ticker"].tolist(), index=0)
        row = frame.loc[frame["ticker"] == ticker].iloc[0]
        meta = window_meta(_cfg(), date)
        model, payload = _model(str(meta.get("window_id", f"{date.year}-01")))
        index = int(np.where(tickers == ticker)[0][0])
        layers = [np.asarray(blob[f"layer_{i}"][index], dtype=float) for i in range(3)]
        output = float(blob["outputs"][index])
        bundle = layer_bundle(
            model.state_dict(),
            [str(name) for name in payload["feature_names"]],
            np.asarray(blob["inputs"][index], dtype=float),
            layers,
            output,
        )
        st.plotly_chart(plotly_network(bundle), width="stretch")
        with st.expander("Interactive graph (pyvis)"):
            html = pyvis_html(bundle)
            st.iframe(html, height=560)
        st.caption(
            f"Seed-0 score {output:+.3f}. Ensemble score {float(row['score']):+.3f}. "
            "Green edges are positive weights, red edges are negative. Wide layers are subsampled."
        )
        features = np.asarray(row["features"], dtype=float).reshape(1, -1)
        attributions = integrated_gradients_for_ticker(
            model,
            features,
            [str(name) for name in payload["feature_names"]],
        )
        order = attributions.abs().sort_values(ascending=True)
        figure = go.Figure(
            go.Bar(
                x=order.to_numpy(),
                y=order.index.astype(str),
                orientation="h",
                marker_color=["#3DDC97" if value >= 0 else "#E85D4C" for value in order.to_numpy()],
            )
        )
        figure.update_layout(
            template="plotly_dark",
            height=420,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
            title="Integrated gradients versus a median-rank baseline",
            xaxis_title="Contribution to the seed-0 score",
        )
        st.plotly_chart(figure, width="stretch")


def _backtest_tab() -> None:
    st.subheader("Costs are applied to the saved turnover, not refit")
    model = st.selectbox("Model", list(MODEL_LABELS), format_func=lambda key: MODEL_LABELS[key], index=3)
    book = st.radio(
        "Book",
        ["long_short", "long_only"],
        format_func=lambda key: "Long/short" if key == "long_short" else "Long only",
        horizontal=True,
    )
    cost = st.slider("Cost per side (bps)", min_value=0, max_value=50, value=10, step=1)
    series = _net_returns(model, book, float(cost))
    indexed = series.set_index("signal_date")
    stats = summarize_returns(indexed["net"], turnover=indexed["turnover"])
    published = metrics_table(_cfg())
    gross_row = published.loc[
        (published["model"] == model) & (published["book"] == book) & (published["cost"] == "gross")
    ]
    a, b, c, d = st.columns(4)
    a.metric("Ann. return", f"{stats['ann_return']:.1%}")
    b.metric("Sharpe", f"{stats['sharpe']:.2f}")
    c.metric("Max drawdown", f"{stats['max_drawdown']:.1%}")
    d.metric("Avg. turnover", f"{stats['turnover']:.2f}")
    if not gross_row.empty:
        st.caption(
            f"Gross Sharpe at the saved book is {float(gross_row.iloc[0]['sharpe']):.2f}. "
            "The slider only changes the cost drag."
        )
    st.plotly_chart(_wealth_figure(float(cost), book), width="stretch")
    deciles = _deciles()
    means = deciles.loc[deciles["model"] == model].groupby("decile")["ret"].mean()
    bars = go.Figure(go.Bar(x=means.index.astype(int), y=means.to_numpy(), marker_color=COLORS[model]))
    bars.update_layout(
        template="plotly_dark",
        height=320,
        title="Mean next-month return by score decile (gross)",
        xaxis_title="Decile",
        yaxis_title="Return",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )
    ic = _ic().loc[_ic()["model"] == model].sort_values("signal_date")
    rolled = ic["rank_ic"].rolling(12, min_periods=6).mean()
    line = go.Figure(go.Scatter(x=ic["signal_date"], y=rolled, line={"color": COLORS[model]}))
    line.update_layout(
        template="plotly_dark",
        height=320,
        title="12-month rolling rank IC",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )
    left, right = st.columns(2)
    left.plotly_chart(bars, width="stretch")
    right.plotly_chart(line, width="stretch")


def _video(path: Path, caption: str) -> None:
    if path.exists():
        st.video(str(path))
        st.caption(caption)
    else:
        st.info(f"{caption} has not been rendered yet. Run `make animate`.")


def _how_tab() -> None:
    media = ROOT / "media" / "videos" / "manim_scenes" / "1080p60"
    sections = [
        ("DataToFeatures.mp4", "Prices collapse into a rank-normalized feature."),
        ("ForwardPass.mp4", "A real forward pass for a top, middle, and bottom name."),
        ("WalkForward.mp4", "The expanding window, the embargo, and each refit."),
        ("RankingToPortfolio.mp4", "Decile books and the walk-forward equity curves."),
        ("LearningOverEpochs.mp4", "Layer-1 weights and training loss across epochs."),
    ]
    for name, caption in sections:
        st.subheader(caption)
        _video(media / name, name)
    st.markdown(
        "The network is a 12 → 64 → 32 → 16 → 1 MLP with SiLU, batch norm, and dropout. "
        "It is trained on the cross-sectional rank of next-month return. Five seeds are averaged "
        "for the portfolio; the diagram shows seed 0, whose activations were saved."
    )


def main() -> None:
    st.set_page_config(page_title="xsection-nn-alpha", layout="wide")
    st.title("xsection-nn-alpha")
    st.caption("Walk-forward cross-sectional ranking of current S&P 500 names. Not a live trading system.")
    rank, backtest, how = st.tabs(["Rank a month", "Backtest", "How it works"])
    with rank:
        _rank_tab()
    with backtest:
        _backtest_tab()
    with how:
        _how_tab()


main()
