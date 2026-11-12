"""Manim explainer scenes driven by the trained network and its activations.

Render with Manim Community. The scenes use ``Text`` rather than ``MathTex``,
so a LaTeX install is not required. Final quality is 1080p60 (``manim -qh``).

The five published scenes are ``DataToFeatures``, ``ForwardPass``,
``RankingToPortfolio``, ``WalkForward``, and ``LearningOverEpochs``.
``ForwardPassLoop`` is a short cycle of the forward pass used for the README GIF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from manim import (  # noqa: E402
    DOWN,
    GREEN,
    GREY,
    LEFT,
    RED,
    RIGHT,
    TEAL,
    UP,
    WHITE,
    YELLOW,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    ImageMobject,
    LaggedStart,
    Line,
    Rectangle,
    ReplacementTransform,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    VMobject,
    Write,
    always_redraw,
)

from xsnn.metrics import sharpe_ratio  # noqa: E402
from xsnn.models.mlp import load_mlp  # noqa: E402
from xsnn.viz.loaders import (  # noqa: E402
    artifacts,
    epoch_payloads,
    explainer_cross_section,
    monthly_prices,
    monthly_returns,
    panel,
)

BACKGROUND = "#101418"
MODEL_COLORS = {
    "momentum": GREY,
    "ridge": TEAL,
    "lgbm": "#E09F3E",
    "mlp": "#E85D4C",
}
MODEL_LABELS = {
    "momentum": "Momentum",
    "ridge": "Ridge",
    "lgbm": "LightGBM",
    "mlp": "MLP",
}


def _sample_indices(count: int, keep: int) -> list[int]:
    if count <= keep:
        return list(range(count))
    return [int(round(value)) for value in np.linspace(0, count - 1, keep)]


def _price_curve(series) -> VMobject:
    values = series.to_numpy(dtype=float)
    values = values / values[0]
    xs = np.linspace(0.0, 1.55, len(values))
    lo, hi = float(values.min()), float(values.max())
    ys = (values - lo) / (hi - lo + 1e-8) * 0.5
    curve = VMobject(color=TEAL, stroke_width=2.2)
    curve.set_points_as_corners([np.array([x, y, 0.0]) for x, y in zip(xs, ys, strict=True)])
    return curve


def _signed_hex(value: float, scale: float) -> str:
    clipped = float(np.clip(value / (scale + 1e-8), -1.0, 1.0))
    if clipped >= 0:
        red = int(80 + 150 * clipped)
        green = int(90 + 140 * clipped)
        blue = int(90 - 40 * clipped)
    else:
        red = int(80 + 160 * (-clipped))
        green = int(80 - 30 * (-clipped))
        blue = int(90 - 20 * (-clipped))
    red_i = max(0, min(255, red))
    green_i = max(0, min(255, green))
    blue_i = max(0, min(255, blue))
    return f"#{red_i:02x}{green_i:02x}{blue_i:02x}"


class DataToFeatures(Scene):
    """Thirty liquid names: prices draw, then 12-1 momentum is rank-normalized."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        cfg = artifacts()
        bundle = explainer_cross_section(cfg)
        date = bundle["date"]
        month = panel(cfg).xs(date, level="date")
        liquid = month["log_dollar_volume_raw"].dropna().sort_values(ascending=False)
        names: list[str] = []
        cells = VGroup()
        for ticker in liquid.index.astype(str):
            series = monthly_prices(cfg, ticker, date, months=18)
            if len(series) < 12:
                continue
            curve = _price_curve(series)
            label = Text(ticker, font_size=16, color=WHITE)
            label.next_to(curve, DOWN, buff=0.04)
            cells.add(VGroup(curve, label))
            names.append(ticker)
            if len(names) == 30:
                break
        if len(names) < 8:
            raise RuntimeError("Not enough price histories to draw the feature scene")
        cells.arrange_in_grid(rows=5, cols=6, buff=0.16)
        if cells.width > 12.5:
            cells.scale_to_fit_width(12.5)
        cells.shift(DOWN * 0.2)
        title = Text("Thirty large-caps, one month-end", font_size=34).to_edge(UP)
        stamp = Text(str(date.date()), font_size=22, color=GREY).next_to(title, DOWN, buff=0.08)
        self.play(Write(title), FadeIn(stamp))
        self.play(LaggedStart(*[Create(cell[0]) for cell in cells], lag_ratio=0.04), run_time=3.2)
        self.play(FadeIn(VGroup(*[cell[1] for cell in cells])), run_time=0.5)
        self.wait(0.4)

        raw = month.loc[names, "mom_12_1_raw"].astype(float)
        norm = month.loc[names, "mom_12_1"].astype(float)
        raw_bars = _feature_bars(names, raw.to_numpy(), signed=False)
        norm_order = np.argsort(norm.to_numpy())
        norm_bars = _feature_bars(
            [names[int(i)] for i in norm_order],
            norm.to_numpy()[norm_order],
            signed=True,
        )
        raw_title = Text("12-1 momentum, as a raw return", font_size=34).to_edge(UP)
        norm_title = Text("Rank-normalized to [-1, 1], re-sorted", font_size=34).to_edge(UP)
        self.play(
            FadeOut(cells),
            FadeOut(stamp),
            ReplacementTransform(title, raw_title),
            FadeIn(raw_bars),
            run_time=1.1,
        )
        self.wait(0.5)
        self.play(ReplacementTransform(raw_title, norm_title), ReplacementTransform(raw_bars, norm_bars), run_time=2.2)
        note = Text("Every bar uses only prices known on this date.", font_size=22, color=GREY).to_edge(DOWN)
        self.play(FadeIn(note))
        self.wait(1.0)


def _feature_bars(names: list[str], values: np.ndarray, *, signed: bool) -> VGroup:
    group = VGroup()
    span = 12.2
    gap = span / max(len(names), 1)
    if signed:
        scale = 1.7
    else:
        lo, hi = float(np.min(values)), float(np.max(values))
        scale = 2.4 / (hi - lo + 1e-8)
    for index, (_name, value) in enumerate(zip(names, values, strict=True)):
        x = -span / 2 + gap * (index + 0.5)
        number = float(value)
        if signed:
            height = max(0.04, abs(number) * scale)
            color = GREEN if number >= 0 else RED
            y = np.sign(number) * height / 2
        else:
            lo = float(np.min(values))
            height = max(0.05, (number - lo) * scale)
            color = TEAL
            y = -1.4 + height / 2
        bar = Rectangle(
            width=gap * 0.72,
            height=height,
            stroke_width=0,
            fill_color=color,
            fill_opacity=0.9,
        )
        bar.move_to(np.array([x, y, 0.0]))
        group.add(bar)
    group.shift(DOWN * 0.2)
    return group


class _NetworkView:
    """Neurons and edges for a subset of a real MLP."""

    def __init__(self, state: dict, feature_names: list[str]) -> None:
        self.weights = [
            state["hidden.0.0.weight"].detach().cpu().numpy(),
            state["hidden.1.0.weight"].detach().cpu().numpy(),
            state["hidden.2.0.weight"].detach().cpu().numpy(),
            state["head.weight"].detach().cpu().numpy(),
        ]
        self.shown = [
            list(range(self.weights[0].shape[1])),
            _sample_indices(self.weights[0].shape[0], 8),
            _sample_indices(self.weights[1].shape[0], 6),
            _sample_indices(self.weights[2].shape[0], 6),
            [0],
        ]
        self.xs = [-5.4, -2.7, 0.0, 2.7, 5.2]
        self.neurons: list[VGroup] = []
        self.edges = VGroup()
        self.labels = VGroup()
        self._build(feature_names)

    def _positions(self, count: int, x: float) -> list[np.ndarray]:
        if count == 1:
            return [np.array([x, 0.0, 0.0])]
        ys = np.linspace(2.5, -2.5, count)
        return [np.array([x, float(y), 0.0]) for y in ys]

    def _build(self, feature_names: list[str]) -> None:
        edge_weights = []
        positions: list[list[np.ndarray]] = []
        for layer, (indices, x) in enumerate(zip(self.shown, self.xs, strict=True)):
            positions.append(self._positions(len(indices), x))
            circles = VGroup()
            for pos in positions[-1]:
                circles.add(Circle(radius=0.16, color=GREY, stroke_width=2).move_to(pos).set_fill(GREY, opacity=0.35))
            self.neurons.append(circles)
            if layer in {1, 2}:
                ellipsis = Text("...", font_size=28, color=GREY).next_to(circles, DOWN, buff=0.12)
                self.labels.add(ellipsis)
        for layer, weight in enumerate(self.weights):
            for out_i, out_idx in enumerate(self.shown[layer + 1]):
                for in_i, in_idx in enumerate(self.shown[layer]):
                    value = float(weight[out_idx, in_idx])
                    edge_weights.append(abs(value))
                    line = Line(positions[layer][in_i], positions[layer + 1][out_i], stroke_width=1.0)
                    line.set_stroke(_signed_hex(value, 0.2), width=1.2, opacity=0.45)
                    line.weight_value = value  # type: ignore[attr-defined]
                    self.edges.add(line)
        scale = float(np.percentile(edge_weights, 80)) if edge_weights else 1.0
        self.edge_scale = max(scale, 1e-4)
        for line in self.edges:
            value = float(line.weight_value)
            width = 0.6 + 3.2 * min(1.0, abs(value) / self.edge_scale)
            line.set_stroke(_signed_hex(value, self.edge_scale), width=width, opacity=0.55)
        for index, name in enumerate(feature_names):
            if index >= len(self.neurons[0]):
                break
            label = Text(_short_feature(name), font_size=14, color=GREY)
            label.next_to(self.neurons[0][index], LEFT, buff=0.08)
            self.labels.add(label)

    def group(self) -> VGroup:
        return VGroup(self.edges, *self.neurons, self.labels)

    def apply_activations(self, inputs: np.ndarray, layers: list[np.ndarray], output: float) -> None:
        values = [inputs, *layers, np.array([output], dtype=float)]
        scales = [max(1.0, float(np.percentile(np.abs(values[0]), 90)))]
        for layer_values in layers:
            scales.append(max(0.25, float(np.percentile(np.abs(layer_values), 90))))
        scales.append(max(0.5, abs(output)))
        shown_values = []
        for layer, indices in enumerate(self.shown):
            shown_values.append([float(values[layer][index]) for index in indices])
            for circle, value in zip(self.neurons[layer], shown_values[-1], strict=True):
                color = _signed_hex(value, scales[layer])
                opacity = 0.35 + 0.6 * min(1.0, abs(value) / scales[layer])
                circle.set_fill(color, opacity=opacity)
                circle.set_stroke(WHITE, width=1.5)


class ForwardPass(Scene):
    """Real weights and activations for a top, middle, and bottom decile name."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        self._run(hold=0.7)

    def _run(self, hold: float) -> None:
        cfg = artifacts()
        bundle = explainer_cross_section(cfg)
        model, payload = load_mlp(Path(_checkpoint(cfg, bundle["window_id"])))
        feature_names = [str(name) for name in payload["feature_names"]]
        view = _NetworkView(model.state_dict(), feature_names)
        diagram = view.group()
        diagram.scale(0.92).shift(DOWN * 0.15)
        title = Text("One forward pass, real weights", font_size=32).to_edge(UP)
        self.play(Write(title), FadeIn(diagram), run_time=1.2)
        activation = bundle["activation"]
        tickers = np.asarray(activation["tickers"]).astype(str)
        lookup = {ticker: index for index, ticker in enumerate(tickers)}
        score_label = Text(" ", font_size=26).to_edge(DOWN)
        self.add(score_label)
        for pick in bundle["picks"]:
            ticker = str(pick["ticker"])
            row = lookup[ticker]
            inputs = np.asarray(activation["inputs"][row], dtype=float)
            layers = [np.asarray(activation[f"layer_{i}"][row], dtype=float) for i in range(3)]
            score = float(activation["outputs"][row])
            view.apply_activations(inputs, layers, score)
            decile = int(pick["decile"])
            caption = Text(f"{ticker}   decile {decile}   score {score:+.3f}", font_size=26)
            caption.to_edge(DOWN)
            self.play(ReplacementTransform(score_label, caption), run_time=0.35)
            score_label = caption
            self._pulse(view)
            self.wait(hold)

    def _pulse(self, view: _NetworkView) -> None:
        for layer in range(4):
            start = view.neurons[layer][len(view.neurons[layer]) // 2].get_center()
            end = view.neurons[layer + 1][len(view.neurons[layer + 1]) // 2].get_center()
            pulse = VMobject(color=YELLOW, stroke_width=6)
            pulse.set_points_as_corners([start, end])
            self.play(Create(pulse), run_time=0.28)
            self.play(FadeOut(pulse), run_time=0.12)


class ForwardPassLoop(ForwardPass):
    """Shorter cycle of :class:`ForwardPass` so the README GIF can loop."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        self._run(hold=0.15)


def _short_feature(name: str) -> str:
    """Compact labels so twelve input names stay inside the frame."""
    aliases = {
        "mom_12_1": "mom 12-1",
        "mom_6_1": "mom 6-1",
        "mom_3_1": "mom 3-1",
        "rev_1m": "reversal",
        "vol_20d": "vol 20d",
        "vol_60d": "vol 60d",
        "downside_vol": "downside vol",
        "log_dollar_volume": "dollar volume",
        "amihud": "amihud",
        "beta_252d": "beta",
        "idio_vol": "idio vol",
        "max_daily_ret": "max day",
    }
    return aliases.get(name, name.replace("_", " "))


def _checkpoint(cfg: dict, window_id: str) -> str:
    from xsnn.viz.loaders import inference_checkpoint

    return str(inference_checkpoint(cfg, window_id))


class RankingToPortfolio(Scene):
    """Sort one cross-section, mark the decile books, then draw the equity curves."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        cfg = artifacts()
        bundle = explainer_cross_section(cfg)
        scores = np.asarray(bundle["activation"]["ensemble_outputs"], dtype=float)
        order = np.argsort(scores)
        column = VGroup()
        count = len(order)
        span = 6.4
        bar_h = span / count
        for rank in range(count):
            y = -span / 2 + bar_h * (rank + 0.5)
            rect = Rectangle(
                width=1.15,
                height=max(bar_h * 0.86, 0.004),
                stroke_width=0,
                fill_color=GREY,
                fill_opacity=0.85,
            )
            rect.move_to(np.array([-4.6, y, 0.0]))
            column.add(rect)
        title = Text(f"MLP scores on {bundle['date'].date()}", font_size=32).to_edge(UP)
        column.shift(UP * 2.5).set_opacity(0)
        self.play(Write(title))
        self.play(column.animate.shift(DOWN * 2.5).set_opacity(1), run_time=1.6)
        tail = max(1, int(round(count * 0.10)))
        self.play(
            VGroup(*list(column)[:tail]).animate.set_fill(RED, opacity=0.95),
            VGroup(*list(column)[-tail:]).animate.set_fill(GREEN, opacity=0.95),
            run_time=0.7,
        )
        long_label = Text("long", font_size=22, color=GREEN).next_to(column[-1], RIGHT, buff=0.12)
        short_label = Text("short", font_size=22, color=RED).next_to(column[0], RIGHT, buff=0.12)
        self.play(FadeIn(long_label), FadeIn(short_label), run_time=0.4)

        books = monthly_returns(cfg)
        books = books.loc[books["book"] == "long_short"].copy()
        wealth = {
            model: (1.0 + books.loc[books["model"] == model].sort_values("signal_date")["net"]).cumprod()
            for model in ("momentum", "ridge", "lgbm", "mlp")
        }
        hi = max(float(series.max()) for series in wealth.values())
        lines = VGroup()
        counters: list[tuple[str, float, ValueTracker]] = []
        origin_x, origin_y = -1.3, -2.6
        width, height = 6.3, 4.6
        for model, series in wealth.items():
            values = series.to_numpy(dtype=float)
            xs = np.linspace(0.0, width, len(values))
            ys = (values - 0.6) / (hi - 0.6 + 1e-8) * height
            line = VMobject(color=MODEL_COLORS[model], stroke_width=3.4 if model == "mlp" else 2.2)
            line.set_points_as_corners(
                [np.array([origin_x + x, origin_y + y, 0.0]) for x, y in zip(xs, ys, strict=True)]
            )
            lines.add(line)
            returns = books.loc[books["model"] == model].sort_values("signal_date")["net"]
            counters.append((model, float(sharpe_ratio(returns)), ValueTracker(0.0)))
        self.play(LaggedStart(*[Create(line) for line in lines], lag_ratio=0.25), run_time=3.2)
        row = VGroup()
        placeholders = []
        for model, sharpe, _tracker in counters:
            tag = Text(MODEL_LABELS[model], font_size=20, color=MODEL_COLORS[model])
            placeholder = Text(f"{sharpe:+.2f}", font_size=24, color=MODEL_COLORS[model])
            placeholders.append(placeholder)
            row.add(VGroup(tag, placeholder).arrange(RIGHT, buff=0.12))
        row.arrange(RIGHT, buff=0.42).move_to(np.array([2.15, -3.35, 0.0]))
        numbers = []
        for (model, _sharpe, tracker), placeholder in zip(counters, placeholders, strict=True):
            anchor = placeholder.get_center().copy()
            number = always_redraw(
                lambda tracker=tracker, model=model, anchor=anchor: Text(
                    f"{tracker.get_value():+.2f}",
                    font_size=24,
                    color=MODEL_COLORS[model],
                ).move_to(anchor)
            )
            numbers.append(number)
            placeholder.set_opacity(0)
        self.play(FadeIn(row), *[FadeIn(number) for number in numbers])
        self.play(*[tracker.animate.set_value(sharpe) for _model, sharpe, tracker in counters], run_time=1.4)
        self.wait(0.8)


class WalkForward(Scene):
    """Expanding train window, a one-month embargo, and a sliding test block."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        cfg = artifacts()
        root = Path(cfg["outputs"]["checkpoints"])
        metas = []
        for path in sorted(root.glob("*/meta.json")):
            import json

            metas.append(json.loads(path.read_text(encoding="utf-8")))
        if not metas:
            raise RuntimeError("Walk-forward metadata is missing. Run `make train` first.")
        title = Text("Expanding window, one-month embargo", font_size=32).to_edge(UP)
        legend = VGroup(
            Text("train", font_size=20, color=TEAL),
            Text("embargo", font_size=20, color=YELLOW),
            Text("test", font_size=20, color=GREEN),
        ).arrange(RIGHT, buff=0.5).next_to(title, DOWN, buff=0.15)
        axis = Line(np.array([-6.2, 0.0, 0.0]), np.array([6.2, 0.0, 0.0]), color=GREY)
        start = Text("2006", font_size=20, color=GREY).next_to(axis, DOWN, buff=0.15).align_to(axis, LEFT)
        end = Text("2026", font_size=20, color=GREY).next_to(axis, DOWN, buff=0.15).align_to(axis, RIGHT)
        self.play(Write(title), FadeIn(legend), Create(axis), FadeIn(start), FadeIn(end))
        t0 = np.datetime64("2006-01-31")
        t1 = np.datetime64("2026-10-31")

        def xpos(date_text: str) -> float:
            value = np.datetime64(date_text)
            frac = (value - t0) / (t1 - t0)
            return float(-6.2 + 12.4 * frac)

        current = VGroup()
        self.add(current)
        for meta in metas:
            train_end = str(meta["train_end"])
            test_start = str(meta["test_start"])
            test_end = str(meta["test_end_exclusive"])
            bars = VGroup(
                _span(xpos(str(meta["train_start"])), xpos(train_end), TEAL, 0.55),
                _span(xpos(train_end), xpos(test_start), YELLOW, 0.55),
                _span(xpos(test_start), xpos(test_end), GREEN, 0.55),
            )
            caption = Text(f"refit {meta['window_id']}", font_size=26, color=YELLOW).to_edge(DOWN)
            if len(current) == 0:
                self.play(FadeIn(bars), FadeIn(caption), run_time=0.45)
            else:
                self.play(ReplacementTransform(current, bars), FadeIn(caption), run_time=0.45)
            current = bars
            self.play(FadeOut(caption), run_time=0.15)
        self.wait(0.8)


def _span(x0: float, x1: float, color: str, y: float) -> Rectangle:
    width = max(0.06, x1 - x0)
    rect = Rectangle(width=width, height=0.42, stroke_width=0, fill_color=color, fill_opacity=0.9)
    rect.move_to(np.array([(x0 + x1) / 2 if x1 > x0 else x0, y, 0.0]))
    return rect


class LearningOverEpochs(Scene):
    """Layer-1 weights and training loss across the saved epoch checkpoints."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        cfg = artifacts()
        window_id = _richest_window(cfg)
        payloads = epoch_payloads(cfg, window_id)
        title = Text(f"Learning the first layer  ·  window {window_id}", font_size=32).to_edge(UP)
        self.play(Write(title))
        images = [_heatmap(payload["model_state"]["hidden.0.0.weight"].detach().cpu().numpy()) for payload in payloads]
        mobs = [ImageMobject(image).set_height(5.2).move_to(np.array([-3.3, -0.2, 0.0])) for image in images]
        caption = Text("64 neurons × 12 features", font_size=20, color=GREY).next_to(mobs[0], DOWN, buff=0.15)
        losses = [float(payload["train_loss"]) for payload in payloads]
        loss_title = Text("training loss", font_size=22, color=GREY).move_to(np.array([3.4, 2.3, 0.0]))
        points = _loss_points(losses)
        self.play(FadeIn(mobs[0]), FadeIn(caption), FadeIn(loss_title))
        drawn = VGroup()
        prev_epoch: Text | None = None
        for index in range(1, len(mobs)):
            segment = VMobject(color=YELLOW, stroke_width=3)
            segment.set_points_as_corners([points[index - 1], points[index]])
            epoch = Text(f"epoch {index + 1}", font_size=22).next_to(mobs[index], UP, buff=0.12)
            self.play(ReplacementTransform(mobs[index - 1], mobs[index]), Create(segment), run_time=0.55)
            drawn.add(segment)
            self.add(epoch)
            if prev_epoch is not None:
                self.remove(prev_epoch)
            prev_epoch = epoch
        self.wait(0.8)


def _richest_window(cfg: dict) -> str:
    root = Path(cfg["outputs"]["checkpoints"])
    best_name = ""
    best_count = -1
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        count = len(list(folder.glob("epoch_*.pt")))
        if count > best_count or (count == best_count and folder.name > best_name):
            best_name, best_count = folder.name, count
    if best_count < 1:
        raise RuntimeError("No epoch checkpoints were found")
    return best_name


def _heatmap(weight) -> np.ndarray:
    matrix = np.asarray(weight, dtype=float)
    limit = float(np.percentile(np.abs(matrix), 98))
    limit = max(limit, 1e-6)
    scaled = np.clip(matrix / limit, -1.0, 1.0)
    # Blue-gray-red image, nearest-neighbor upsampled so neurons stay visible.
    red = np.where(scaled >= 0, 40 + 200 * scaled, 40 + 20 * (1 + scaled))
    blue = np.where(scaled >= 0, 40 + 20 * (1 - scaled), 40 + 200 * (-scaled))
    green = 40 + 30 * (1 - np.abs(scaled))
    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return np.repeat(np.repeat(rgb, 8, axis=0), 18, axis=1)


def _loss_points(losses: list[float]) -> list[np.ndarray]:
    lo, hi = min(losses), max(losses)
    span = max(hi - lo, 1e-4)
    points = []
    for index, loss in enumerate(losses):
        x = 1.6 + 3.6 * index / max(len(losses) - 1, 1)
        y = -1.8 + 3.2 * (loss - lo) / span
        points.append(np.array([x, y, 0.0]))
    return points
