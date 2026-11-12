"""Interactive network diagrams for one stock's real forward pass."""

from __future__ import annotations

from typing import Any

import numpy as np


def _sample(count: int, keep: int) -> list[int]:
    if count <= keep:
        return list(range(count))
    return [int(round(value)) for value in np.linspace(0, count - 1, keep)]


def _hex(value: float, scale: float) -> str:
    clipped = float(np.clip(value / (scale + 1e-8), -1.0, 1.0))
    if clipped >= 0:
        return f"#{int(40 + 30 * (1 - clipped)):02x}{int(80 + 150 * clipped):02x}{70:02x}"
    return f"#{int(80 + 150 * -clipped):02x}{50:02x}{60:02x}"


def layer_bundle(
    state: dict[str, Any],
    feature_names: list[str],
    inputs: np.ndarray,
    layers: list[np.ndarray],
    output: float,
) -> dict[str, Any]:
    """Sampled neurons, real activations, and the weights that connect them."""
    weights = [
        state["hidden.0.0.weight"].detach().cpu().numpy(),
        state["hidden.1.0.weight"].detach().cpu().numpy(),
        state["hidden.2.0.weight"].detach().cpu().numpy(),
        state["head.weight"].detach().cpu().numpy(),
    ]
    shown = [
        list(range(weights[0].shape[1])),
        _sample(weights[0].shape[0], 10),
        _sample(weights[1].shape[0], 8),
        _sample(weights[2].shape[0], 8),
        [0],
    ]
    hidden = [np.asarray(layer, dtype=float) for layer in layers]
    values = [np.asarray(inputs, dtype=float), *hidden, np.array([output])]
    labels = [
        [feature_names[index].replace("_", " ") for index in shown[0]],
        [f"h1.{index}" for index in shown[1]],
        [f"h2.{index}" for index in shown[2]],
        [f"h3.{index}" for index in shown[3]],
        ["score"],
    ]
    nodes = []
    for layer, indices in enumerate(shown):
        scale = max(float(np.percentile(np.abs(values[layer]), 90)), 0.2)
        for slot, index in enumerate(indices):
            value = float(values[layer][index])
            nodes.append(
                {
                    "id": f"L{layer}_{index}",
                    "layer": layer,
                    "slot": slot,
                    "label": labels[layer][slot],
                    "activation": value,
                    "color": _hex(value, scale),
                    "size": 12 + 18 * min(1.0, abs(value) / scale),
                }
            )
    edges = []
    for layer, weight in enumerate(weights):
        scale = max(float(np.percentile(np.abs(weight), 90)), 1e-4)
        for out_index in shown[layer + 1]:
            for in_index in shown[layer]:
                value = float(weight[out_index, in_index])
                edges.append(
                    {
                        "source": f"L{layer}_{in_index}",
                        "target": f"L{layer + 1}_{out_index}",
                        "weight": value,
                        "width": 0.4 + 3.0 * min(1.0, abs(value) / scale),
                        "color": "#3DDC97" if value >= 0 else "#E85D4C",
                    }
                )
    return {"nodes": nodes, "edges": edges, "score": float(output)}


def plotly_network(bundle: dict[str, Any]):
    """Plotly scatter of the sampled MLP. Returns a figure."""
    import plotly.graph_objects as go

    xs, ys, text, color, size = [], [], [], [], []
    positions = {}
    for node in bundle["nodes"]:
        layer_nodes = [item for item in bundle["nodes"] if item["layer"] == node["layer"]]
        count = len(layer_nodes)
        y = 0.0 if count == 1 else 1.0 - 2.0 * node["slot"] / max(count - 1, 1)
        positions[node["id"]] = (node["layer"], y)
        xs.append(node["layer"])
        ys.append(y)
        text.append(f"{node['label']}<br>{node['activation']:+.3f}")
        color.append(node["color"])
        size.append(node["size"])
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for edge in bundle["edges"]:
        x0, y0 = positions[edge["source"]]
        x1, y1 = positions[edge["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"color": "#445", "width": 0.6}, hoverinfo="skip")
    )
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker={"size": size, "color": color, "line": {"width": 0.5, "color": "#ddd"}},
            text=text,
            hoverinfo="text",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        height=560,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=False,
        xaxis={"visible": False},
        yaxis={"visible": False},
        title="Sampled MLP. Wide layers show an even subset; brightness is the real activation.",
    )
    return figure


def pyvis_html(bundle: dict[str, Any]) -> str:
    """A pyvis graph of the same sampled network, as an HTML fragment."""
    from pyvis.network import Network

    net = Network(height="540px", width="100%", directed=True, bgcolor="#0e1117", font_color="#f5f5f5")
    net.barnes_hut(gravity=-1800, spring_length=90, spring_strength=0.04)
    for node in bundle["nodes"]:
        net.add_node(
            node["id"],
            label=node["label"],
            title=f"{node['label']}: {node['activation']:+.3f}",
            color=node["color"],
            size=node["size"],
            level=node["layer"],
        )
    for edge in bundle["edges"]:
        net.add_edge(
            edge["source"],
            edge["target"],
            color=edge["color"],
            width=edge["width"],
            title=f"{edge['weight']:+.3f}",
        )
    net.set_options(
        """
        var options = {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed",
              "levelSeparation": 160
            }
          },
          "physics": {"enabled": false},
          "edges": {"smooth": {"type": "cubicBezier"}}
        }
        """
    )
    return net.generate_html(notebook=False)
