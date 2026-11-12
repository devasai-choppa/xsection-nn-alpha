"""Render the Manim scenes and the README GIFs.

Final scenes are 1080p60. Set ``XSNN_MANIM_QUALITY=l`` to iterate at 480p15.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from xsnn.config import load_config, project_root
from xsnn.log import setup_logging
from xsnn.viz.animations import build_all_gifs
from xsnn.viz.loaders import export_inference_checkpoints

SCENES = [
    "DataToFeatures",
    "ForwardPass",
    "RankingToPortfolio",
    "WalkForward",
    "LearningOverEpochs",
]


def render_scenes(quality: str) -> None:
    root = project_root()
    scene_file = root / "src" / "xsnn" / "viz" / "manim_scenes.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["XSNN_ROOT"] = str(root)
    for scene in SCENES:
        logging.getLogger(__name__).info("Rendering %s at quality %s", scene, quality)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "manim",
                f"-q{quality}",
                "--media_dir",
                str(root / "media"),
                str(scene_file),
                scene,
            ],
            cwd=root,
            env=env,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render explainer scenes and GIFs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quality", default=os.environ.get("XSNN_MANIM_QUALITY", "h"))
    parser.add_argument("--skip-manim", action="store_true")
    parser.add_argument("--skip-gifs", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(Path(cfg["outputs"]["logs"]) / "animate.log")
    export_inference_checkpoints(cfg)
    if not args.skip_manim:
        render_scenes(args.quality)
    if not args.skip_gifs:
        build_all_gifs(cfg)


if __name__ == "__main__":
    main()
