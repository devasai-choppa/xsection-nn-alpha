"""Build features, train the walk-forward models, and write the backtest report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from xsnn.config import load_config
from xsnn.log import setup_logging
from xsnn.pipeline import build_feature_store, run_backtest, run_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cross-sectional ranking pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", choices=("all", "features", "train", "backtest"), default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(Path(cfg["outputs"]["logs"]) / "pipeline.log")
    logger = logging.getLogger("xsnn.pipeline.cli")
    if args.stage in {"all", "features"}:
        logger.info("Building features")
        build_feature_store(cfg)
    if args.stage in {"all", "train"}:
        logger.info("Training walk-forward models")
        run_train(cfg)
    if args.stage in {"all", "backtest"}:
        logger.info("Running the backtest report")
        path = run_backtest(cfg)
        logger.info("Report written to %s", path)


if __name__ == "__main__":
    main()
