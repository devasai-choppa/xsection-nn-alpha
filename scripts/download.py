"""Download the raw cache. Existing parquet files are not fetched again."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from xsnn.config import load_config
from xsnn.data import download_all
from xsnn.log import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache S&P 500 prices and Ken French factors")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(Path(cfg["outputs"]["logs"]) / "download.log")
    logging.getLogger(__name__).info("Downloading into %s", cfg["data"]["cache_dir"])
    download_all(cfg)


if __name__ == "__main__":
    main()
