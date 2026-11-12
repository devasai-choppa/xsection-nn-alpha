PYTHON ?= .venv/bin/python
CONFIG ?= configs/default.yaml
export PYTHONPATH := src:$(PYTHONPATH)
export XSNN_ROOT := $(CURDIR)

.PHONY: data train backtest test lint animate app video all

data:
	$(PYTHON) scripts/download.py --config $(CONFIG)

train:
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --stage train

backtest:
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --stage backtest

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts app

animate:
	$(PYTHON) scripts/render_animations.py --config $(CONFIG)

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.headless true --server.port 8501

video:
	$(PYTHON) scripts/make_demo_video.py --config $(CONFIG)

all: data train backtest test animate
