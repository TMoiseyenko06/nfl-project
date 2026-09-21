# Convenience targets. Everything here is also available via `nflpred <cmd>`.
PY := .venv/bin/python
CLI := .venv/bin/nflpred

.PHONY: setup fetch features backtest test audit predict score clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -e .
	.venv/bin/pip install -q pytest

# The neural network is optional: torch is a large install and everything else
# works without it.
setup-neural:
	.venv/bin/pip install -q torch

fetch:
	$(CLI) fetch

features:
	$(CLI) build-features

backtest:
	$(CLI) backtest --by-season

test:
	$(PY) -m pytest tests/ -q

# The leakage audit on its own - the gate that makes every other number mean something.
leakage:
	$(PY) -m pytest tests/test_leakage.py -v

audit:
	$(CLI) audit

predict:
	$(CLI) predict-week

score:
	$(CLI) score

clean:
	rm -rf data/cache/features artifacts/*.parquet .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
