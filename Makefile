PY ?= .venv/bin/python
UV ?= uv

.PHONY: setup build test lint eval eval-llm api charts powerbi docker clean

setup:            ## create venv and install everything
	$(UV) sync --locked --extra llm --extra dev

build:            ## generate data, run every analysis, write data/results.json
	$(PY) -m rml.cli build

test:             ## unit + integration tests (offline, mock LLM)
	$(PY) -m pytest

lint:             ## ruff + format check + mypy
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

eval:             ## agent eval on golden questions with the offline mock provider
	$(PY) -m rml.evals.run_evals --provider mock --min-pass-rate 0.95

eval-llm:         ## agent eval with a real model (set RML_PROVIDER credentials and RML_MODEL)
	$(PY) -m rml.evals.run_evals --provider $${RML_PROVIDER:-anthropic}

api:              ## REST API on :8000
	$(PY) -m uvicorn rml.api:app --reload --port 8000

charts:           ## README charts
	$(PY) -m rml.cli charts

powerbi:          ## Power BI star schema CSVs in powerbi/data
	$(PY) -m rml.cli export-powerbi

docker:
	docker build -t retail-media-lab .

clean:
	rm -rf data .pytest_cache .mypy_cache .ruff_cache powerbi/data
