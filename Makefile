# svrptw — common developer workflows.
#
# Use: make <target>
# All targets assume `pip install -e .` has been run and that the
# Python venv is on PATH (or use `make PY=.venv/Scripts/python.exe`).

PY ?= python
PYTHONPATH := .

.PHONY: help test test-quick bench-quick bench-full report serve clean

help:
	@echo "svrptw developer targets:"
	@echo "  test         Run full unit test suite (~2 min, 177 tests)"
	@echo "  test-quick   Run cost-terms + public-API smoke only (~12 s)"
	@echo "  bench-quick  Run headline_results.py --quick (~6 min)"
	@echo "  bench-full   Run headline_results.py full mode (~60-90 min)"
	@echo "  report       Render Progress_Report.html via webui /reports/progress"
	@echo "  clean        Remove __pycache__ + bench/runs/*.tmp"

test:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m pytest tests/unit/ -q

test-quick:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m pytest tests/unit/test_cost_terms_smoke.py tests/unit/test_public_api_smoke.py -q

bench-quick:
	PYTHONPATH=$(PYTHONPATH) $(PY) bench/scripts/headline_results.py --quick

bench-full:
	PYTHONPATH=$(PYTHONPATH) $(PY) bench/scripts/headline_results.py

report:
	@echo "Open http://127.0.0.1:8765/reports/progress in your browser"
	@echo "(start webui first if needed: PYTHONPATH=. python -m webui.server)"

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find bench/runs -name '*.tmp' -delete 2>/dev/null || true
