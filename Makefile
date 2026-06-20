.PHONY: install train score serve verify help

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PIP := $(if $(wildcard .venv/bin/pip),.venv/bin/pip,pip)

# Default target
help:
	@echo "East Africa Credit Scoring — available targets:"
	@echo "  make install   Create venv and install dependencies"
	@echo "  make train     Train model (required before score/serve)"
	@echo "  make score     Score sample applicants (CLI + SHAP + audits)"
	@echo "  make serve     Start FastAPI on http://127.0.0.1:8000"
	@echo "  make verify    Train + score (offline smoke test)"

install:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) train.py

score:
	$(PYTHON) score.py

serve:
	$(PYTHON) serve.py

verify: train score
	@echo "==> Offline verify complete. Run 'make serve' and curl /health for live API check."
