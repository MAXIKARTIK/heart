# Makefile for the heart-disease-prediction test suite.
#
# Targets:
#   make test      Run the complete suite (Python + frontend)
#   make test-py   Run the heart_ml library + backend service pytest suite
#   make test-ui   Run the frontend Vitest suite
#
# PYTHON defaults to the project virtualenv when it exists so the configured
# pytest testpaths (tests/ + backend/tests) and the >=100-example Hypothesis
# profile from conftest.py are picked up automatically. Override any variable on
# the command line, e.g. `make test-py PYTHON=python3`.

PYTHON       ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PYTEST       ?= $(PYTHON) -m pytest
PYTEST_ARGS  ?=
FRONTEND_DIR ?= frontend

.DEFAULT_GOAL := help
.PHONY: help test test-py test-ui

help:
	@echo "heart-disease-prediction test targets:"
	@echo "  make test      Run the complete suite (Python + frontend)"
	@echo "  make test-py   Run the heart_ml + backend pytest suite"
	@echo "  make test-ui   Run the frontend Vitest suite"

# Python suite: heart_ml library tests + backend service tests. The test roots
# (tests/, backend/tests) are configured via [tool.pytest.ini_options] in
# pyproject.toml, so no path arguments are needed here.
test-py:
	$(PYTEST) $(PYTEST_ARGS)

# Frontend component suite: a single (non-watch) Vitest run.
test-ui:
	cd $(FRONTEND_DIR) && npm test

# Full suite: Python first, then the frontend. Stops on the first failure.
test: test-py test-ui
