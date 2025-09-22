# Makefile - Dev testing and cleaning

PY ?= python
MAKECMD ?= $(MAKE) --no-print-directory
TEST_DIR := src/tests
REPORTS_DIR := reports/test_result
LOGS_DIR := $(REPORTS_DIR)/logs

# Discover tests of the form src/tests/test_*.py
TEST_FILES := $(wildcard $(TEST_DIR)/test_*.py)
TEST_NAMES := $(patsubst $(TEST_DIR)/test_%.py,%,$(TEST_FILES))

.PHONY: help test test-all list-tests clean ensure-dirs \
	$(addprefix test-,$(TEST_NAMES)) run-dev up-prod down-prod logs

## Allow syntax: make test name1 name2 ...
ifeq (test,$(firstword $(MAKECMDGOALS)))
  GOALS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  DO_CLEAN := $(filter clean,$(MAKECMDGOALS))
  TESTS := $(filter-out clean,$(GOALS))
  # Prevent make from treating the TESTS as independent targets
  $(eval $(TESTS):;@:)
endif

help:
	@echo "Targets:"
	@echo "  make test [name ...]    Run all tests or specific tests by name (e.g., worker_labeling_strategy)"
	@echo "  make test-all           Run all discovered tests"
	@echo "  make test-<name>        Run specific test by name"
	@echo "  make list-tests         List discoverable test names"
	@echo "  make clean              Remove test artifacts (reports, logs, __pycache__)"
	@echo "  make test clean         Run tests then clean artifacts"
	@echo "  make run-dev            Start dev API (hot-reload) via docker-compose.dev.yml"
	@echo "  make up-prod            Start prod API (no code mount) via docker-compose.yml"
	@echo "  make down-prod          Stop prod services"
	@echo "  make logs               Tail API logs"
	@echo ""
	@echo "Available tests: $(TEST_NAMES)"

# Main test entry point
test: ensure-dirs
ifeq ($(strip $(TESTS)),)
	@echo "Running all tests: $(TEST_NAMES)"
	@$(MAKE) --no-print-directory test-all
else
	@echo "Running specified tests: $(TESTS)"
	@$(foreach t,$(TESTS), \
		echo "==> Running test: $(t)"; \
		if [ -f "$(TEST_DIR)/test_$(t).py" ]; then \
			$(PY) "$(TEST_DIR)/test_$(t).py" || (echo "Test $(t) failed" && exit 1); \
		else \
			echo "Error: No test named '$(t)' found in $(TEST_DIR)"; \
			exit 1; \
		fi; \
	)
	@echo "All specified tests completed successfully"
endif
ifdef DO_CLEAN
	@$(MAKE) --no-print-directory clean
endif

# Run every discovered test_*.py
test-all: ensure-dirs
	@echo "Running all discovered tests..."
	@failed_tests=""; \
	for t in $(TEST_NAMES); do \
		echo "==> Running test: $$t"; \
		if $(PY) "$(TEST_DIR)/test_$$t.py"; then \
			echo "    ✓ $$t passed"; \
		else \
			echo "    ✗ $$t failed"; \
			failed_tests="$$failed_tests $$t"; \
		fi; \
	done; \
	if [ -n "$$failed_tests" ]; then \
		echo ""; \
		echo "Failed tests:$$failed_tests"; \
		exit 1; \
	else \
		echo ""; \
		echo "All tests passed successfully!"; \
	fi

# Pattern target: make test-<name> runs src/tests/test_<name>.py
test-%: ensure-dirs
	@echo "==> Running test: $*"
	@if [ -f "$(TEST_DIR)/test_$*.py" ]; then \
		$(PY) "$(TEST_DIR)/test_$*.py" && echo "    ✓ $* passed" || (echo "    ✗ $* failed" && exit 1); \
	else \
		echo "Error: No test named '$*' found in $(TEST_DIR)"; \
		echo "Available tests: $(TEST_NAMES)"; \
		exit 1; \
	fi

list-tests:
	@echo "Discovered test files:"
	@$(foreach t,$(TEST_NAMES),echo "  $(t)";)
	@echo ""
	@echo "Usage examples:"
	@echo "  make test $(word 1,$(TEST_NAMES))"
	@echo "  make test-$(word 1,$(TEST_NAMES))"

ensure-dirs:
	@$(PY) -c "import os; os.makedirs('$(LOGS_DIR)', exist_ok=True)" 2>/dev/null || true

clean:
	@echo "Cleaning test artifacts..."
	@$(PY) -c "import shutil, os; shutil.rmtree('$(REPORTS_DIR)', ignore_errors=True); os.makedirs('$(LOGS_DIR)', exist_ok=True)"
	@echo "  ✓ Cleaned reports directory"
	@$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	@echo "  ✓ Removed __pycache__ directories"
	@echo "Cleanup completed."

# Debug target to show parsed variables
debug:
	@echo "TEST_FILES: $(TEST_FILES)"
	@echo "TEST_NAMES: $(TEST_NAMES)"
	@echo "MAKECMDGOALS: $(MAKECMDGOALS)"
ifdef GOALS
	@echo "GOALS: $(GOALS)"
	@echo "TESTS: $(TESTS)"
	@echo "DO_CLEAN: $(DO_CLEAN)"
endif

dev-up:
	docker compose --profile dev up -d
dev-down:
	docker compose --profile dev down
dev-logs:
	docker compose --profile dev logs -f

prod-up:
	docker compose --profile prod up -d
prod-down:
	docker compose --profile prod down
prod-logs:
	docker compose --profile prod logs -f