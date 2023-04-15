.PHONY: install
install: ## Install the poetry environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment and install dependencies"
	@python3 -m venv .venv
	@. .venv/bin/activate
	@pip install -r requirements-dev.txt
	
.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Linting code: Running ruff"
	@ruff check . --force-exclude
	@echo "🚀 Static type checking: Running mypy"
	@mypy .
	@echo "🚀 Autoformat: Running black"
	@black .