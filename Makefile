# lint
lint:
	ruff check .
fix-lint: 
	ruff check --fix .

# test
tests: 
	uv run pytest -v tests/