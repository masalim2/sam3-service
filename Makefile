format:
	ruff format .
	ruff check --select I --fix .
lint:
	ruff check .
lint-fix:
	ruff check --fix .
