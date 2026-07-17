.PHONY: install test lint sandbox demo eval

install:
	python -m pip install -e '.[all]'

test:
	pytest -q

lint:
	ruff check src tests

sandbox:
	docker build -f Dockerfile.sandbox -t agentic-harness-sandbox:latest .

demo:
	agentic-harness run benchmarks/tasks/fizzbuzz.yaml --provider mock --sandbox local

eval:
	agentic-harness eval benchmarks/tasks --provider mock --sandbox local --parallel 3
