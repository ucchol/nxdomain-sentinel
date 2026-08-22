.PHONY: install data test train eval analyze export bench serve docker clean

install:
	pip install -e ".[dev,serve]" --extra-index-url https://download.pytorch.org/whl/cpu

data:
	python scripts/build_dataset.py

test:
	python -m pytest tests/ -q

train:
	python scripts/train.py --folds 0 1 2 3 4 --epochs 12

eval:
	python scripts/evaluate.py --folds 0 1 2 3 4

analyze:
	python scripts/analyze.py

export:
	python scripts/export_model.py --fold 0

bench:
	python bench/latency.py

serve:
	uvicorn nxsentinel.serve.app:app --reload --port 8080 --app-dir src

docker:
	docker build -t nxdomain-sentinel:local .
	docker run --rm -p 8080:8080 nxdomain-sentinel:local

clean:
	rm -rf artifacts/*.pt data/processed .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
