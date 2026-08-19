.PHONY: install test data train-direction backtest

install:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

data:
	python scripts/run_data_pipeline.py

train-direction:
	python scripts/train_direction_model.py

backtest:
	python scripts/run_backtest.py
