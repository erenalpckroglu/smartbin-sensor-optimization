# Windows without make: copy the command after the colon line into the terminal.
.PHONY: setup demo smoke run test clean

setup:            ## install the package + test tools
	pip install -e ".[dev]"

demo:             ## synthetic fleet with known truth -> full pipeline -> reports/demo
	python -m smartbin demo

smoke:            ## 1-minute check that the code reads the REAL file format (random-row sample)
	python -m smartbin run --measurements data/raw/measurements_10mb.csv --reports reports/smoke

run:              ## the real thing: data/raw/measurements.csv (7.5M rows) -> reports/
	python -m smartbin run

test:             ## unit tests + end-to-end recovery test
	pytest -q

clean:
	rm -rf data/interim/* reports/smoke reports/demo
