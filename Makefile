.PHONY: install test verify demo bench figures all clean

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

verify:
	$(PY) scripts/verify_claims.py

demo:
	$(PY) scripts/demo.py

bench:
	$(PY) scripts/benchmark.py --seeds 3 --episodes 10

bench-fast:
	$(PY) scripts/benchmark.py --seeds 2 --episodes 4 --train-steps 800 --max-steps 150

bench-full:
	$(PY) scripts/benchmark.py --seeds 5 --episodes 20

figures:
	$(PY) scripts/make_figures.py

all: test verify bench figures

clean:
	rm -rf results/raw.csv results/summary.csv results/figures/
