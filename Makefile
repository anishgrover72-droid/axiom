.PHONY: install dev lint fmt test smoke data traces compress sft prm-label prm-train grpo eval serve clean

install:        ## full GPU stack
	pip install -e . && pip install -r requirements.txt

dev:            ## CPU-only subset for working on the contracts
	pip install -e . pydantic hydra-core omegaconf numpy ruff pytest

lint:
	ruff check src tests && ruff format --check src tests

fmt:
	ruff format src tests && ruff check --fix src tests

test:
	pytest -m "not slow"

smoke:          ## end-to-end sanity on a toy example (G0)
	python scripts/07_eval.py eval=smoke

data:    ; python scripts/00_download_data.py
traces:  ; python scripts/01_build_traces.py
compress:; python scripts/02_compress.py
sft:     ; python scripts/03_sft.py
prm-label:; python scripts/04_prm_label.py
prm-train:; python scripts/05_prm_train.py
grpo:    ; python scripts/06_grpo.py
eval:    ; python scripts/07_eval.py
serve:   ; python scripts/08_serve.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + ; rm -rf .pytest_cache .ruff_cache
