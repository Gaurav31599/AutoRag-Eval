# Thin wrappers over `python -m rag ...`. On Windows without make installed,
# run the underlying python commands directly (shown in the README).

.PHONY: setup ingest rag

setup:
	python -m pip install -r requirements.txt

ingest:
	python -m rag ingest

# Usage: make rag q="who wrote the theme for the 1966 batman tv series"
rag:
	python -m rag "$(q)"
