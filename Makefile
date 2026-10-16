.PHONY: help setup models models-llm llama server chat mic test lint fmt clean

PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## create venv and install dependencies
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip wheel
	$(BIN)/pip install -r requirements.txt
	bash scripts/download-models.sh

models:  ## download VAD + TTS models
	bash scripts/download-models.sh

models-llm:  ## download the Qwen3-4B GGUF (2.4 GB)
	bash scripts/download-models.sh --llm

llama:  ## run llama-server (terminal 1)
	bash scripts/run-llama.sh

server:  ## run the assistant server (terminal 2)
	$(BIN)/python -m server.main

chat:  ## text client — fastest way to test the agent
	$(BIN)/python -m clients.chat_client

mic:  ## microphone client
	$(BIN)/python -m clients.mic_client

test:  ## run offline unit tests
	$(BIN)/python -m pytest

lint:  ## check style
	$(BIN)/ruff check .

fmt:  ## apply fixes and format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

clean:  ## remove caches (keeps models and data)
	find . -path ./$(VENV) -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
	rm -rf .pytest_cache .ruff_cache
