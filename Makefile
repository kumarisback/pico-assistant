.PHONY: help setup models models-llm llama server chat mic test lint fmt clean

# 3.12 then 3.11 — onnxruntime/faster-whisper/piper-tts do not all ship wheels
# for 3.13+. Falling back to a bare python3 there installs mismatched versions
# silently, so fail loudly instead. Override with: make setup PY=/path/to/python
PY ?= $(shell command -v python3.12 || command -v python3.11)
VENV := .venv
BIN := $(VENV)/bin

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## create venv and install dependencies
	@test -n "$(PY)" || { \
	  echo "No python3.12 or python3.11 found."; \
	  echo "  macOS:  brew install python@3.12"; \
	  echo "  Ubuntu: sudo apt install python3.12-venv"; \
	  exit 1; }
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
