#!/usr/bin/env bash
# One-time environment setup. Linux / macOS / Git Bash.
# Windows PowerShell equivalents are in README.md.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"

if [ ! -d .venv ]; then
  echo "creating virtualenv"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  source .venv/Scripts/activate   # Git Bash on Windows
fi

python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

echo
echo "fetching models"
bash scripts/download-models.sh

cat <<'EOF'

setup complete.

next:
  1. bash scripts/download-models.sh --llm        # 2.4 GB of model weights
  2. bash scripts/run-llama.sh                    # terminal 1
  3. python -m server.main                        # terminal 2
  4. python -m clients.chat_client                # terminal 3 — type to test
  5. python -m clients.mic_client                 # then talk to it

Set a real token before exposing this to the internet:
  export ASSISTANT_TOKEN="$(head -c 24 /dev/urandom | base64)"
EOF
