#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${HOME}/.streamlit"

PORT="${PORT:-8501}"

cat > "${HOME}/.streamlit/config.toml" <<EOF
[server]
port = ${PORT}
headless = true
enableCORS = false

[browser]
gatherUsageStats = false
EOF

if [ ! -f models/titles.pkl ]; then
  python scripts/train_model.py
fi
