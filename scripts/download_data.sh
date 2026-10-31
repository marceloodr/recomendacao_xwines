#!/usr/bin/env bash
# Baixa a base X-Wines (versão Slim) do Kaggle para data/raw/xwines_slim/.
# Requer ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token).
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/kaggle datasets download -d rogerioxavier/x-wines-slim-version \
  -p data/raw/xwines_slim --unzip
ls -lh data/raw/xwines_slim
