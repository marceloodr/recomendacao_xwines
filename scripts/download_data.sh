#!/usr/bin/env bash
# Baixa a base Food.com do Kaggle para data/raw/.
# Requer ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token).
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions \
  -p data/raw --unzip
ls -lh data/raw
