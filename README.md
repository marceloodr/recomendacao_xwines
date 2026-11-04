# Sistema de Recomendação de Vinhos — X-Wines

Trabalho T1 de Sistemas de Recomendação (PUCRS). Recomendação baseada em conteúdo + escores de produto. Ver [PLANO.md](PLANO.md).

## Ambiente

```bash
~/.pyenv/versions/3.12.13/bin/python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Dados

Base: [X-Wines](https://github.com/rogerioxavier/X-Wines) (de Azambuja, Morais & Filipe, 2023, *Big Data Cogn. Comput.* 7(1), 20; CC0).
Coloque os CSVs `XWines_<versão>_*_wines.csv` e `XWines_<versão>_*_ratings.csv` em qualquer subpasta de `data/raw/`.
O código usa a maior versão encontrada (Full > Slim > Test), ou a indicada em `--version`.

- **Slim** (recomendada): [Kaggle](https://www.kaggle.com/datasets/rogerioxavier/x-wines-slim-version) — `scripts/download_data.sh` baixa para `data/raw/xwines_slim/` (requer `~/.kaggle/kaggle.json`)
- **Test** (só para desenvolvimento): já em `data/raw/xwines_test/`, baixada do GitHub do autor

Tratamento (gera `data/processed/*.parquet` e `cleaning_log.csv`):

```bash
.venv/bin/python -m src.data --version Slim
```

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `src/data.py` | carga, limpeza, deduplicação de safras e filtro k-core |
| `src/features.py` | vetor do vinho: uvas, harmonização, estilo, origem, numéricos |
| `src/model.py` | recomendador conteúdo + escore de produto (os 3 cenários) |
| `src/split.py` | split temporal, usuários novos e itens novos |
| `src/metrics.py`, `src/evaluate.py` | métricas de ranking, cobertura/diversidade/novidade e baselines |
| `src/tune.py` | ajuste de hiperparâmetros em validação |
| `notebooks/01_eda.ipynb` | estatísticas descritivas e gráficos (figuras em `slides/figs/`) |
| `notebooks/02_modelo.ipynb` | split, vetorização e ajuste de hiperparâmetros (`data/processed/best_params.json`) |
| `notebooks/03_avaliacao.ipynb` | métricas nos 3 cenários vs baselines, exemplos e modelo final (`models/recommender.pkl`) |
| `app/app.py` | interface Streamlit com os 3 cenários (`streamlit run app/app.py`; requer `models/recommender.pkl`) |
