"""Ajuste de hiperparâmetros em validação (splits refeitos só dentro do treino)."""

import itertools

import pandas as pd

from src.evaluate import eval_cold_items, eval_warm
from src.features import WineFeaturizer
from src.model import ContentRecommender
from src.split import make_splits

WEIGHT_PRESETS = {
    "todos": {"grape": 1.0, "harm": 1.0, "style": 1.0, "origin": 0.5, "num": 0.3},
    "só uvas": {"grape": 1.0, "harm": 0.0, "style": 0.0, "origin": 0.0, "num": 0.0},
    "só harmonização": {"grape": 0.0, "harm": 1.0, "style": 0.0, "origin": 0.0, "num": 0.0},
    "só estilo": {"grape": 0.0, "harm": 0.0, "style": 1.0, "origin": 0.0, "num": 0.0},
    "só origem": {"grape": 0.0, "harm": 0.0, "style": 0.0, "origin": 1.0, "num": 0.0},
    "uva+harm+estilo": {"grape": 1.0, "harm": 1.0, "style": 1.0, "origin": 0.0, "num": 0.0},
    "origem forte": {"grape": 1.0, "harm": 0.5, "style": 1.0, "origin": 1.5, "num": 0.3},
}
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
BETAS = [0.0, 0.5, 0.8, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
METRIC = "ndcg@10"


def validation_splits(train: pd.DataFrame, items: pd.DataFrame, seed: int = 7) -> dict:
    return make_splits(train, items, seed=seed)


def tune_warm(featurizer: WineFeaturizer, items: pd.DataFrame, val: dict,
              presets=WEIGHT_PRESETS, ms=(10,), centered_opts=(True, False),
              verbose: bool = True) -> pd.DataFrame:
    """Grade em (pesos dos blocos, centralização, m) × (α, β) no cenário de usuário antigo."""
    warm = items[items["item_id"].isin(val["warm_item_ids"])]
    blocks = featurizer.transform_blocks(warm)
    rows = []
    for (preset, w), centered, m in itertools.product(presets.items(), centered_opts, ms):
        model = ContentRecommender(featurizer, m=m, centered=centered).fit(val["train"], warm, blocks, w)
        for alpha, beta in itertools.product(ALPHAS, BETAS):
            model.set_params(alpha=alpha, beta=beta)
            res = eval_warm(model, val["test"], baselines=False, extra=False).iloc[0]
            rows.append({"pesos": preset, "centralizado": centered, "m": m, "alpha": alpha,
                         "beta": beta, **res[["ndcg@10", "precision@10", "recall@10", "map@10", "hit@10"]]})
        if verbose:
            best = pd.DataFrame(rows).query("pesos == @preset and centralizado == @centered")[METRIC].max()
            print(f"{preset:18s} centralizado={centered!s:5s} m={m}: melhor {METRIC} = {best:.4f}")
    return pd.DataFrame(rows).sort_values(METRIC, ascending=False).reset_index(drop=True)


def tune_gamma(model: ContentRecommender, items: pd.DataFrame, val: dict) -> pd.DataFrame:
    """γ do cenário vinho novo → usuários (visão do item), escolhido pela AUC."""
    cold = items[items["item_id"].isin(val["cold_item_ids"])]
    rows = []
    for gamma in GAMMAS:
        model.set_params(gamma=gamma)
        res = eval_cold_items(model, cold, val["cold_items_test"], baselines=False)
        res = res[res["visão"] == "vinho novo → usuários"].iloc[0]
        rows.append({"gamma": gamma, **res[["auc", "ndcg@10", "recall@10", "hit@10"]]})
    return pd.DataFrame(rows)
