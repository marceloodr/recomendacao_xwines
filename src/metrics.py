"""Métricas de ranking (Precision, Recall, NDCG, MAP, Hit Rate) e de além da acurácia."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score


def ranking_metrics(recs: np.ndarray, keys, relevant: pd.Series, ks=(5, 10, 20)) -> dict:
    """recs[j] é a lista ordenada recomendada para keys[j]; relevant mapeia key -> set.
    Só entram usuários (ou itens) com ao menos um relevante no teste."""
    out = {f"{m}@{k}": [] for k in ks for m in ["precision", "recall", "ndcg", "map", "hit"]}
    discounts = 1 / np.log2(np.arange(2, max(ks) + 2))
    for key, rec in zip(keys, recs):
        rel = relevant.get(key)
        if not rel:
            continue
        hits = np.fromiter((r in rel for r in rec[:max(ks)]), dtype=float, count=min(len(rec), max(ks)))
        for k in ks:
            h = hits[:k]
            n_hit = h.sum()
            out[f"precision@{k}"].append(n_hit / k)
            out[f"recall@{k}"].append(n_hit / len(rel))
            ideal = discounts[:min(len(rel), k)].sum()
            out[f"ndcg@{k}"].append((h * discounts[:len(h)]).sum() / ideal)
            prec_at_i = np.cumsum(h) / np.arange(1, len(h) + 1)
            out[f"map@{k}"].append((prec_at_i * h).sum() / min(len(rel), k))
            out[f"hit@{k}"].append(float(n_hit > 0))
    res = {m: float(np.mean(v)) for m, v in out.items()}
    res["n_avaliados"] = len(out[f"hit@{ks[0]}"])
    return res


def catalog_coverage(recs: np.ndarray, n_items: int) -> float:
    return len(np.unique(recs)) / n_items


def intra_list_diversity(recs: np.ndarray, X: sp.csr_matrix, item_pos: pd.Series,
                         sample: int = 2000, seed: int = 0) -> float:
    """1 − similaridade de cosseno média entre os pares de itens de cada lista."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(recs), size=min(sample, len(recs)), replace=False)
    vals = []
    for r in rows:
        V = X[item_pos[recs[r]].to_numpy()]
        S = (V @ V.T).toarray()
        k = len(S)
        vals.append(1 - (S.sum() - np.trace(S)) / (k * (k - 1)))
    return float(np.mean(vals))


def novelty(recs: np.ndarray, popularity: pd.Series, n_users: int) -> float:
    """Média de −log2(p_i), com p_i a fração de usuários que avaliou o item no treino."""
    p = popularity.reindex(recs.ravel()).fillna(0).to_numpy() + 1
    return float(np.mean(-np.log2(p / (n_users + 1))))


def beyond_accuracy(recs, X, item_pos, popularity, n_users, n_items) -> dict:
    return {
        "cobertura": catalog_coverage(recs, n_items),
        "diversidade": intra_list_diversity(recs, X, item_pos),
        "novidade": novelty(recs, popularity, n_users),
    }


def ndcg_per_key(recs: np.ndarray, keys, relevant: pd.Series, k: int = 10) -> np.ndarray:
    """NDCG@k de cada usuário (ou item) com ao menos um relevante — base do bootstrap."""
    disc = 1 / np.log2(np.arange(2, k + 2))
    out = []
    for key, rec in zip(keys, recs):
        rel = relevant.get(key)
        if not rel:
            continue
        h = np.fromiter((r in rel for r in rec[:k]), dtype=float)
        out.append((h * disc[:len(h)]).sum() / disc[:min(len(rel), k)].sum())
    return np.array(out)


def mean_auc(scores: np.ndarray, keys, candidates: np.ndarray, relevant: pd.Series) -> float:
    """AUC média: para cada linha (ex.: um vinho novo), a probabilidade de um candidato
    relevante (um fã) ficar acima de um não relevante na ordenação por scores."""
    aucs = []
    for row, key in zip(scores, keys):
        y = np.isin(candidates, list(relevant.get(key, ())))
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, row))
    return float(np.mean(aucs)) if aucs else float("nan")


def bootstrap_gain(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """IC 95% (bootstrap pareado) da diferença e do ganho relativo entre médias de a e b."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    ma, mb = a[idx].mean(axis=1), b[idx].mean(axis=1)
    diff = np.percentile(ma - mb, [2.5, 97.5])
    gain = np.percentile(ma / np.where(mb > 0, mb, np.nan) - 1, [2.5, 97.5])
    return {"modelo": a.mean(), "baseline": b.mean(), "dif_ic_inf": diff[0], "dif_ic_sup": diff[1],
            "ganho_ic_inf": gain[0], "ganho_ic_sup": gain[1], "n": len(a)}
