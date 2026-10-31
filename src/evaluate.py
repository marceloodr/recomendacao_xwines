"""Avaliação dos três cenários e baselines de comparação."""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.metrics import beyond_accuracy, bootstrap_gain, mean_auc, ndcg_per_key, ranking_metrics
from src.model import ContentRecommender
from src.split import make_splits, relevance_threshold, relevant_sets, split_cold_users

KS = (5, 10, 20)


# ------------------------------------------------------------------- baselines
def popularity_recs(model: ContentRecommender, exclude: sp.csr_matrix | None, n_rows: int,
                    k: int, by: str = "count") -> np.ndarray:
    """Mesma lista para todos: mais avaliadas (count) ou maior média bayesiana (wr)."""
    score = model.n_ratings if by == "count" else model.wr
    order = np.argsort(-score)
    out = np.empty((n_rows, k), dtype=model.item_ids.dtype)
    for r in range(n_rows):
        seen = set(exclude[r].indices) if exclude is not None else set()
        out[r] = model.item_ids[[i for i in order[:k + len(seen)] if i not in seen][:k]]
    return out


def random_recs(item_ids: np.ndarray, exclude: sp.csr_matrix | None, n_rows: int, k: int,
                seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n_rows, k), dtype=item_ids.dtype)
    for r in range(n_rows):
        seen = set(exclude[r].indices) if exclude is not None else set()
        cand = rng.choice(len(item_ids), size=min(k + len(seen), len(item_ids)), replace=False)
        out[r] = item_ids[[i for i in cand if i not in seen][:k]]
    return out


# ------------------------------------------------------------------- cenários
def _row(name, recs, keys, relevant, model, extra=True):
    res = {"modelo": name, **ranking_metrics(recs, keys, relevant, KS)}
    if extra:
        pop = pd.Series(model.n_ratings, index=model.item_ids)
        res.update(beyond_accuracy(recs, model.X, model.item_pos, pop, len(model.user_ids),
                                   len(model.item_ids)))
    return res


def eval_warm(model: ContentRecommender, test: pd.DataFrame, baselines: bool = True,
              extra: bool = True) -> pd.DataFrame:
    """Cenário 1: usuários antigos, notas futuras (split temporal)."""
    relevant = relevant_sets(test)
    users = np.array([u for u in relevant.index if u in model.user_pos.index])
    rows = model.user_pos[users].to_numpy()
    P, seen = model.P[rows], model.seen[rows]
    k = max(KS)
    out = [_row("conteúdo + escore", model.recommend_batch(P, k, seen), users, relevant, model, extra)]
    if baselines:
        out.append(_row("popularidade", popularity_recs(model, seen, len(users), k), users, relevant, model, extra))
        out.append(_row("média bayesiana", popularity_recs(model, seen, len(users), k, "wr"), users, relevant, model, extra))
        out.append(_row("aleatório", random_recs(model.item_ids, seen, len(users), k), users, relevant, model, extra))
    return pd.DataFrame(out).set_index("modelo")


def eval_cold_users(model: ContentRecommender, cold_users: pd.DataFrame,
                    n_given_list=(1, 3, 5, 10), baselines: bool = True) -> pd.DataFrame:
    """Cenário 2: usuário novo informa n_given notas; avaliado nas notas a partir da 11ª."""
    k = max(KS)
    out = []
    for n_given in n_given_list:
        given, test = split_cold_users(cold_users, n_given)
        relevant = relevant_sets(test)
        user_ids, P, seen = model.profiles_from_ratings(given)
        keep = np.isin(user_ids, relevant.index)
        recs = model.recommend_batch(P[keep], k, seen[keep])
        out.append({"n_notas_informadas": n_given,
                    **_row("conteúdo + escore", recs, user_ids[keep], relevant, model, extra=False)})
    if baselines:
        _, test = split_cold_users(cold_users, 0)
        relevant = relevant_sets(test)
        users = relevant.index.to_numpy()
        for name, recs in [("popularidade", popularity_recs(model, None, len(users), k)),
                           ("média bayesiana", popularity_recs(model, None, len(users), k, "wr")),
                           ("aleatório", random_recs(model.item_ids, None, len(users), k))]:
            out.append({"n_notas_informadas": 0, **_row(name, recs, users, relevant, model, extra=False)})
    return pd.DataFrame(out)


def eval_cold_items(model: ContentRecommender, cold_items: pd.DataFrame,
                    cold_items_test: pd.DataFrame, baselines: bool = True) -> pd.DataFrame:
    """Cenário 3: vinhos sem nenhuma nota no treino.

    - visão do usuário: para cada usuário antigo, ranquear só os vinhos novos;
    - visão do item: para cada vinho novo, ranquear os usuários antigos (a quem mostrar).
    """
    cold_items = cold_items.reset_index(drop=True)
    X_cold = model.featurizer.transform(cold_items)
    ids = cold_items["item_id"].to_numpy()
    prior = np.full(len(ids), model.product_score_for(0, model.C))  # sem notas: todas iguais
    k = max(KS)
    out = []

    rel_u = relevant_sets(cold_items_test)
    users = np.array([u for u in rel_u.index if u in model.user_pos.index])
    P = model.P[model.user_pos[users].to_numpy()]
    recs = model.recommend_batch(P, k, X=X_cold, prior=prior, item_ids=ids)
    out.append({"visão": "usuário → vinhos novos", **_row("conteúdo", recs, users, rel_u, model, False)})
    if baselines:
        out.append({"visão": "usuário → vinhos novos",
                    **_row("aleatório", random_recs(ids, None, len(users), k), users, rel_u, model, False)})

    rel_i = relevant_sets(cold_items_test, key="item_id", value="user_id")
    items = rel_i.index.to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)[items].to_numpy()
    # Cada vinho novo tem poucos fãs entre milhares de usuários: o top-10 quase nunca acerta,
    # então a métrica principal desta visão é a AUC (o fã fica acima do não fã?).
    aff = model.user_affinity((X_cold[pos] @ model.P.T).toarray())
    top = np.argsort(-aff, axis=1)[:, :k]
    out.append({"visão": "vinho novo → usuários", "auc": mean_auc(aff, items, model.user_ids, rel_i),
                **_row("conteúdo", model.user_ids[top], items, rel_i, model, False)})
    if baselines:
        # Baseline: mostrar o vinho aos usuários mais ativos.
        activity = np.tile(model.user_activity, (len(items), 1))
        active = model.user_ids[np.argsort(-model.user_activity, kind="stable")[:k]]
        out.append({"visão": "vinho novo → usuários", "auc": mean_auc(activity, items, model.user_ids, rel_i),
                    **_row("usuários mais ativos", np.tile(active, (len(items), 1)), items, rel_i, model, False)})
        rng = np.random.default_rng(0)  # média de 20 sorteios: com poucos vinhos, um sorteio só é ruidoso
        rnd_auc = np.mean([mean_auc(rng.random((len(items), len(model.user_ids))), items, model.user_ids, rel_i)
                           for _ in range(20)])
        out.append({"visão": "vinho novo → usuários", "auc": rnd_auc,
                    **_row("aleatório", random_recs(model.user_ids, None, len(items), k), items, rel_i, model, False)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------- robustez
def _best(df: pd.DataFrame, ours: str, col: str = "ndcg@10") -> tuple[float, float]:
    others = df[df["modelo"] != ours]
    return df.loc[df["modelo"] == ours, col].iloc[0], others[col].max()


def robustness(featurizer, params: dict, ratings: pd.DataFrame, items: pd.DataFrame,
               seeds=(42, 1, 2, 3, 4), thresholds=(4.0, 4.5)) -> pd.DataFrame:
    """Reavalia os três cenários variando a semente (sorteio dos usuários novos) e o limiar
    de relevância. Cada linha traz o modelo e o melhor baseline em NDCG@10 (AUC no vinho → usuários)."""
    rows = []
    for thr in thresholds:
        with relevance_threshold(thr):
            for seed in seeds:
                s = make_splits(ratings, items, seed=seed)
                warm_items = items[items["item_id"].isin(s["warm_item_ids"])]
                m = ContentRecommender(featurizer, **{k: v for k, v in params.items() if k != "weights"})
                m.fit(s["train"], warm_items)
                w = eval_warm(m, s["test"], extra=False).reset_index()
                cu = eval_cold_users(m, s["cold_users"], (5,))
                ci = eval_cold_items(m, items[items["item_id"].isin(s["cold_item_ids"])], s["cold_items_test"])
                cu_ = ci[ci["visão"] == "usuário → vinhos novos"]
                iu_ = ci[ci["visão"] == "vinho novo → usuários"]
                for cen, (a, b) in {"usuário antigo": _best(w, "conteúdo + escore"),
                                    "usuário novo (5 notas)": _best(cu, "conteúdo + escore"),
                                    "vinho novo: vitrine do usuário": _best(cu_, "conteúdo"),
                                    "vinho novo → usuários (AUC)": _best(iu_, "conteúdo", "auc")}.items():
                    rows.append({"limiar": thr, "semente": seed, "cenário": cen, "modelo": a, "melhor baseline": b})
    return pd.DataFrame(rows)


def confidence_intervals(model: ContentRecommender, splits: dict, cold_items: pd.DataFrame,
                         k: int = 10) -> pd.DataFrame:
    """IC 95% por bootstrap pareado (por usuário) do NDCG@k do modelo contra o melhor baseline."""
    out = []
    # usuário antigo
    rel = relevant_sets(splits["test"])
    users = np.array([u for u in rel.index if u in model.user_pos.index])
    rows = model.user_pos[users].to_numpy()
    P, seen = model.P[rows], model.seen[rows]
    ours = ndcg_per_key(model.recommend_batch(P, k, seen), users, rel, k)
    bases = {"popularidade": popularity_recs(model, seen, len(users), k),
             "média bayesiana": popularity_recs(model, seen, len(users), k, "wr")}
    name, base = max(((n, ndcg_per_key(r, users, rel, k)) for n, r in bases.items()), key=lambda t: t[1].mean())
    out.append({"cenário": "usuário antigo", "baseline": name, **bootstrap_gain(ours, base)})
    # usuário novo (5 notas)
    given, test = split_cold_users(splits["cold_users"], 5)
    rel = relevant_sets(test)
    uids, Pc, seen_c = model.profiles_from_ratings(given)
    keep = np.isin(uids, rel.index)
    ours = ndcg_per_key(model.recommend_batch(Pc[keep], k, seen_c[keep]), uids[keep], rel, k)
    bases = {"popularidade": popularity_recs(model, None, keep.sum(), k),
             "média bayesiana": popularity_recs(model, None, keep.sum(), k, "wr")}
    name, base = max(((n, ndcg_per_key(r, uids[keep], rel, k)) for n, r in bases.items()), key=lambda t: t[1].mean())
    out.append({"cenário": "usuário novo (5 notas)", "baseline": name, **bootstrap_gain(ours, base)})
    # vinho novo: vitrine do usuário (baseline aleatório = média de 50 sorteios)
    cold_items = cold_items.reset_index(drop=True)
    X_cold, ids = model.featurizer.transform(cold_items), cold_items["item_id"].to_numpy()
    rel = relevant_sets(splits["cold_items_test"])
    users = np.array([u for u in rel.index if u in model.user_pos.index])
    P = model.P[model.user_pos[users].to_numpy()]
    ours = ndcg_per_key(model.recommend_batch(P, k, X=X_cold, prior=np.zeros(len(ids)), item_ids=ids), users, rel, k)
    base = np.mean([ndcg_per_key(random_recs(ids, None, len(users), k, seed=i), users, rel, k)
                    for i in range(50)], axis=0)
    out.append({"cenário": "vinho novo: vitrine do usuário", "baseline": "aleatório", **bootstrap_gain(ours, base)})
    return pd.DataFrame(out)
