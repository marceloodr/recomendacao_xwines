"""Divisão em treino/teste e simulação dos cenários de cold start."""

from contextlib import contextmanager

import numpy as np
import pandas as pd

RELEVANT_RATING = 4.5  # nota mínima para o vinho contar como "gostou" (ver EDA)


def temporal_user_split(ratings: pd.DataFrame, test_frac: float = 0.2,
                        min_test: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Para cada usuário, as notas mais recentes (test_frac) vão para o teste."""
    df = ratings.sort_values(["user_id", "date"], kind="stable")
    rank = df.groupby("user_id").cumcount()
    n = df.groupby("user_id")["user_id"].transform("size")
    n_test = np.maximum(min_test, np.floor(n * test_frac)).astype(int)
    is_test = (rank >= n - n_test) & (n > 1)  # quem tem 1 nota só fica no treino
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def make_splits(ratings: pd.DataFrame, items: pd.DataFrame, cold_item_frac: float = 0.05,
                cold_user_frac: float = 0.10, cold_user_min_ratings: int = 15,
                test_frac: float = 0.2, seed: int = 42) -> dict:
    """Separa os três cenários de avaliação.

    - itens novos: os vinhos que entraram por último no catálogo (data da 1ª avaliação mais
      recente; a base não tem data de cadastro) têm todas as suas notas removidas do treino;
    - usuários novos: uma amostra de usuários (cold_user_frac) sai inteira do treino;
    - usuários antigos: o restante é dividido temporalmente por usuário.
    """
    rng = np.random.default_rng(seed)
    first_seen = ratings.groupby("item_id")["date"].min()
    n_cold_items = max(1, int(round(cold_item_frac * len(first_seen))))
    cold_items = first_seen.nlargest(n_cold_items).index.to_numpy()

    is_cold_item = ratings["item_id"].isin(cold_items)
    cold_item_ratings, rest = ratings[is_cold_item], ratings[~is_cold_item]

    counts = rest["user_id"].value_counts()
    eligible = counts[counts >= cold_user_min_ratings].index.to_numpy()
    n_cold_users = int(round(cold_user_frac * rest["user_id"].nunique()))
    cold_users = rng.choice(eligible, size=min(n_cold_users, len(eligible)), replace=False)

    is_cold_user = rest["user_id"].isin(cold_users)
    cold_user_ratings = rest[is_cold_user].sort_values(["user_id", "date"], kind="stable")
    train, test = temporal_user_split(rest[~is_cold_user], test_frac)

    rated_items = items[items["item_id"].isin(first_seen.index)]["item_id"].to_numpy()
    return {
        "train": train,
        "test": test,
        "cold_users": cold_user_ratings.reset_index(drop=True),
        "cold_items_test": cold_item_ratings[cold_item_ratings["user_id"].isin(train["user_id"])]
        .reset_index(drop=True),
        "cold_item_ids": cold_items,
        "warm_item_ids": np.setdiff1d(rated_items, cold_items),
    }


def split_cold_users(cold_user_ratings: pd.DataFrame, n_given: int,
                     holdout_from: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Usuário novo informa suas n_given primeiras notas; avalia-se a partir da posição
    holdout_from, o que mantém o mesmo conjunto de teste para qualquer n_given."""
    df = cold_user_ratings.sort_values(["user_id", "date"], kind="stable")
    rank = df.groupby("user_id").cumcount()
    return df[rank < n_given].reset_index(drop=True), df[rank >= holdout_from].reset_index(drop=True)


def relevant_sets(test: pd.DataFrame, key: str = "user_id", value: str = "item_id",
                  threshold: float | None = None) -> pd.Series:
    """Conjunto de itens relevantes por usuário (ou de usuários por item, com key/value trocados)."""
    rel = test[test["rating"] >= (RELEVANT_RATING if threshold is None else threshold)]
    return rel.groupby(key)[value].agg(set)


@contextmanager
def relevance_threshold(threshold: float):
    """Troca temporariamente o limiar de relevância (análise de sensibilidade)."""
    global RELEVANT_RATING
    old, RELEVANT_RATING = RELEVANT_RATING, threshold
    try:
        yield
    finally:
        RELEVANT_RATING = old


def split_summary(splits: dict) -> pd.DataFrame:
    rows = []
    for name in ["train", "test", "cold_users", "cold_items_test"]:
        df = splits[name]
        rows.append({"conjunto": name, "notas": len(df), "usuários": df["user_id"].nunique(),
                     "vinhos": df["item_id"].nunique(),
                     "% relevantes": (df["rating"] >= RELEVANT_RATING).mean() * 100})
    return pd.DataFrame(rows).round(1)
