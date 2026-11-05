"""Recomendador baseado em conteúdo + escore de produto (média bayesiana e popularidade).

score(u, i) = α · cos(p_u, v_i) + (1 − α) · E_i

E_i = (1 − β) · WR_i normalizado + β · Pop_i  (escore de produto)

- v_i: vetor de atributos do vinho (src/features.py)
- p_u: perfil do usuário = Σ_i w_ui · v_i, com w_ui = r_ui (notas brutas) ou r_ui − b_u
  (centralizado; b_u é a média do usuário "encolhida" em direção à média global)
- WR_i = (n_i · R_i + m · C) / (n_i + m), a média bayesiana estilo IMDb
- Pop_i = log(1 + n_i) / log(1 + max n), a popularidade no treino

Para o item novo → usuários, a afinidade é combinada com a atividade do usuário:
aff(u, i) = γ · cos(p_u, v_i) + (1 − γ) · Ativ_u, com Ativ_u = log(1 + n_u) / log(1 + max n).

No onboarding do usuário novo por preferências declaradas (tipo, uvas, pratos), usa-se
α_onboarding no lugar de α: o α validado vale para perfis montados a partir de notas, e com
ele as escolhas explícitas da pessoa ficariam abafadas pela popularidade.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.data import fill_attributes
from src.features import WineFeaturizer

NEUTRAL_RATING = 3.0  # meio da escala 1–5: nota informada no onboarding acima disso conta a favor
DISPLAY_COLS = ["item_id", "name", "winery", "type", "country", "region", "grapes", "abv"]
_LABELS = {"grape": "uva", "harm": "combina com", "type": "tipo", "body": "corpo",
           "acidity": "acidez", "elaborate": "elaboração", "country": "país",
           "region": "região", "winery": "vinícola"}


def pretty_feature(name: str) -> str:
    """'grape:Merlot' -> 'uva Merlot'; 'style:body=Full-bodied' -> 'corpo Full-bodied'."""
    block, value = name.split(":", 1)
    if "=" in value:
        col, value = value.split("=", 1)
        return f"{_LABELS.get(col, col)} {value}"
    return f"{_LABELS.get(block, block)} {value}"


class ContentRecommender:
    def __init__(self, featurizer: WineFeaturizer, alpha: float = 0.5, beta: float = 0.5,
                 gamma: float = 0.5, m: float = 10, shrink: float = 5.0, centered: bool = True,
                 alpha_onboarding: float = 0.6):
        self.featurizer = featurizer
        self.alpha = alpha
        self.alpha_onboarding = alpha_onboarding
        self.beta = beta
        self.gamma = gamma
        self.m = m
        self.shrink = shrink
        self.centered = centered

    # ---------------------------------------------------------------- ajuste
    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame, blocks: dict | None = None,
            weights: dict | None = None) -> "ContentRecommender":
        """ratings: notas de treino; items: catálogo de candidatos (com atributos)."""
        self._set_items(items.reset_index(drop=True))
        blocks = blocks if blocks is not None else self.featurizer.transform_blocks(self.items)
        self.X = self.featurizer.combine(blocks, weights)

        ratings = ratings[ratings["item_id"].isin(self.item_pos.index)]
        self.C = ratings["rating"].mean()
        stats = ratings.groupby("item_id")["rating"].agg(["count", "mean"])
        stats = stats.reindex(self.item_ids).fillna({"count": 0, "mean": self.C})
        self.n_ratings = stats["count"].to_numpy()
        self.wr = (self.n_ratings * stats["mean"].to_numpy() + self.m * self.C) / (self.n_ratings + self.m)
        self.wr_min, self.wr_max = self.wr.min(), self.wr.max()
        self.max_log_count = np.log1p(self.n_ratings.max())
        self._update_product_score()

        self.user_ids, self.P, self.seen = self._profiles(ratings)
        self.user_pos = pd.Series(np.arange(len(self.user_ids)), index=self.user_ids)
        n_user = np.asarray(self.seen.sum(axis=1)).ravel()
        self.user_activity = np.log1p(n_user) / np.log1p(n_user.max())
        return self

    def _set_items(self, items: pd.DataFrame):
        self.items = items
        self.item_ids = items["item_id"].to_numpy()
        self.item_pos = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids)
        self.item_grapes = items["grapes"].map(set)

    def _norm_wr(self, wr):
        return np.clip((wr - self.wr_min) / max(self.wr_max - self.wr_min, 1e-9), 0, 1)

    def product_score_for(self, n_ratings, wr):
        pop = np.log1p(n_ratings) / self.max_log_count
        return (1 - self.beta) * self._norm_wr(wr) + self.beta * pop

    def _update_product_score(self):
        self.product_score = self.product_score_for(self.n_ratings, self.wr)

    def set_params(self, **params) -> "ContentRecommender":
        """Troca α, β, γ sem reajustar (β recalcula o escore de produto)."""
        for k, v in params.items():
            setattr(self, k, v)
        self._update_product_score()
        return self

    def _profiles(self, ratings: pd.DataFrame):
        """Retorna (ids, perfis normalizados, matriz de itens já vistos) para os usuários de ratings."""
        ratings = ratings[ratings["item_id"].isin(self.item_pos.index)]
        user_ids = np.sort(ratings["user_id"].unique())
        rows = pd.Series(np.arange(len(user_ids)), index=user_ids)[ratings["user_id"]].to_numpy()
        cols = self.item_pos[ratings["item_id"]].to_numpy()
        r = ratings["rating"].to_numpy(dtype=float)
        if self.centered:
            g = ratings.groupby("user_id")["rating"].agg(["sum", "count"])
            b = (g["sum"] + self.shrink * self.C) / (g["count"] + self.shrink)
            w = r - b.reindex(ratings["user_id"]).to_numpy()
        else:
            w = r
        shape = (len(user_ids), len(self.item_ids))
        W = sp.csr_matrix((w, (rows, cols)), shape=shape)
        seen = sp.csr_matrix((np.ones_like(w), (rows, cols)), shape=shape)
        return user_ids, normalize(W @ self.X), seen

    def profiles_from_ratings(self, ratings: pd.DataFrame):
        """Perfis de usuários que não estavam no treino (cenário de usuário novo)."""
        return self._profiles(ratings)

    # ---------------------------------------------------------------- escores
    def scores(self, P: sp.csr_matrix, X: sp.csr_matrix | None = None,
               prior: np.ndarray | None = None, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        X = self.X if X is None else X
        prior = self.product_score if prior is None else prior
        alpha = self.alpha if alpha is None else alpha
        sim = (P @ X.T).toarray()
        return alpha * sim + (1 - alpha) * prior[None, :], sim

    def recommend_batch(self, P: sp.csr_matrix, k: int = 10, exclude: sp.csr_matrix | None = None,
                        X=None, prior=None, item_ids=None, batch: int = 1024) -> np.ndarray:
        """Top-k (ids de item) para cada linha de P, em lotes para caber na memória."""
        item_ids = self.item_ids if item_ids is None else item_ids
        k = min(k, len(item_ids) - 1)
        out = np.empty((P.shape[0], k), dtype=item_ids.dtype)
        for start in range(0, P.shape[0], batch):
            sl = slice(start, start + batch)
            s, _ = self.scores(P[sl], X, prior)
            if exclude is not None:
                ex = exclude[sl].tocoo()
                s[ex.row, ex.col] = -np.inf
            top = np.argpartition(-s, k, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
            out[sl] = item_ids[np.take_along_axis(top, order, axis=1)]
        return out

    def explain(self, profile: sp.csr_matrix, item_idx: int, n: int = 4) -> str:
        """Atributos que mais contribuíram para a similaridade (perfil ∘ item)."""
        contrib = profile.multiply(self.X[item_idx]).tocoo()
        order = np.argsort(-contrib.data)
        names = [self.featurizer.feature_names[contrib.col[j]] for j in order
                 if contrib.data[j] > 0 and not self.featurizer.feature_names[contrib.col[j]].startswith("num:")]
        return ", ".join(pretty_feature(x) for x in names[:n]) or "vinho bem avaliado"

    def _result(self, profile, score, sim, idx) -> pd.DataFrame:
        df = self.items.iloc[idx][[c for c in DISPLAY_COLS if c in self.items]].copy()
        df["score"] = score[idx]
        df["similaridade"] = sim[idx]
        df["nota_bayesiana"] = self.wr[idx]
        df["n_notas"] = self.n_ratings[idx].astype(int)
        df["porque"] = [self.explain(profile, i) for i in idx]
        return df.reset_index(drop=True)

    def _filter_mask(self, types=None, countries=None, avoid_grapes=None, max_abv=None) -> np.ndarray:
        """Restrições duras do usuário (tipo, país, uvas que não quer, teor alcoólico máximo)."""
        ok = np.ones(len(self.item_ids), dtype=bool)
        if types:
            ok &= self.items["type"].isin(types).to_numpy()
        if countries:
            ok &= self.items["country"].isin(countries).to_numpy()
        if avoid_grapes:
            avoid = set(avoid_grapes)
            ok &= ~self.item_grapes.map(lambda s: bool(s & avoid)).to_numpy()
        if max_abv:
            ok &= self.items["abv"].to_numpy() <= max_abv
        return ok

    def _top(self, profile, exclude_idx=(), k=10, alpha=None, **filters) -> pd.DataFrame:
        score, sim = self.scores(profile, alpha=alpha)
        score, sim = score[0], sim[0]
        score[~self._filter_mask(**filters)] = -np.inf
        score[list(exclude_idx)] = -np.inf
        idx = np.argsort(-score)[:k]
        idx = idx[np.isfinite(score[idx])]
        return self._result(profile, score, sim, idx)

    # ---------------------------------------------------------------- cenários
    def recommend_existing_user(self, user_id, k: int = 10, **filters) -> pd.DataFrame:
        """Cenário 1: lista para um usuário antigo (exclui o que ele já avaliou)."""
        u = self.user_pos[user_id]
        return self._top(self.P[u], self.seen[u].indices, k, **filters)

    def user_history(self, user_id, ratings: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        h = ratings[ratings["user_id"] == user_id].merge(self.items[DISPLAY_COLS], on="item_id")
        return h.sort_values(["rating", "date"], ascending=False).head(n)[
            ["item_id", "name", "type", "country", "grapes", "rating", "date"]]

    def recommend_new_user(self, preferences: dict | None = None, rated: dict | None = None,
                           k: int = 10, **filters) -> pd.DataFrame:
        """Cenário 2: usuário novo. preferences vai para WineFeaturizer.transform_preferences
        (uvas, harmonização, tipos, corpo, acidez, países); rated = {item_id: nota}, com peso
        nota − NEUTRAL_RATING. Sem nenhuma informação, cai no ranking pelo escore de produto."""
        parts = []
        if preferences and any(preferences.values()):
            parts.append(self.featurizer.transform_preferences(**preferences))
        rated = {i: v for i, v in (rated or {}).items() if i in self.item_pos.index}
        if rated:
            # Notas declaradas são centralizadas no meio da escala: acima de 3 aproxima,
            # abaixo afasta. Com notas brutas (centered=False) e um vinho só, a normalização
            # apagaria a nota, e um vinho com nota 1 traria vinhos parecidos com ele.
            w = np.array(list(rated.values()), dtype=float) - NEUTRAL_RATING
            parts.append(normalize(sp.csr_matrix(w) @ self.X[self.item_pos[list(rated)].to_numpy()]))
        if not parts:
            profile = sp.csr_matrix((1, self.X.shape[1]))
            score = self.product_score.copy()
            score[~self._filter_mask(**filters)] = -np.inf
            idx = np.argsort(-score)[:k]
            return self._result(profile, score, np.zeros_like(score), idx[np.isfinite(score[idx])])
        profile = normalize(sum(parts))
        alpha = self.alpha_onboarding if preferences and any(preferences.values()) else None
        return self._top(profile, self.item_pos[list(rated)].to_numpy() if rated else (), k, alpha, **filters)

    def add_items(self, new_items: pd.DataFrame) -> np.ndarray:
        """Cenário 3: adiciona vinhos ao catálogo. Sem notas, a média bayesiana é a global (C).
        Atributos ausentes são preenchidos; vinhos cujo item_id já está no catálogo são
        ignorados (o app pode cadastrar o mesmo vinho de novo ao recarregar)."""
        ids = new_items["item_id"].to_numpy()
        new_items = fill_attributes(new_items.drop_duplicates("item_id"), reference=self.items)
        new_items = new_items[~new_items["item_id"].isin(self.item_pos.index)].reset_index(drop=True)
        if new_items.empty:
            return ids
        X_new = self.featurizer.transform(new_items)
        self.X = sp.vstack([self.X, X_new], format="csr")
        self._set_items(pd.concat([self.items, new_items], ignore_index=True))
        self.n_ratings = np.concatenate([self.n_ratings, np.zeros(len(new_items))])
        self.wr = np.concatenate([self.wr, np.full(len(new_items), self.C)])
        self._update_product_score()
        self.seen = sp.hstack([self.seen, sp.csr_matrix((self.seen.shape[0], len(new_items)))], format="csr")
        return ids

    def user_affinity(self, sim: np.ndarray) -> np.ndarray:
        return self.gamma * sim + (1 - self.gamma) * self.user_activity

    def recommend_new_item(self, item_id, k: int = 10) -> pd.DataFrame:
        """Cenário 3: usuários antigos com maior afinidade com um vinho (novo ou não)."""
        i = self.item_pos[item_id]
        sim = (self.P @ self.X[i].T).toarray().ravel()
        aff = self.user_affinity(sim)
        top = np.argsort(-aff)[:k]
        return pd.DataFrame({
            "user_id": self.user_ids[top], "afinidade": aff[top], "similaridade": sim[top],
            "porque": [self.explain(self.P[u], i) for u in top],
        })

    def recommend_new_items_for_user(self, user_id, k: int = 5) -> pd.DataFrame:
        """Cenário 3, visão do usuário: "vitrine de novidades" — só os vinhos sem nenhuma nota,
        ordenados pela similaridade com o perfil (sem popularidade, todos empatariam no escore
        de produto e ficariam no fim da lista completa)."""
        u = self.user_pos[user_id]
        new = np.flatnonzero(self.n_ratings == 0)
        sim = (self.P[u] @ self.X[new].T).toarray().ravel()
        order = new[np.argsort(-sim)[:k]]
        full_sim = np.zeros(len(self.item_ids))
        full_sim[new] = sim
        return self._result(self.P[u], full_sim, full_sim, order)
