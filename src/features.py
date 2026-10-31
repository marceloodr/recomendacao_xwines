"""Vetorização dos vinhos: uvas, harmonização, estilo, origem e atributos numéricos."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

NUM_COLS = ["abv", "n_grapes", "n_vintages"]
TOKEN_BLOCKS = ["grape", "harm", "style", "origin"]
BLOCKS = TOKEN_BLOCKS + ["num"]
DEFAULT_WEIGHTS = {"grape": 1.0, "harm": 1.0, "style": 1.0, "origin": 0.5, "num": 0.3}
STYLE_COLS = ["type", "elaborate", "body", "acidity"]
ORIGIN_COLS = ["country", "region", "winery"]


def _identity(tokens):
    return tokens


def item_tokens(items: pd.DataFrame) -> dict[str, pd.Series]:
    """Tokens de cada bloco. Atributos categóricos viram 'coluna=valor' (ex.: 'type=Red')."""
    def cat(cols):
        return items[cols].astype(str).apply(lambda r: [f"{c}={v}" for c, v in r.items()], axis=1)
    return {
        "grape": items["grapes"].map(list),
        "harm": items["harmonize"].map(list),
        "style": cat(STYLE_COLS),
        "origin": cat(ORIGIN_COLS),
    }


class WineFeaturizer:
    """Transforma vinhos em vetores esparsos v_i = [w_grape·UVA, w_harm·HARM, w_style·ESTILO,
    w_origin·ORIGEM, w_num·NUM].

    Cada bloco é um multi-hot ponderado por IDF (atributos raros pesam mais que 'type=Red')
    e normalizado (norma L2 = 1) antes de receber o peso, para que o peso expresse a
    importância relativa do bloco e não o tamanho do vocabulário.
    """

    def __init__(self, weights: dict | None = None, min_df: int = 2):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.min_df = min_df

    def fit(self, items: pd.DataFrame) -> "WineFeaturizer":
        """Ajusta vocabulários e IDF no catálogo inteiro (só conteúdo, sem notas)."""
        tokens = item_tokens(items)
        self.vecs = {}
        for b in TOKEN_BLOCKS:
            self.vecs[b] = TfidfVectorizer(analyzer=_identity, min_df=self.min_df, binary=True,
                                           lowercase=False).fit(tokens[b])
        self.scaler = StandardScaler().fit(self._num(items))
        return self

    @staticmethod
    def _num(items):
        X = items[NUM_COLS].astype(float).copy()
        X[["n_grapes", "n_vintages"]] = np.log1p(X[["n_grapes", "n_vintages"]])
        return X

    def transform_blocks(self, items: pd.DataFrame) -> dict:
        tokens = item_tokens(items)
        blocks = {b: normalize(self.vecs[b].transform(tokens[b])) for b in TOKEN_BLOCKS}
        num = np.clip(self.scaler.transform(self._num(items)), -3, 3) / np.sqrt(len(NUM_COLS))
        blocks["num"] = sp.csr_matrix(num)
        return blocks

    def combine(self, blocks: dict, weights: dict | None = None) -> sp.csr_matrix:
        w = {**self.weights, **(weights or {})}
        X = sp.hstack([w[b] * blocks[b] for b in BLOCKS], format="csr")
        return normalize(X)

    def transform(self, items: pd.DataFrame, weights: dict | None = None) -> sp.csr_matrix:
        return self.combine(self.transform_blocks(items), weights)

    def transform_preferences(self, grapes=(), harmonize=(), types=(), body=(), acidity=(),
                              countries=()) -> sp.csr_matrix:
        """Vetor de um usuário novo a partir do que ele escolhe no onboarding."""
        style = ([f"type={t}" for t in types] + [f"body={b}" for b in body]
                 + [f"acidity={a}" for a in acidity])
        tokens = {"grape": list(grapes), "harm": list(harmonize), "style": style,
                  "origin": [f"country={c}" for c in countries]}
        blocks = {b: normalize(self.vecs[b].transform([tokens[b]])) for b in TOKEN_BLOCKS}
        blocks["num"] = sp.csr_matrix((1, len(NUM_COLS)))
        return self.combine(blocks)

    @property
    def feature_names(self) -> np.ndarray:
        names = [f"{b}:" + self.vecs[b].get_feature_names_out().astype(object) for b in TOKEN_BLOCKS]
        return np.concatenate(names + [np.array(["num:" + c for c in NUM_COLS], dtype=object)])

    def vocabulary(self, block: str, prefix: str | None = None) -> list[str]:
        """Valores disponíveis (ex.: vocabulary('style', 'type') -> ['Dessert', 'Red', ...])."""
        vocab = sorted(self.vecs[block].vocabulary_)
        if prefix:
            vocab = [v.split("=", 1)[1] for v in vocab if v.startswith(prefix + "=")]
        return vocab
