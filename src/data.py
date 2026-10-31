"""Carga e limpeza da base X-Wines (arquivos XWines_<versão>_*_wines.csv e *_ratings.csv)."""

import ast
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

VERSIONS = ["Full", "Slim", "Test"]  # ordem de preferência quando há mais de uma baixada
LIST_COLS = ["grapes", "harmonize", "vintages"]
VALID_RATINGS = np.arange(1, 5.5, 0.5)
ITEM_COLUMNS = {
    "WineID": "item_id", "WineName": "name", "Type": "type", "Elaborate": "elaborate",
    "Grapes": "grapes", "Harmonize": "harmonize", "ABV": "abv", "Body": "body",
    "Acidity": "acidity", "Code": "country_code", "Country": "country", "RegionID": "region_id",
    "RegionName": "region", "WineryID": "winery_id", "WineryName": "winery",
    "Website": "website", "Vintages": "vintages",
}
RATING_COLUMNS = {"RatingID": "rating_id", "UserID": "user_id", "WineID": "item_id",
                  "Vintage": "vintage", "Rating": "rating", "Date": "date"}


@dataclass
class CleaningLog:
    """Registra quantas linhas restam após cada passo (usado no slide de tratamento)."""
    steps: list = field(default_factory=list)

    def add(self, table: str, step: str, n_rows: int, **extra):
        self.steps.append({"tabela": table, "passo": step, "linhas": n_rows, **extra})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def _parse_list(value):
    if isinstance(value, str) and value.startswith("["):
        try:
            return [v for v in ast.literal_eval(value) if v not in ("", None)]
        except (ValueError, SyntaxError):
            return []
    return []


def find_files(raw_dir: Path = RAW_DIR, version: str | None = None) -> tuple[Path, Path]:
    """Localiza o par wines/ratings em raw_dir (e subpastas). Sem version, usa a maior baixada."""
    for v in ([version] if version else VERSIONS):
        wines = sorted(raw_dir.rglob(f"XWines_{v}_*wines*.csv"))
        ratings = sorted(raw_dir.rglob(f"XWines_{v}_*ratings*.csv"))
        if wines and ratings:
            return wines[0], ratings[0]
    raise FileNotFoundError(f"Arquivos XWines_* não encontrados em {raw_dir}")


def load_raw(raw_dir: Path = RAW_DIR, version: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê os CSVs originais e padroniza nomes de colunas (item_id, user_id, rating, date...)."""
    wines_path, ratings_path = find_files(raw_dir, version)
    items = pd.read_csv(wines_path).rename(columns=ITEM_COLUMNS)
    for col in LIST_COLS:
        items[col] = items[col].map(_parse_list)
    ratings = pd.read_csv(ratings_path, dtype={"Vintage": str}, parse_dates=["Date"]).rename(columns=RATING_COLUMNS)
    return items, ratings


def clean_items(items: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Remove duplicados, vinhos sem uva/tipo e teor alcoólico impossível."""
    df = items.copy()
    log.add("vinhos", "original", len(df))

    df = df.drop_duplicates(subset="item_id")
    log.add("vinhos", "remove WineID duplicado", len(df))

    df = df.dropna(subset=["name", "type"])
    df = df[df["grapes"].map(len) > 0]
    log.add("vinhos", "remove sem nome/tipo/uva", len(df))

    for col in ["elaborate", "body", "acidity", "country", "region", "winery"]:
        df[col] = df[col].fillna("Desconhecido").astype(str).str.strip()
    df["abv"] = pd.to_numeric(df["abv"], errors="coerce")
    df = df[df["abv"].between(0.5, 25) | df["abv"].isna()]
    df["abv"] = df["abv"].fillna(df.groupby("type")["abv"].transform("median"))
    log.add("vinhos", "remove destilados e ABV inválido (fora de 0,5–25%, ex.: grappas)", len(df))

    # Safras misturam anos e 'N.V.' (sem safra): guardadas como texto.
    df["vintages"] = df["vintages"].map(lambda l: [str(v) for v in l])
    df["n_grapes"] = df["grapes"].map(len)
    df["n_vintages"] = df["vintages"].map(len)
    return df.reset_index(drop=True)


def clean_ratings(ratings: pd.DataFrame, valid_items: pd.Series, log: CleaningLog) -> pd.DataFrame:
    df = ratings.copy()
    log.add("notas", "original", len(df))

    df = df.dropna(subset=["user_id", "item_id", "rating", "date"])
    df = df[np.isin(df["rating"], VALID_RATINGS)]
    log.add("notas", "remove ausentes/fora da escala 1–5", len(df))

    # Na versão Full o mesmo usuário pode avaliar o mesmo vinho em safras diferentes:
    # o sistema recomenda o rótulo, então fica a avaliação mais recente.
    n_before = len(df)
    df = df.sort_values("date").drop_duplicates(subset=["user_id", "item_id"], keep="last")
    log.add("notas", "remove par usuário–vinho repetido (outras safras)", len(df),
            repetidas=n_before - len(df))

    df = df[df["item_id"].isin(valid_items)]
    log.add("notas", "remove vinhos excluídos na limpeza", len(df))
    return df.reset_index(drop=True)


def k_core(ratings: pd.DataFrame, min_user: int = 5, min_item: int = 5,
           log: CleaningLog | None = None, max_iter: int = 50) -> pd.DataFrame:
    """Filtra iterativamente até todo usuário e todo vinho terem o mínimo de notas."""
    df = ratings
    for _ in range(max_iter):
        user_counts = df["user_id"].map(df["user_id"].value_counts())
        item_counts = df["item_id"].map(df["item_id"].value_counts())
        mask = (user_counts >= min_user) & (item_counts >= min_item)
        if mask.all():
            break
        df = df[mask]
    if log is not None:
        log.add("notas", f"k-core (usuário ≥ {min_user}, vinho ≥ {min_item})", len(df),
                usuarios=df["user_id"].nunique(), vinhos=df["item_id"].nunique())
    return df.reset_index(drop=True)


def build_dataset(raw_dir: Path = RAW_DIR, version: str | None = None, min_user: int = 5,
                  min_item: int = 5, save: bool = True) -> dict:
    """Pipeline completo de tratamento. Retorna as tabelas limpas e o log de passos."""
    log = CleaningLog()
    items_raw, ratings_raw = load_raw(raw_dir, version)
    items = clean_items(items_raw, log)
    ratings_all = clean_ratings(ratings_raw, items["item_id"], log)
    ratings = k_core(ratings_all, min_user, min_item, log)

    # Mantém o catálogo completo (limpo): vinhos fora do k-core podem servir de "itens novos".
    out = {"items": items, "ratings_all": ratings_all, "ratings": ratings, "log": log.to_frame()}
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name in ["items", "ratings_all", "ratings"]:
            out[name].to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        out["log"].to_csv(PROCESSED_DIR / "cleaning_log.csv", index=False)
    return out


def load_processed() -> dict:
    """Lê as tabelas já tratadas por build_dataset."""
    out = {name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
           for name in ["items", "ratings_all", "ratings"]}
    for col in LIST_COLS:
        out["items"][col] = out["items"][col].map(list)
    out["log"] = pd.read_csv(PROCESSED_DIR / "cleaning_log.csv")
    return out


def sparsity(ratings: pd.DataFrame) -> float:
    n_users, n_items = ratings["user_id"].nunique(), ratings["item_id"].nunique()
    return 1 - len(ratings) / (n_users * n_items)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=VERSIONS)
    ap.add_argument("--min-user", type=int, default=5)
    ap.add_argument("--min-item", type=int, default=5)
    a = ap.parse_args()
    data = build_dataset(version=a.version, min_user=a.min_user, min_item=a.min_item)
    print(data["log"].to_string(index=False))
    r = data["ratings"]
    print(f"\nk-core: {r['user_id'].nunique()} usuários, {r['item_id'].nunique()} vinhos, "
          f"{len(r)} notas, esparsidade {sparsity(r):.4%}")
