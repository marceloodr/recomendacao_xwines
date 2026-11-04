"""Interface Streamlit do recomendador de vinhos (X-Wines) com os 3 cenários do enunciado.

Rodar a partir da raiz do projeto:  streamlit run app/app.py
Requer models/recommender.pkl (gerado ao final de notebooks/03_avaliacao.ipynb) e
data/processed/ (gerado por `python -m src.data`).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # permite `import src...` e desserializar o pickle

from src.data import load_processed  # noqa: E402

MODEL_PATH = ROOT / "models" / "recommender.pkl"
NEW_ID_BASE = 900_000_000  # ids de vinhos cadastrados na interface, fora da faixa da base
MAX_CATALOG_NOVELTIES = 100

st.set_page_config(page_title="Recomendador de Vinhos", page_icon="🍷", layout="wide")


# ------------------------------------------------------------------ carga
@st.cache_resource(show_spinner="Carregando modelo...")
def load_system():
    """Modelo treinado + notas + vinhos do catálogo que estão fora do modelo (itens 'novos')."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    data = load_processed()
    ratings = data["ratings"]
    outside = data["items"][~data["items"]["item_id"].isin(model.item_pos.index)]
    # Vinhos do catálogo sem nenhuma nota no treino: entram como "recém-adicionados".
    novelties = outside.sample(min(MAX_CATALOG_NOVELTIES, len(outside)), random_state=0)
    if len(novelties):
        model.add_items(novelties)
    return model, ratings, novelties["item_id"].tolist()


def show(df: pd.DataFrame, cols: dict):
    """Tabela de resultados com colunas renomeadas (cols: nome_interno -> rótulo)."""
    out = df[[c for c in cols if c in df]].copy()
    if "grapes" in out:
        out["grapes"] = out["grapes"].map(lambda g: ", ".join(g))
    out.index = range(1, len(out) + 1)
    st.dataframe(out.rename(columns=cols), width="stretch")


REC_COLS = {"name": "Vinho", "winery": "Vinícola", "type": "Tipo", "country": "País",
            "grapes": "Uvas", "score": "Score", "nota_bayesiana": "Nota média*",
            "n_notas": "Nº notas", "porque": "Por que recomendamos"}
NEW_COLS = {"name": "Vinho", "winery": "Vinícola", "type": "Tipo", "country": "País",
            "grapes": "Uvas", "similaridade": "Similaridade", "porque": "Por que recomendamos"}


try:
    model, ratings, catalog_new_ids = load_system()
except FileNotFoundError:
    st.error("Modelo não encontrado. Rode `notebooks/03_avaliacao.ipynb` até o final (gera "
             f"`{MODEL_PATH.relative_to(ROOT)}`) e `python -m src.data` (gera `data/processed/`).")
    st.stop()

fz = model.featurizer
types_all = fz.vocabulary("style", "type")
body_all = fz.vocabulary("style", "body")
acidity_all = fz.vocabulary("style", "acidity")
countries_all = fz.vocabulary("origin", "country")
grapes_all = fz.vocabulary("grape")
harm_all = fz.vocabulary("harm")

st.title("🍷 Recomendador de Vinhos")
st.caption("Sistema baseado em conteúdo + escore de produto (média bayesiana e popularidade) · "
           "base X-Wines · *Nota média bayesiana: média das notas ajustada pelo nº de avaliações.")

k = st.sidebar.slider("Quantidade de recomendações", 3, 20, 10)

tab1, tab2, tab3 = st.tabs(["1 · Usuário antigo", "2 · Usuário novo", "3 · Vinho novo no catálogo"])

# ------------------------------------------------------------------ cenário 1
with tab1:
    st.subheader("Recomendar para um usuário que já avaliou vinhos")
    counts = ratings["user_id"].value_counts()
    user_id = st.selectbox("Usuário", model.user_ids.tolist(),
                           format_func=lambda u: f"Usuário {u} ({counts.get(u, 0)} notas)")
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Vinhos mais bem avaliados por ele**")
        hist = model.user_history(user_id, ratings, 8)
        hist["grapes"] = hist["grapes"].map(lambda g: ", ".join(g))
        st.dataframe(hist[["name", "type", "rating"]].rename(
            columns={"name": "Vinho", "type": "Tipo", "rating": "Nota"}),
            hide_index=True, width="stretch")
    with right:
        st.markdown(f"**Top {k} recomendados**")
        recs = model.recommend_existing_user(user_id, k)
        if recs.empty:
            st.info("Nenhum vinho atende aos filtros.")
        else:
            show(recs, REC_COLS)

# ------------------------------------------------------------------ cenário 2
with tab2:
    st.subheader("Recomendar para quem ainda não tem histórico")
    st.write("Diga do que você gosta e/ou avalie alguns vinhos. Sem nada preenchido, "
             "mostramos os vinhos de maior escore de produto (os mais bem avaliados e populares).")
    c1, c2 = st.columns(2)
    with c1:
        p_types = st.multiselect("Tipos de vinho", types_all, key="p_types")
        p_grapes = st.multiselect("Uvas preferidas", grapes_all, key="p_grapes")
        p_harm = st.multiselect("Para harmonizar com", harm_all, key="p_harm")
    with c2:
        p_body = st.multiselect("Corpo", body_all, key="p_body")
        p_acid = st.multiselect("Acidez", acidity_all, key="p_acid")
        p_countries = st.multiselect("Países", countries_all, key="p_countries")

    popular = model.items.iloc[model.n_ratings.argsort()[::-1][:30]]
    chosen = st.multiselect("Vinhos que você já provou (populares)", popular["item_id"].tolist(),
                            format_func=lambda i: popular.set_index("item_id").loc[i, "name"],
                            key="p_rated")
    rated = {i: st.slider(popular.set_index("item_id").loc[i, "name"], 1.0, 5.0, 4.0, 0.5,
                          key=f"rate_{i}") for i in chosen}

    if st.button("Recomendar", type="primary", key="go2"):
        prefs = {"types": p_types, "grapes": p_grapes, "harmonize": p_harm,
                 "body": p_body, "acidity": p_acid, "countries": p_countries}
        recs = model.recommend_new_user(prefs, rated, k)
        if recs.empty:
            st.info("Nenhum vinho atende aos filtros.")
        else:
            if not any(prefs.values()) and not rated:
                st.caption("Sem preferências: ranking por escore de produto.")
            show(recs, REC_COLS)

# ------------------------------------------------------------------ cenário 3
def affinity_report(user_id, item_id):
    """Quão bem um vinho (sem notas) combina com o perfil de um usuário antigo."""
    u, i = model.user_pos[user_id], model.item_pos[item_id]
    sim = (model.P[u] @ model.X.T).toarray().ravel()
    unseen = np.ones(len(sim), dtype=bool)
    unseen[model.seen[u].indices] = False
    pool = sim[unseen]
    rank = int((pool > sim[i]).sum()) + 1
    return sim[i], rank, int(unseen.sum()), model.explain(model.P[u], i)


with tab3:
    st.subheader("Recomendar um vinho recém-adicionado a um usuário antigo")
    st.write("O vinho novo não tem nenhuma nota. Ele é posicionado só pelos seus atributos "
             "(uvas, estilo, origem, harmonização) e comparado com o perfil de gosto do usuário, "
             "montado a partir do que ele já avaliou.")
    if "new_ids" not in st.session_state:
        st.session_state.new_ids = []

    with st.expander("➕ Cadastrar um vinho novo no catálogo", expanded=False):
        with st.form("novo_vinho"):
            a, b = st.columns(2)
            name = a.text_input("Nome", "Tannat Reserva (novo)")
            winery = b.text_input("Vinícola", "Vinícola Exemplo")
            n_type = a.selectbox("Tipo", types_all, index=types_all.index("Red") if "Red" in types_all else 0)
            n_country = b.selectbox("País", countries_all)
            n_region = a.text_input("Região", "Desconhecido")
            n_abv = b.number_input("Teor alcoólico (%)", 5.0, 25.0, 13.5, 0.5)
            n_grapes = st.multiselect("Uvas (ao menos uma)", grapes_all)
            n_harm = st.multiselect("Harmoniza com", harm_all)
            c, d = st.columns(2)
            n_body = c.selectbox("Corpo", body_all)
            n_acid = d.selectbox("Acidez", acidity_all)
            n_vintage = st.text_input("Safra", "2023")
            submitted = st.form_submit_button("Cadastrar", type="primary")
        if submitted:
            if not n_grapes:
                st.warning("Escolha ao menos uma uva.")
            else:
                new_id = NEW_ID_BASE + len(st.session_state.new_ids)
                while new_id in model.item_pos.index:  # modelo é compartilhado entre sessões
                    new_id += 1
                novo = pd.DataFrame([{
                    "item_id": new_id, "name": name, "type": n_type,
                    "elaborate": "Varietal/100%" if len(n_grapes) == 1 else "Assemblage/Blend",
                    "grapes": n_grapes, "harmonize": n_harm, "abv": n_abv, "body": n_body,
                    "acidity": n_acid, "country": n_country, "region": n_region, "winery": winery,
                    "vintages": [n_vintage], "n_grapes": len(n_grapes), "n_vintages": 1,
                }])
                model.add_items(novo)
                st.session_state.new_ids.append(new_id)
                st.session_state.sel_new = new_id
                st.success(f"'{name}' cadastrado. Selecione-o abaixo.")

    new_ids = st.session_state.new_ids + catalog_new_ids
    if not new_ids:
        st.info("Nenhum vinho novo ainda. Cadastre um acima.")
    else:
        names = model.items.set_index("item_id")["name"]
        label = lambda i: names.loc[i] + (" · cadastrado agora" if i in st.session_state.new_ids else " · novo no catálogo")
        c1, c2 = st.columns(2)
        sel = c1.selectbox("Vinho novo", new_ids, key="sel_new", format_func=label)
        counts3 = ratings["user_id"].value_counts()
        u = c2.selectbox("Usuário antigo", model.user_ids.tolist(), key="u3",
                         format_func=lambda x: f"Usuário {x} ({counts3.get(x, 0)} notas)")

        def pick_top_fan():
            st.session_state.u3 = int(model.recommend_new_item(st.session_state.sel_new, 1)["user_id"].iloc[0])
        c2.button("Escolher o usuário com maior afinidade", on_click=pick_top_fan)

        w = model.items.iloc[model.item_pos[sel]]
        sim, rank, n_unseen, why = affinity_report(u, sel)
        top_n = max(k, int(0.1 * n_unseen))  # "recomendado" = entre os 10% mais afins ao perfil
        st.markdown(f"### {w['name']} → usuário {u}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Similaridade com o perfil", f"{sim:.2f}")
        m2.metric("Posição entre os vinhos que ele não provou", f"{rank}º de {n_unseen}")
        m3.metric("Veredito", "Recomendar ✅" if rank <= top_n else "Não prioritário")
        st.caption(f"Em comum com o gosto do usuário: {why}. "
                   f"(Recomendado = entre os {top_n} mais afins, os 10% do topo.)")

        l, r = st.columns(2)
        with l:
            st.markdown(f"**Vitrine de novidades do usuário {u}**")
            showcase = model.recommend_new_items_for_user(u, min(k, 10))
            show(showcase, NEW_COLS)
        with r:
            st.markdown(f"**Outros usuários com afinidade com '{w['name']}'**")
            fans = model.recommend_new_item(sel, k)
            fans.index = range(1, len(fans) + 1)
            st.dataframe(fans.rename(columns={"user_id": "Usuário", "afinidade": "Afinidade",
                                              "similaridade": "Similaridade", "porque": "Por que"}),
                         width="stretch")
