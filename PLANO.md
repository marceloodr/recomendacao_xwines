# Plano — Trabalho T1 de Sistemas de Recomendação

## 1. O que o enunciado pede

| Requisito | Detalhe |
|---|---|
| Técnica | **Uma** entre: vizinhos próximos (KNN), baseada em conteúdo + escores de produto, baseada em conhecimento, ou em grafos |
| Base | Pública, com **ratings por usuário** (senão a nota é zero), diferente da dos outros grupos e divulgada no fórum do Moodle |
| Originalidade | Copiar uma modelagem da internet, ou quase copiar, dá nota zero |
| Interface simples com 3 cenários | (1) lista para um **usuário antigo**; (2) lista para um **usuário novo**; (3) um **item novo** do catálogo para um usuário antigo |
| Entregáveis | PPT (seções a–i), a base de dados, o código-fonte e a apresentação em sala |

## 2. Decisões a tomar antes de tudo

### 2.1 Técnica (recomendação: **baseada em conteúdo + escores de produto**)

Os cenários 2 e 3 são **cold start**, e é isso que pesa na escolha:

- **KNN colaborativo**: atende bem o cenário 1, mas não consegue recomendar um item novo (cenário 3), porque o item não tem nenhum rating. Seria preciso um "remendo" fora da técnica.
- **Conteúdo + escores**: atende os três cenários de forma natural:
  - *Usuário antigo*: monta-se um perfil do usuário (média ponderada pelos ratings dos vetores de atributos dos itens que ele avaliou) e calcula-se a similaridade de cosseno com os itens que ele ainda não viu. O resultado é combinado com um escore de produto (média bayesiana do rating, popularidade).
  - *Usuário novo*: um onboarding curto em que o usuário escolhe gêneros ou atributos de que gosta, ou avalia de 3 a 5 itens. Isso vira um perfil inicial. Sem nenhuma entrada, usa-se o ranking por escore de produto (média bayesiana).
  - *Item novo*: o item já tem atributos (gênero, descrição, tags), então dá para vetorizá-lo na hora e calcular a similaridade com o perfil de cada usuário antigo. O sistema mostra o item para o usuário certo, ou lista os usuários com maior afinidade com ele.

**Consequência**: a base precisa ter **metadados dos itens** (texto, gênero, tags, categorias), além de `user_id, item_id, rating`. Ter um timestamp também ajuda no split de treino e teste.

### 2.2 Base de dados

Já escolhidas por outros grupos (fórum, out/2026): Food.com, Goodbooks-10k, Amazon Reviews 2023 (Beauty), Last.FM (HetRec) e BoardGameGeek Reviews.

Candidatas avaliadas inicialmente:

Candidatas com ratings por usuário **e** atributos de item:

| Base | Ratings | Atributos de item | Observação |
|---|---|---|---|
| **BoardGameGeek Reviews** (Kaggle) | 1–10 por usuário | categorias, mecânicas, descrição, ano | Pouco usada, metadados ricos |
| **Food.com Recipes & Interactions** (Kaggle) | 0–5 + texto | ingredientes, tags, tempo, nutrição | Ótima para TF-IDF |
| **Anime Recommendations / MyAnimeList** (Kaggle) | 1–10 | gêneros, tipo, estúdio, sinopse | Bastante usada, então conferir com os outros grupos |
| **Goodreads (UCSD Book Graph)**, subconjunto por gênero | 1–5 | descrição, shelves, autor | Grande, precisa de amostragem |
| **Amazon Reviews 2023**, uma categoria (ex.: Video Games) | 1–5 | título, descrição, categorias, preço | Precisa filtrar k-core |

**Escolhida: X-Wines** (de Azambuja, Morais & Filipe, 2023, *Big Data Cogn. Comput.* 7(1), 20; licença CC0). A Food.com já tinha sido escolhida por outro grupo. **Ação**: publicar no fórum do Moodle o quanto antes. Detalhes na seção 2.4.

### 2.4 X-Wines — o que a base tem e o que muda no plano

**Versões** (GitHub `rogerioxavier/X-Wines`):

| Versão | Vinhos | Usuários | Notas | Onde |
|---|---|---|---|---|
| Test | 100 | 636 | 1 mil | GitHub (público) — usada só para desenvolver o código |
| **Slim** | 1.007 | 10.561 | 150 mil | Kaggle `rogerioxavier/x-wines-slim-version` — **versão recomendada** |
| Full | 100.646 | 1.056.079 | 21 milhões | Google Drive do autor — pesada; só faz sentido com amostragem |

**Colunas**:
- vinhos: `WineID, WineName, Type, Elaborate, Grapes, Harmonize, ABV, Body, Acidity, Code, Country, RegionID, RegionName, WineryID, WineryName, Website, Vintages`;
- notas: `RatingID, UserID, WineID, Vintage, Rating, Date`.

**Particularidades que afetam o trabalho:**
- **Escala de 1 a 5 em passos de 0,5**, sem a concentração extrema da Food.com (na amostra Test a moda é 4,0 e ~20% das notas são ≥ 4,5). Na Slim, 4,0 é a moda (36%) e só 22,4% das notas são ≥ 4,5: **relevante = nota ≥ 4,5**.
- **Na versão Full, o mesmo usuário pode avaliar o mesmo vinho em safras diferentes.** O sistema recomenda o rótulo, então fica só a avaliação mais recente, e a quantidade removida entra no slide de tratamento.
- **Os campos de lista estão em formato de string** (`Grapes`, `Harmonize`, `Vintages`). As safras misturam anos com `'N.V.'` (sem safra).
- **Não há data de cadastro do vinho.** Para o cenário de item novo, a data da **primeira avaliação** é usada como data de entrada no catálogo, e os 5% de vinhos que entraram por último saem do treino.
- **As notas têm data (2012–2021)**, o que permite o split temporal por usuário.

**Vetor de atributos do vinho** (núcleo do modelo de conteúdo). Cada bloco é um multi-hot ponderado por IDF e normalizado, e os pesos dos blocos são ajustados na validação:
1. **Uvas** (`Grapes`): ex.: Merlot, Touriga Nacional.
2. **Harmonização** (`Harmonize`): ex.: Beef, Pasta, Shellfish.
3. **Estilo**: tipo (Red, White, Rosé, Sparkling, Dessert…), elaboração (varietal × blend), corpo e acidez.
4. **Origem**: país, região e vinícola.
5. **Numéricos**: teor alcoólico, nº de uvas e nº de safras.

**Os 3 cenários na X-Wines:**
- *Usuário antigo*: perfil a partir dos vinhos que ele avaliou, cosseno com os vinhos que ele ainda não avaliou, combinado com o escore de produto (média bayesiana + popularidade).
- *Usuário novo*: onboarding em que a pessoa escolhe tipos, uvas, pratos com que quer harmonizar, corpo e países, com filtros duros (só tinto, uvas a evitar, teor alcoólico máximo). Sem nenhuma entrada, mostra o top pelo escore de produto.
- *Item novo*: a pessoa cadastra um vinho (nome, tipo, uvas, harmonização, corpo, acidez, país) na interface. O vinho é vetorizado na hora, e o app lista os usuários antigos com maior afinidade.

**Ideias de EDA específicas**: tipo, corpo e acidez (catálogo × notas); top uvas, harmonizações e países; nota média por tipo, país e uva; **idade do vinho no momento da avaliação × nota**; teor alcoólico × nota.

**Código**: o que foi feito para a Food.com foi adaptado. Divisão, modelo, métricas, baselines e ajuste são genéricos (`item_id`). Carga, limpeza e vetorização foram reescritas para vinhos. O material da Food.com está em `_foodcom_antigo/`.

### 2.3 Grupo
Definir os integrantes (até 5) e dividir as frentes: dados/EDA, modelo, avaliação, interface, slides.

## 3. Estrutura do projeto

```
trabalho_final/
├── data/
│   ├── raw/                # base original (entregável 2)
│   └── processed/          # base limpa (parquet/csv)
├── notebooks/
│   ├── 01_eda.ipynb        # estatísticas descritivas e gráficos
│   ├── 02_preprocessing.ipynb
│   ├── 03_modelo.ipynb     # estimação
│   └── 04_avaliacao.ipynb  # métricas
├── src/
│   ├── data.py             # carga e limpeza
│   ├── features.py         # vetorização dos itens (TF-IDF, one-hot de gêneros)
│   ├── model.py            # classe ContentRecommender (fit / recommend_*)
│   ├── metrics.py          # Precision@K, Recall@K, NDCG, MAP, RMSE, cobertura
│   └── split.py            # treino/teste
├── app/
│   └── app.py              # interface Streamlit com os 3 cenários
├── slides/                 # PPT final
├── requirements.txt
└── README.md               # como rodar
```

Stack: Python, pandas, numpy, scikit-learn (TF-IDF, cosseno), matplotlib/seaborn e Streamlit.

## 4. Etapas detalhadas

### Etapa 1 — Aquisição e EDA (slide c)
- Baixar a base e salvar em `data/raw/`.
- Estatísticas: número de usuários, itens e ratings; **esparsidade** (`1 - n_ratings / (n_users * n_items)`); distribuição dos ratings; média por usuário e por item.
- Gráficos: histograma dos ratings; **cauda longa** (ratings por item, em escala log); ratings por usuário; top-N itens; distribuição dos atributos (gêneros/categorias); série temporal de ratings, se houver timestamp.

### Etapa 2 — Tratamento dos dados (slide e)
- Remover duplicados (mesmo user–item: ficar com o mais recente ou com a média).
- Missing: itens sem metadados (remover ou imputar "desconhecido") e ratings nulos.
- Normalizar a escala do rating, se necessário, e tratar ratings implícitos (ex.: 0 = "sem nota" no Food.com/Book-Crossing).
- **Filtro k-core** (ex.: usuários com ≥ 5 ratings e itens com ≥ 5), para reduzir a esparsidade e ter algo para testar.
- Amostragem, se a base for muito grande, para caber na memória e manter a interface ágil.
- Pré-processar o texto: minúsculas, stopwords, lematização opcional.
- Documentar o antes e o depois (quantidade de linhas em cada passo) para o slide.

### Etapa 3 — Conceitos teóricos (slide d)
- Representação dos itens: TF-IDF sobre o texto, one-hot/multi-hot de gêneros, concatenação ponderada.
- Perfil do usuário: `p_u = Σ (r_ui − r̄_u) · v_i / Σ |r_ui − r̄_u|`. Centralizar na média do usuário faz itens mal avaliados pesarem contra.
- Similaridade de cosseno.
- Escore de produto: **média bayesiana (IMDb weighted rating)** `WR = (v/(v+m))·R + (m/(v+m))·C`.
- Escore final (híbrido dentro da própria técnica): `score(u,i) = α · cos(p_u, v_i) + (1−α) · WR_i normalizado`, com α ajustado na validação.
- Cold start: como cada cenário é tratado.

### Etapa 4 — Treino e teste (slide f)
- **Split por usuário, temporal ou aleatório**: para cada usuário, os últimos 20% dos ratings (pelo timestamp) vão para o teste, e o resto para o treino. Sem timestamp, usar 80/20 aleatório por usuário (estratificado).
- Separar uma validação dentro do treino para ajustar α, `m`, o número de features do TF-IDF e o limiar de "relevante".
- Definir **item relevante**: rating ≥ 4 (escala 1–5) ou ≥ 7 (escala 1–10).
- Simular cold start na avaliação:
  - *Usuário novo*: separar ~10% dos usuários inteiros, dar ao modelo apenas 3–5 ratings deles e avaliar o restante.
  - *Item novo*: separar ~5% dos itens inteiros (remover todos os ratings deles do treino) e medir se o modelo os recomenda aos usuários que de fato os avaliaram bem.

### Etapa 5 — Estimação do modelo (slide g)
- Implementar `ContentRecommender` com `fit(train)`, `recommend_existing_user(u, k)`, `recommend_new_user(prefs, k)` e `recommend_new_item(item_features, k_users)`.
- Implementação **própria**, sem copiar notebooks do Kaggle (risco de nota zero). Usar só o sklearn para TF-IDF e cosseno.
- Baselines para comparação: **popularidade** e **aleatório**. Opcionalmente, um KNN item-item simples, só como referência.
- Ajustar os hiperparâmetros na validação.

### Etapa 6 — Avaliação (slide h)
- Métricas de ranking @K (K = 5, 10): **Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate**.
- Métricas de beyond-accuracy: **cobertura** de catálogo, **diversidade** intra-lista e novidade (opcional).
- Se o modelo prever notas, também **RMSE/MAE**.
- Tabela comparando o modelo com os baselines, separada por cenário: antigo, novo usuário e novo item.
- Conferir no material de aula quais métricas o professor cobrou e priorizar essas.

### Etapa 7 — Interface (slide i)
Streamlit com 3 abas:
1. **Usuário antigo**: selectbox com `user_id`, mostra o histórico (top itens avaliados) e o Top-N recomendado com o escore e o "porquê" (atributos em comum).
2. **Usuário novo**: multiselect de gêneros/atributos e/ou avaliação de alguns itens populares, depois o Top-N.
3. **Item novo**: formulário para cadastrar um item (título, gêneros, descrição). O item é adicionado ao catálogo em memória, e o app mostra os usuários antigos com maior afinidade. Também deve dar para escolher um usuário e ver o item novo aparecendo na lista dele.

Gravar um vídeo ou tirar prints de backup caso a demo ao vivo falhe.

### Etapa 8 — Slides e entrega
Seções obrigatórias do PPT:
- a. Capa
- b. Descrição do sistema (diagrama do pipeline)
- c. EDA
- d. Teoria
- e. Tratamento dos dados
- f. Split
- g. Modelo
- h. Métricas
- i. Demo da interface (prints e ao vivo)

Para entregar: o PPT, a base (`data/`, ou link + script de download se for grande), o código-fonte (repositório zipado + README com `pip install -r requirements.txt` e `streamlit run app/app.py`) e a apresentação em sala.

## 5. Cronograma sugerido

| Semana | Entregas |
|---|---|
| 1 | Grupo definido, base escolhida e publicada no Moodle, EDA |
| 2 | Tratamento dos dados, split, modelo base funcionando |
| 3 | Avaliação + baselines + ajuste fino; início da interface |
| 4 | Interface finalizada, slides, ensaio da apresentação |

## 6. Checklist de riscos

- [ ] A base tem `user_id` + `item_id` + `rating` explícito? (senão, nota zero)
- [ ] A base tem atributos de item para o cold start de item?
- [ ] Nenhum outro grupo usa a mesma base?
- [ ] Código autoral (sem copiar notebook pronto)?
- [ ] Os 3 cenários funcionam na interface?
- [ ] Todas as seções a–i estão no PPT?
