# app.py — Comentários + Dashboard (% por sentimento em PT-BR com cores por categoria)
import os, json, re, unicodedata
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from html import escape

# -----------------------------
# CONFIG / TEMA
# -----------------------------
st.set_page_config(page_title="Análise de Avaliações", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1000px; padding-top: 1.4rem;}
h1, h2, h3 { margin-top: .6rem !important; }

/* cards de comentários */
.comment-card{
  color:#111; border:1px solid #e5e7eb; border-left-width:6px;
  border-radius:8px; padding:8px 10px; margin:6px 0;
}
.header-row{ display:flex; justify-content:space-between; align-items:center; margin:0 0 4px 0; }
.meta{ font-size:12px; color:#111; opacity:.85; font-weight:600; }
.comment-text{ font-size:14px; line-height:1.3; white-space:pre-wrap; }
.badge{
  background:#ffffffd9; color:#111; border:1px solid #e5e7eb; border-radius:999px;
  padding:1px 8px; font-size:12px; display:inline-flex; gap:6px; align-items:center;
}
.badge svg{ width:12px; height:12px; display:block; }
.small-caption {font-size:12px; opacity:.8; margin-top:.2rem;}

/* Estilos para a página inicial */
.rating-summary { text-align: left; }
.rating-summary .score { font-size: 4rem; font-weight: bold; line-height: 1; margin: 0; }
.rating-summary .stars { font-size: 1.5rem; margin-top: -5px; margin-bottom: 5px; color: #f59e0b; }
.rating-summary .total-reviews { font-size: 1rem; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CAMINHOS PADRÃO (NOVO: Estrutura para múltiplas fontes)
# -----------------------------
PATHS = {
    "SIGEPE": {
        "csv": "./data/sigepe_dataFrame.csv",
        "json": "./sentiment_analysis/resources/outLLM/sigepe_sentiment_analysis.json"
    },
    "SouGov": {
        "csv": "./data/souGov_dataFrame.csv",
        "json": "./sentiment_analysis/resources/outLLM/sigepe_sentiment_analysis.json"
    }
}

# Sentimentos internos (não mexer)
VALID_SENTIMENTS = {"criticism","suggestion","positive feedback","not pertinent"}
SENT_ORDER = ["criticism","suggestion","positive feedback","not pertinent"]

# Rótulos e cores em PT-BR para gráficos
PT_LABEL = {
    "criticism": "Crítica",
    "suggestion": "Sugestão",
    "positive feedback": "Elogio",
    "not pertinent": "Não pertinente",
}
COLOR_MAP = {
    "positive feedback": "#16a34a",
    "criticism": "#dc2626",
    "suggestion": "#f59e0b",
    "not pertinent": "#6b7280",
}

# -----------------------------
# FUNÇÕES AUXILIARES (dados) - MODIFICADAS para aceitar caminhos
# -----------------------------
def load_csv(path):
    if not os.path.exists(path):
        st.error(f"ERRO: Arquivo CSV não encontrado no caminho esperado: {path}")
        st.info("Por favor, verifique se o arquivo existe e se a estrutura de pastas está correta.")
        st.stop()  # Para a execução do app se o arquivo não for encontrado
    return pd.read_csv(path, encoding="utf-8")

def load_json(path):
    if not os.path.exists(path):
        st.error(f"ERRO: Arquivo JSON não encontrado no caminho esperado: {path}")
        st.info("Por favor, verifique se o arquivo existe e se a estrutura de pastas está correta.")
        st.stop()  # Para a execução do app se o arquivo não for encontrado
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _detect_comment_col(df):
    for c in ["comments","Comentário","comentario","comment","texto","text"]:
        if c in df.columns: return c
    st.error("Não encontrei a coluna de comentários (ex.: 'comments')."); st.stop()

def _norm_text(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return ""
    s = str(x)
    if s.strip().lower() == "nan": return ""
    s = unicodedata.normalize("NFKC", s).replace("\xa0", " ")
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    return s.strip()

def _normalize_sentiment(s):
    s = str(s).strip().lower()
    mapping = {
        "critica":"criticism","crítica":"criticism","criticismo":"criticism",
        "sugestao":"suggestion","sugestão":"suggestion",
        "positivo":"positive feedback","elogio":"positive feedback",
        "nao pertinente":"not pertinent","não pertinente":"not pertinent",
    }
    s = mapping.get(s, s)
    return s if s in VALID_SENTIMENTS else None

def attach_sentiment_on_comments(df, info, id_col="ID"):
    df = df.copy()
    ccol = _detect_comment_col(df)
    has_comment = df[ccol].astype(str).map(_norm_text).ne("") & df[ccol].notna()
    idx = df[has_comment].index.to_list()

    preds = None
    if isinstance(info, dict) and isinstance(info.get("y_pred_text"), list):
        preds = [_normalize_sentiment(x) for x in info["y_pred_text"]]

    df["sentiment"] = None
    if preds is None or len(idx) == 0: return df

    n = min(len(preds), len(idx))
    if n > 0:
        df.loc[idx[:n], "sentiment"] = preds[:n]
        if len(preds) != len(idx):
            st.info(f"O número de predições ({len(preds)}) é diferente do número de comentários ({len(idx)}).")
    return df

# -----------------------------
# FUNÇÕES AUXILIARES (UI/estilo)
# -----------------------------
def style_for(sentiment):
    palette = {
        "positive feedback": {"bg": "#d9f4e3", "edge": "#16a34a"},
        "criticism":         {"bg": "#ffd9d9", "edge": "#dc2626"},
        "suggestion":        {"bg": "#fff0b3", "edge": "#f59e0b"},
        "not pertinent":     {"bg": "#ececec", "edge": "#6b7280"},
        None:                {"bg": "#f7f7f7", "edge": "#9ca3af"},
        pd.NA:               {"bg": "#f7f7f7", "edge": "#9ca3af"},
    }
    return palette.get(sentiment, palette[None])

def nota_span_html(v):
    try: x = float(v)
    except Exception: return ""
    show = str(int(x)) if abs(x - int(x)) < 1e-9 else f"{x:.1f}"
    return f"""<span class="badge"><svg viewBox="0 0 24 24" aria-hidden="true"><path fill="#f59e0b" d="M12 .587l3.668 7.431L24 9.748l-6 5.848L19.335 24 12 19.897 4.665 24 6 15.596 0 9.748l8.332-1.73z"/></svg><strong>{escape(show)}</strong></span>"""

def fmt_id(v):
    if v is None or (isinstance(v, float) and pd.isna(v)): return None
    try:
        f = float(v); return str(int(f)) if abs(f - int(f)) < 1e-9 else str(v)
    except Exception: return str(v)

def plot_perc_barh(percs, title, xlabel):
    labels_pt = [PT_LABEL[s] for s in SENT_ORDER]
    values = [float(percs[s]) for s in SENT_ORDER]
    colors = [COLOR_MAP[s] for s in SENT_ORDER]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.barh(labels_pt, values, color=colors)
    for i, v in enumerate(values): ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, max(100, (max(values) if values else 0) + 5))
    return fig

def plot_nota_distribution(percs):
    labels = percs.index.astype(str).to_list()
    values = percs.values.tolist()
    colors = ['#16a34a', '#84cc16', '#f59e0b', '#f97316', '#dc2626']
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', va='center')
    ax.set_title("Distribuição das Avaliações por Nota")
    ax.set_xlabel("% do Total de Avaliações")
    ax.set_xlim(0, max(100, max(values) * 1.1 if values else 100))
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    return fig

# -----------------------------
# MENU LATERAL E SELEÇÃO DE DADOS
# -----------------------------
st.sidebar.title("Fonte de Dados")
# NOVO: Seletor para escolher entre os DataFrames
fonte_selecionada = st.sidebar.selectbox("Selecione a fonte de dados:", list(PATHS.keys()))

st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Início", "Comentários", "Dashboard"], index=0)

# -----------------------------
# CARREGAMENTO DE DADOS (DINÂMICO)
# -----------------------------
# NOVO: Carrega os dados com base na seleção do usuário
csv_path = PATHS[fonte_selecionada]["csv"]
json_path = PATHS[fonte_selecionada]["json"]

# MODIFICADO: Os nomes das variáveis agora são genéricos
dataFrame_bruto = load_csv(csv_path)
info_sentimentos = load_json(json_path)

comment_col = _detect_comment_col(dataFrame_bruto)

# MODIFICADO: A variável principal agora se chama 'dataFrame_processado'
# e ela conterá os dados da fonte que você escolher (SIGEPE ou a outra).
dataFrame_processado = dataFrame_bruto.copy()
dataFrame_processado = attach_sentiment_on_comments(dataFrame_processado, info_sentimentos, id_col="ID")

# Normalizações
if "Data" in dataFrame_processado.columns:
    try: dataFrame_processado["Data"] = pd.to_datetime(dataFrame_processado["Data"], errors="coerce", dayfirst=True)
    except Exception: dataFrame_processado["Data"] = pd.to_datetime(dataFrame_processado["Data"], errors="coerce")
dataFrame_processado["sentiment"] = dataFrame_processado["sentiment"].map(lambda x: x if x in VALID_SENTIMENTS else None)


# =========================================================================================
# RENDERIZAÇÃO DAS PÁGINAS (o código abaixo usa 'dataFrame_processado' e funciona para ambos)
# =========================================================================================

# -----------------------------
# PÁGINA: INÍCIO
# -----------------------------
if page == "Início":
    st.title(f"Visão Geral: {fonte_selecionada}")
    st.markdown("---")
    
    if "Nota" not in dataFrame_processado.columns:
        st.warning("Coluna 'Nota' não encontrada. Não é possível exibir o resumo das avaliações.")
        st.stop()

    notas = pd.to_numeric(dataFrame_processado["Nota"], errors="coerce").dropna()
    notas = notas[notas.between(1, 5)]

    if notas.empty:
        st.info("Nenhuma avaliação com nota válida (1 a 5) encontrada para exibir o resumo.")
        st.stop()

    nota_media = notas.mean()
    total_avaliacoes = len(notas)
    nota_counts = notas.value_counts().reindex([5, 4, 3, 2, 1], fill_value=0)
    nota_percs = (nota_counts / total_avaliacoes * 100) if total_avaliacoes > 0 else nota_counts

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown(f"""
        <div class="rating-summary">
            <p class="score">{nota_media:.2f}</p>
            <p class="stars">{"⭐" * int(round(nota_media))}{"☆" * (5 - int(round(nota_media)))}</p>
            <p class="total-reviews">{total_avaliacoes:,} avaliações</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        fig = plot_nota_distribution(nota_percs)
        st.pyplot(fig); plt.close(fig)

# -----------------------------
# PÁGINA: COMENTÁRIOS
# -----------------------------
elif page == "Comentários":
    st.title(f"Comentários: {fonte_selecionada}")
    cols = [comment_col]
    if "Nota" in dataFrame_processado.columns: cols.append("Nota")
    if "ID" in dataFrame_processado.columns: cols.append("ID")
    view = dataFrame_processado[cols + ["sentiment"]].copy()

    c1, c2 = st.columns([2, 1])
    with c1:
        label_to_key = {"Todos": None, "Crítica": "criticism", "Sugestão": "suggestion", "Elogio": "positive feedback", "Não pertinente": "not pertinent"}
        choice = st.radio("Filtrar por sentimento:", list(label_to_key.keys()), horizontal=True, index=0)
        chosen_key = label_to_key[choice]
    with c2:
        nota_range = None
        if "Nota" in view.columns:
            view["__nota_num__"] = pd.to_numeric(view["Nota"], errors="coerce")
            s = view["__nota_num__"].dropna()
            min_avail, max_avail = (int(s.min()), int(s.max())) if not s.empty else (1, 5)
            nota_range = st.slider("Faixa de nota", 1, 5, (min_avail, max_avail), 1)

    if chosen_key: view = view[view["sentiment"] == chosen_key]
    if nota_range:
        lo, hi = nota_range
        view = view[view["__nota_num__"].between(lo, hi, inclusive="both") | view["__nota_num__"].isna()]

    user_num = 0
    for _, row in view.iterrows():
        txt = _norm_text(row[comment_col])
        if not txt: continue
        user_num +=1
        sty = style_for(row.get("sentiment"))
        nota_html = nota_span_html(row.get("Nota")) if "Nota" in view.columns else ""
        id_str = fmt_id(row.get("ID")) if "ID" in view.columns else str(user_num)
        st.markdown(f"""<div class="comment-card" style="background:{sty['bg']}; border-left-color:{sty['edge']};"><div class="header-row"><div class="meta">Usuário {escape(id_str)}</div>{nota_html}</div><div class="comment-text">{escape(txt)}</div></div>""", unsafe_allow_html=True)
    
    if user_num == 0: st.info("Nenhum comentário para o filtro selecionado.")

# -----------------------------
# PÁGINA: DASHBOARD
# -----------------------------
else:
    st.title(f"Dashboard de Sentimentos: {fonte_selecionada}")

    def pct_vector(df_subset):
        counts = df_subset["sentiment"].value_counts().reindex(SENT_ORDER, fill_value=0)
        total = int(counts.sum())
        percs = (counts / total * 100) if total > 0 else pd.Series([0]*len(SENT_ORDER), index=SENT_ORDER, dtype=float)
        return counts, percs.round(1), total

    base = st.radio("Filtrar por:", ["Todos", "Assunto", "Funcionalidade", "Nota"], horizontal=True, index=0)
    df_base = dataFrame_processado[dataFrame_processado["sentiment"].notna()].copy()

    if base == "Todos":
        st.subheader("Distribuição geral de sentimentos (%)")
        counts, percs, total = pct_vector(df_base)
        notas = pd.to_numeric(df_base.get("Nota"), errors="coerce")
        n_notas = int(notas.notna().sum())
        nota_media = float(notas.mean()) if n_notas > 0 else None
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rotulado", f"{total:,}".replace(",", "."))
        c2.metric("Com nota", f"{n_notas:,}".replace(",", "."))
        c3.metric("Nota média", f"{nota_media:.2f}" if nota_media is not None else "-")
        
        st.dataframe(pd.DataFrame({"Sentimento": [PT_LABEL[s] for s in SENT_ORDER], "Quantidade": counts, "Percentual (%)": percs}), use_container_width=True, hide_index=True)
        fig = plot_perc_barh(percs, "Distribuição geral de sentimentos", "% do total rotulado")
        st.pyplot(fig); plt.close(fig)

    else: # Filtros por Assunto, Funcionalidade, Nota
        if base not in df_base.columns and base != "Nota":
            st.warning(f"Coluna '{base}' não encontrada no CSV."); st.stop()
        
        st.subheader(f"Distribuição (%) — Filtro por {base}")
        
        if base == "Nota":
            df_base["__nota_num__"] = pd.to_numeric(df_base["Nota"], errors="coerce")
            opts = sorted([int(n) for n in df_base["__nota_num__"].dropna().unique() if 1 <= n <= 5])
            escolha = st.selectbox(f"{base}:", ["Todas"] + opts, index=0)
            if escolha != "Todas": df_base = df_base[df_base["__nota_num__"] == escolha]
        else: # Assunto ou Funcionalidade
            opts = sorted(df_base[base].dropna().astype(str).unique().tolist())
            escolha = st.selectbox(f"{base}:", ["Todos"] + opts, index=0)
            if escolha != "Todos": df_base = df_base[df_base[base].astype(str) == escolha]

        counts, percs, total = pct_vector(df_base)
        if total == 0:
            st.info("Sem dados para o filtro selecionado."); st.stop()

        c1, c2 = st.columns(2)
        c1.metric("Total no filtro", f"{total:,}".replace(",", "."))
        notas_filtro = pd.to_numeric(df_base.get("Nota"), errors="coerce")
        media_filtro = notas_filtro.mean() if notas_filtro.notna().any() else None
        c2.metric("Nota média no filtro", f"{media_filtro:.2f}" if media_filtro is not None else "-")

        st.dataframe(pd.DataFrame({"Sentimento": [PT_LABEL[s] for s in SENT_ORDER], "Quantidade": counts, "Percentual (%)": percs}), use_container_width=True, hide_index=True)
        fig = plot_perc_barh(percs, f"Distribuição por sentimento — {base}: {escolha}", "% no subconjunto")
        st.pyplot(fig); plt.close(fig)