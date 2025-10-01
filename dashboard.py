# app.py — Versão Final Completa
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
# CAMINHOS PADRÃO (CORRIGIDO)
# -----------------------------
PATHS = {
    "SIGEPE": {
        "csv": "./data/sigepe_dataFrame.csv",
        "json": "./sentiment_analysis/resources/outLLM/sigepe_sentiment_analysis.json"
    },
    "SouGov": {
        "csv": "./data/souGov_dataFrame.csv",
        "json": "./sentiment_analysis/resources/outLLM/souGov_sentiment_analysis.json"
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
# FUNÇÕES AUXILIARES (dados)
# -----------------------------
@st.cache_data
def load_data(csv_path, json_path):
    df_bruto = pd.read_csv(csv_path, encoding="utf-8")
    info_sentimentos = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            info_sentimentos = json.load(f)
    else:
        st.warning(f"Arquivo JSON de sentimentos não encontrado em: {json_path}")
    return df_bruto, info_sentimentos

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

def attach_sentiment_on_comments(df, info):
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
    fig, ax = plt.subplots(figsize=(6.2, 4))
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
# MENU LATERAL E CARREGAMENTO
# -----------------------------
st.sidebar.title("Fonte de Dados")
fonte_selecionada = st.sidebar.selectbox("Selecione a fonte de dados:", list(PATHS.keys()))

st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Início", "Comentários", "Dashboard"], index=0)

# Carregamento e processamento inicial dos dados
csv_path = PATHS[fonte_selecionada]["csv"]
json_path = PATHS[fonte_selecionada]["json"]
dataFrame_bruto, info_sentimentos = load_data(csv_path, json_path)

comment_col = _detect_comment_col(dataFrame_bruto)
dataFrame_processado = attach_sentiment_on_comments(dataFrame_bruto, info_sentimentos)

# Normalizações de colunas
if "Data" in dataFrame_processado.columns:
    try: dataFrame_processado["Data"] = pd.to_datetime(dataFrame_processado["Data"], errors="coerce", dayfirst=True)
    except Exception: dataFrame_processado["Data"] = pd.to_datetime(dataFrame_processado["Data"], errors="coerce")
dataFrame_processado["sentiment"] = dataFrame_processado["sentiment"].map(lambda x: x if x in VALID_SENTIMENTS else None)

# -----------------------------
# FILTRO GLOBAL DE DATA
# -----------------------------
st.sidebar.title("Filtros")

# Verifica se a coluna de data existe para criar o filtro
if "Data" not in dataFrame_processado.columns:
    st.sidebar.warning("Coluna 'Data' não encontrada para aplicar filtro de período.")
else:
    # Remove linhas onde a data é inválida para não quebrar o cálculo de min/max
    datas_validas = dataFrame_processado["Data"].dropna()
    
    if not datas_validas.empty:
        min_date_geral = datas_validas.min().date()
        max_date_geral = datas_validas.max().date()

        # --- LÓGICA CORRIGIDA ---
        # Seletor de Data de Início: pode ir do início ao fim do período total.
        start_date = st.sidebar.date_input(
            "Data de Início:",
            value=min_date_geral,
            min_value=min_date_geral,
            max_value=max_date_geral, # O máximo é a data final geral
            format="DD/MM/YYYY"
        )

        # Seletor de Data de Fim: o valor mínimo é dinâmico, baseado na data de início.
        end_date = st.sidebar.date_input(
            "Data de Fim:",
            value=max_date_geral,
            min_value=start_date, # <<< AQUI ESTÁ A CORREÇÃO PRINCIPAL
            max_value=max_date_geral,
            format="DD/MM/YYYY"
        )
        
        # A validação de erro não é mais necessária, pois a própria interface já impede um intervalo inválido.
        # Filtra o DataFrame principal com base nas datas selecionadas.
        dataFrame_processado = dataFrame_processado[
            dataFrame_processado["Data"].dt.date.between(start_date, end_date)
        ]
    else:
        st.sidebar.info("Não há datas válidas nos dados para filtrar.")

# =========================================================================================
# RENDERIZAÇÃO DAS PÁGINAS
# =========================================================================================

# -----------------------------
# PÁGINA: INÍCIO
# -----------------------------
if page == "Início":
    st.title(f"Visão Geral: {fonte_selecionada}")
    st.markdown("---")
    if "Nota" not in dataFrame_processado.columns:
        st.warning("Coluna 'Nota' não encontrada.")
        st.stop()

    notas = pd.to_numeric(dataFrame_processado["Nota"], errors="coerce").dropna()
    notas = notas[notas.between(1, 5)]

    if notas.empty:
        st.info("Nenhuma avaliação com nota válida (1 a 5) encontrada para o período selecionado.")
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
    if "Assunto" in dataFrame_processado.columns: cols.append("Assunto")
    view = dataFrame_processado[cols + ["sentiment"]].copy()

    if "Assunto" not in view.columns:
        st.warning("Coluna 'Assunto' não foi encontrada.")
        escolha_assunto = "Todos"
    else:
        opcoes_assunto = sorted(view["Assunto"].dropna().astype(str).unique().tolist())
        escolha_assunto = st.selectbox("Filtrar por Assunto:", ["Todos"] + opcoes_assunto, index=0)

    if escolha_assunto != "Todos":
        view = view[view["Assunto"] == escolha_assunto]

    sentiment_counts = view["sentiment"].value_counts()
    total_no_filtro = int(sentiment_counts.sum())
    sent_map = {"Críticas": "criticism", "Sugestões": "suggestion", "Elogios": "positive feedback", "Não Pertinentes": "not pertinent"}
    
    cols = st.columns(4)
    for col, label in zip(cols, sent_map.keys()):
        key = sent_map[label]
        count = sentiment_counts.get(key, 0)
        if total_no_filtro > 0:
            percentage = (count / total_no_filtro) * 100
            delta_text = f"{percentage:.1f}% do total"
        else:
            delta_text = "N/A"
        col.metric(label, count, delta=delta_text, delta_color="off")
    
    st.markdown("---")

    user_num = 0
    for _, row in view.iterrows():
        txt = _norm_text(row[comment_col])
        if not txt: continue
        user_num +=1
        sty = style_for(row.get("sentiment"))
        nota_html = nota_span_html(row.get("Nota")) if "Nota" in view.columns else ""
        id_str = fmt_id(row.get("ID")) if "ID" in view.columns else str(user_num)
        st.markdown(f"""<div class="comment-card" style="background:{sty['bg']}; border-left-color:{sty['edge']};"><div class="header-row"><div class="meta">Usuário {escape(id_str)}</div>{nota_html}</div><div class="comment-text">{escape(txt)}</div></div>""", unsafe_allow_html=True)
    
    if user_num == 0: st.info("Nenhum comentário encontrado para os filtros selecionados.")

# -----------------------------
# PÁGINA: DASHBOARD
# -----------------------------
else:
    st.title(f"Dashboard por Assunto: {fonte_selecionada}")

    def pct_vector(df_subset):
        counts = df_subset["sentiment"].value_counts().reindex(SENT_ORDER, fill_value=0)
        total = int(counts.sum())
        percs = (counts / total * 100) if total > 0 else pd.Series([0]*len(SENT_ORDER), index=SENT_ORDER, dtype=float)
        return counts, percs.round(1), total

    if "Assunto" not in dataFrame_processado.columns:
        st.error("ERRO: A coluna 'Assunto' é necessária para este dashboard, mas não foi encontrada no arquivo CSV.")
        st.stop()
    
    todos_os_assuntos = sorted(dataFrame_processado['Assunto'].dropna().unique().tolist())
    df_calculo = dataFrame_processado.dropna(subset=['Assunto', 'sentiment']).copy()
    assuntos_com_score = []
    if not df_calculo.empty:
        total_comentarios_validos = len(df_calculo)
        counts_por_assunto = df_calculo['Assunto'].value_counts()
        criticas_por_assunto = df_calculo[df_calculo['sentiment'] == 'criticism']['Assunto'].value_counts()
        stats_assuntos = pd.DataFrame({'total_comentarios': counts_por_assunto, 'comentarios_criticos': criticas_por_assunto}).fillna(0)
        stats_assuntos['taxa_critica'] = stats_assuntos['comentarios_criticos'] / stats_assuntos['total_comentarios']
        stats_assuntos['prevalencia'] = stats_assuntos['total_comentarios'] / total_comentarios_validos
        stats_assuntos['score_critico'] = stats_assuntos['prevalencia'] * stats_assuntos['taxa_critica']
        stats_assuntos = stats_assuntos.sort_values(by='score_critico', ascending=False)
        assuntos_com_score = stats_assuntos.index.tolist()

    assuntos_sem_score = [assunto for assunto in todos_os_assuntos if assunto not in assuntos_com_score]
    opcoes_assunto = assuntos_com_score + assuntos_sem_score

    st.info("Os assuntos estão ordenados por um score de relevância crítica. Assuntos sem dados de sentimento aparecem no final.")
    escolha_assunto = st.selectbox("Selecione um Assunto para Análise:", ["Todos"] + opcoes_assunto, index=0)
    st.markdown("---")

    df_filtrado = dataFrame_processado.copy()
    if escolha_assunto != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Assunto"] == escolha_assunto]

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Distribuição de Sentimentos")
        df_sentimentos = df_filtrado[df_filtrado["sentiment"].notna()]
        counts, percs, total_sentimentos = pct_vector(df_sentimentos)
        if total_sentimentos == 0:
            st.info("Não há comentários com análise de sentimento para o filtro selecionado.")
        else:
            st.metric("Total de Comentários Analisados", f"{total_sentimentos:,}".replace(",", "."))
            fig_sent = plot_perc_barh(percs, f"Sentimentos para: {escolha_assunto}", "% de comentários")
            st.pyplot(fig_sent); plt.close(fig_sent)

    with col2:
        st.subheader("Distribuição de Notas")
        if "Nota" not in df_filtrado.columns:
            st.warning("Coluna 'Nota' não encontrada nos dados.")
        else:
            notas = pd.to_numeric(df_filtrado["Nota"], errors="coerce").dropna()
            notas = notas[notas.between(1, 5)]
            if notas.empty:
                st.info("Não há avaliações com nota (1-5) para o filtro selecionado.")
            else:
                total_avaliacoes = len(notas)
                nota_media = notas.mean()
                st.metric("Total de Avaliações com Nota", f"{total_avaliacoes:,}".replace(",", "."), delta=f"Média: {nota_media:.2f} ⭐", delta_color="off")
                nota_counts = notas.value_counts().reindex([5, 4, 3, 2, 1], fill_value=0)
                nota_percs = (nota_counts / total_avaliacoes * 100)
                fig_notas = plot_nota_distribution(nota_percs)
                st.pyplot(fig_notas); plt.close(fig_notas)