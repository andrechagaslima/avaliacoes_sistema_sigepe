# app.py — Comentários + Dashboard (% por sentimento em PT-BR com cores por categoria)
import os, json, re, unicodedata
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from html import escape

# -----------------------------
# CONFIG / TEMA
# -----------------------------
st.set_page_config(page_title="Sentimentos — Comentários & Dashboard", layout="wide")
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
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CAMINHOS PADRÃO
# -----------------------------
DEFAULT_CSV_PATH  = "./data/dataFrame.csv"
DEFAULT_JSON_PATH = "./sentiment_analysis/resources/outLLM/sentiment_analysis.json"

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
    "positive feedback": "#16a34a",  # verde
    "criticism": "#dc2626",          # vermelho
    "suggestion": "#f59e0b",         # amarelo
    "not pertinent": "#6b7280",      # cinza
}

# -----------------------------
# FUNÇÕES AUXILIARES (dados)
# -----------------------------
def load_csv():
    if os.path.exists(DEFAULT_CSV_PATH):
        return pd.read_csv(DEFAULT_CSV_PATH, encoding="utf-8")
    upl = st.file_uploader("Envie o CSV", type=["csv"])
    if upl is None: st.stop()
    return pd.read_csv(upl, encoding="utf-8")

def load_json():
    if os.path.exists(DEFAULT_JSON_PATH):
        with open(DEFAULT_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    upl = st.file_uploader("Envie o JSON de predições (y_pred_text)", type=["json"])
    if upl is None:
        st.error("Forneça o JSON com y_pred_text para colorir pelos rótulos."); st.stop()
    return json.load(upl)

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
    """Aplica y_pred_text APENAS nas linhas com comentário (na ordem do DF)."""
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
        if len(preds) > len(idx):
            st.info(f"{len(preds)-len(idx)} predições a mais no JSON foram ignoradas.")
        elif len(preds) < len(idx):
            st.info(f"{len(idx)-len(preds)} comentários ficaram sem predição (JSON curto).")
    return df

# -----------------------------
# FUNÇÕES AUXILIARES (UI/estilo)
# -----------------------------
def style_for(sentiment):
    palette = {
        "positive feedback": {"bg": "#d9f4e3", "edge": "#16a34a"},  # verde
        "criticism":         {"bg": "#ffd9d9", "edge": "#dc2626"},  # vermelho
        "suggestion":        {"bg": "#fff0b3", "edge": "#f59e0b"},  # amarelo
        "not pertinent":     {"bg": "#ececec", "edge": "#6b7280"},  # cinza
        None:                {"bg": "#f7f7f7", "edge": "#9ca3af"},
        pd.NA:               {"bg": "#f7f7f7", "edge": "#9ca3af"},
    }
    return palette.get(sentiment, palette[None])

def nota_span_html(v):
    try:
        x = float(v)
    except Exception:
        return ""
    show = str(int(x)) if abs(x - int(x)) < 1e-9 else f"{x:.1f}"
    return f"""
<span class="badge">
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path fill="#f59e0b" d="M12 .587l3.668 7.431L24 9.748l-6 5.848L19.335 24 12 19.897 4.665 24 6 15.596 0 9.748l8.332-1.73z"/>
  </svg>
  <strong>{escape(show)}</strong>
</span>
"""

def fmt_id(v):
    if v is None or (isinstance(v, float) and pd.isna(v)): return None
    try:
        f = float(v);  return str(int(f)) if abs(f - int(f)) < 1e-9 else str(v)
    except Exception:
        return str(v)

# -------- Plot auxiliar (em PT + cores por sentimento) --------
def plot_perc_barh(percs, title, xlabel):
    labels_pt = [PT_LABEL[s] for s in SENT_ORDER]
    values = [float(percs[s]) for s in SENT_ORDER]
    colors = [COLOR_MAP[s] for s in SENT_ORDER]

    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.barh(labels_pt, values, color=colors)
    for i, v in enumerate(values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, max(100, (max(values) if values else 0) + 5))
    return fig

# -----------------------------
# CARREGAMENTO DE DADOS
# -----------------------------
df = load_csv()
info = load_json()

comment_col = _detect_comment_col(df)

# cópia p/ comentários e dashboard com rótulos aplicados
df_all = df.copy()
df_all = attach_sentiment_on_comments(df_all, info, id_col="ID")

# normalizações leves p/ dashboard
if "Data" in df_all.columns:
    try:
        df_all["Data"] = pd.to_datetime(df_all["Data"], errors="coerce", dayfirst=True)
    except Exception:
        df_all["Data"] = pd.to_datetime(df_all["Data"], errors="coerce")

df_all["sentiment"] = df_all["sentiment"].map(lambda x: x if x in VALID_SENTIMENTS else None)

# -----------------------------
# MENU LATERAL
# -----------------------------
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Comentários", "Dashboard"], index=0)

# -----------------------------
# PÁGINA: COMENTÁRIOS
# -----------------------------
if page == "Comentários":
    st.title("Análise de Sentimento dos Comentários")

    cols = [comment_col]
    if "Nota" in df_all.columns: cols.append("Nota")
    if "ID" in df_all.columns: cols.append("ID")
    view = df_all[cols + ["sentiment"]].copy()

    # Filtros (sentimento + FAIXA de nota [mín, máx])
    c1, c2 = st.columns([2, 1])
    with c1:
        label_to_key = {
            "Todos": None,
            "Negativo": "criticism",
            "Sugestão": "suggestion",
            "Positivo": "positive feedback",
            "Não pertinente": "not pertinent",
        }
        choice = st.radio("Filtrar por sentimento:", list(label_to_key.keys()), horizontal=True, index=0)
        chosen_key = label_to_key[choice]
    with c2:
        nota_range = None
        if "Nota" in view.columns:
            view["__nota_num__"] = pd.to_numeric(view["Nota"], errors="coerce")
            s = view["__nota_num__"].dropna()
            min_avail = int(max(1, s.min())) if not s.empty else 1
            max_avail = int(min(5, s.max())) if not s.empty else 5
            nota_range = st.slider("Faixa de nota", 1, 5, (min_avail, max_avail), 1)

    if chosen_key is not None:
        view = view[view["sentiment"] == chosen_key]
    if "Nota" in view.columns and nota_range is not None:
        lo, hi = nota_range
        view = view[(view["__nota_num__"].isna()) | ((view["__nota_num__"] >= lo) & (view["__nota_num__"] <= hi))]

    # Render dos cards
    user_num = 1
    for _, row in view.iterrows():
        txt = _norm_text(row[comment_col])
        if not txt: continue
        sty = style_for(row.get("sentiment"))
        nota_html = nota_span_html(row.get("Nota")) if "Nota" in view.columns else ""
        id_str = fmt_id(row.get("ID")) if "ID" in view.columns else None
        left = f"Usuário {id_str}"

        st.markdown(
            f"""
<div class="comment-card" style="background:{sty['bg']}; border-left-color:{sty['edge']};">
  <div class="header-row">
    <div class="meta">{escape(left)}</div>
    {nota_html}
  </div>
  <div class="comment-text">{escape(txt)}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        user_num += 1

    if user_num == 1:
        st.info("Nenhum comentário para o filtro selecionado.")

# -----------------------------
# PÁGINA: DASHBOARD (seletores single)
# -----------------------------
else:
    st.title("Dashboard")

    # Função: calcula vetor de % para um subconjunto
    def pct_vector(df_subset):
        counts = df_subset["sentiment"].value_counts().reindex(SENT_ORDER, fill_value=0)
        total = int(counts.sum())
        if total == 0:
            percs = pd.Series([0,0,0,0], index=SENT_ORDER, dtype=float)
        else:
            percs = (counts / total * 100)
        return counts, percs.round(1), total

    # Escolha do eixo de filtro (uma coisa por vez)
    base = st.radio(
        "Filtrar por:",
        ["Todos", "Assunto", "Funcionalidade", "Nota"],
        horizontal=True, index=0
    )

    df_base = df_all[df_all["sentiment"].notna()].copy()

    # ----------------- TODOS -----------------
    if base == "Todos":
        st.subheader("Distribuição geral (%)")
        counts, percs, total = pct_vector(df_base)

        # ==== KPIs ====
        # Nota: média, desvio, mediana (apenas onde há valor numérico)
        notas = pd.to_numeric(df_base.get("Nota", pd.Series(index=df_base.index)), errors="coerce")
        n_notas = int(notas.notna().sum())
        nota_media = float(notas.mean()) if n_notas > 0 else None
        nota_std   = float(notas.std(ddof=1)) if n_notas > 1 else None
        nota_med   = float(notas.median()) if n_notas > 0 else None

        # Nova métrica: Razão Elogio/Crítica (E/C)
        elogios = int(counts["positive feedback"])
        criticas = int(counts["criticism"])
        if criticas == 0:
            ratio_ec = "∞" if elogios > 0 else "-"
        else:
            ratio_ec = f"{elogios / criticas:.2f}"

        # Índice de Sentimento = %Elogio − %Crítica (em p.p.)
        idx_sent = float(percs["positive feedback"] - percs["criticism"]) if total > 0 else 0.0

        # KPIs (duas linhas)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rotulado", f"{total:,}".replace(",", "."))
        c2.metric("Com nota", f"{n_notas:,}".replace(",", "."))
        c3.metric("Razão Elogio/Crítica (E/C)", ratio_ec)

        c4, c5, c6 = st.columns(3)
        c4.metric("Nota média", "-" if nota_media is None else f"{nota_media:.2f}")
        c5.metric("Desvio-padrão da nota", "-" if nota_std is None else f"{nota_std:.2f}")
        c6.metric("Mediana da nota", "-" if nota_med is None else f"{nota_med:.2f}")

        # Tabela resumo
        st.dataframe(
            pd.DataFrame({
                "Sentimento": [PT_LABEL[s] for s in SENT_ORDER],
                "Quantidade": [int(counts[s]) for s in SENT_ORDER],
                "Percentual (%)": [float(percs[s]) for s in SENT_ORDER],
            }),
            use_container_width=True, hide_index=True
        )

        # Gráfico
        fig = plot_perc_barh(percs, "Distribuição geral de sentimentos", "% do total rotulado")
        st.pyplot(fig); plt.close(fig)


    # --------------- ASSUNTO -----------------
    elif base == "Assunto":
        if "Assunto" not in df_base.columns:
            st.warning("Coluna 'Assunto' não encontrada no CSV."); st.stop()
        st.subheader("Distribuição (%) — Filtro por Assunto (único)")
        opts = sorted(df_base["Assunto"].dropna().astype(str).unique().tolist())
        escolha = st.selectbox("Assunto:", ["Todos"] + opts, index=0)
        if escolha != "Todos":
            df_base = df_base[df_base["Assunto"].astype(str) == escolha]

        counts, percs, total = pct_vector(df_base)
        if total == 0:
            st.info("Sem dados para o filtro selecionado."); st.stop()

        # KPIs básicos no filtro
        notas = pd.to_numeric(df_base.get("Nota", pd.Series(index=df_base.index)), errors="coerce")
        n_notas = int(notas.notna().sum())
        nota_media = float(notas.mean()) if n_notas > 0 else None

        c1, c2 = st.columns(2)
        c1.metric("Total no filtro", f"{total:,}".replace(",", "."))
        c2.metric("Nota média no filtro", "-" if nota_media is None else f"{nota_media:.2f}")

        st.dataframe(
            pd.DataFrame({
                "Sentimento": [PT_LABEL[s] for s in SENT_ORDER],
                "Quantidade": [int(counts[s]) for s in SENT_ORDER],
                "Percentual (%)": [float(percs[s]) for s in SENT_ORDER],
            }),
            use_container_width=True, hide_index=True
        )
        fig = plot_perc_barh(percs, f"Distribuição por sentimento — Assunto: {escolha}", "% no subconjunto")
        st.pyplot(fig); plt.close(fig)

    # ----------- FUNCIONALIDADE --------------
    elif base == "Funcionalidade":
        if "Funcionalidade" not in df_base.columns:
            st.warning("Coluna 'Funcionalidade' não encontrada no CSV."); st.stop()
        st.subheader("Distribuição (%) — Filtro por Funcionalidade (único)")
        opts = sorted(df_base["Funcionalidade"].dropna().astype(str).unique().tolist())
        escolha = st.selectbox("Funcionalidade:", ["Todos"] + opts, index=0)
        if escolha != "Todos":
            df_base = df_base[df_base["Funcionalidade"].astype(str) == escolha]

        counts, percs, total = pct_vector(df_base)
        if total == 0:
            st.info("Sem dados para o filtro selecionado."); st.stop()

        notas = pd.to_numeric(df_base.get("Nota", pd.Series(index=df_base.index)), errors="coerce")
        n_notas = int(notas.notna().sum())
        nota_media = float(notas.mean()) if n_notas > 0 else None

        c1, c2 = st.columns(2)
        c1.metric("Total no filtro", f"{total:,}".replace(",", "."))
        c2.metric("Nota média no filtro", "-" if nota_media is None else f"{nota_media:.2f}")

        st.dataframe(
            pd.DataFrame({
                "Sentimento": [PT_LABEL[s] for s in SENT_ORDER],
                "Quantidade": [int(counts[s]) for s in SENT_ORDER],
                "Percentual (%)": [float(percs[s]) for s in SENT_ORDER],
            }),
            use_container_width=True, hide_index=True
        )
        fig = plot_perc_barh(percs, f"Distribuição por sentimento — Funcionalidade: {escolha}", "% no subconjunto")
        st.pyplot(fig); plt.close(fig)

    # -------------------- NOTA ----------------
    else:  # base == "Nota"
        if "Nota" not in df_base.columns:
            st.warning("Coluna 'Nota' não encontrada no CSV."); st.stop()
        st.subheader("Distribuição (%) — Filtro por Nota (única)")
        df_base["__nota_num__"] = pd.to_numeric(df_base["Nota"], errors="coerce")
        notas = sorted([int(n) for n in df_base["__nota_num__"].dropna().unique() if 1 <= n <= 5])
        escolha = st.selectbox("Nota:", ["Todas"] + [str(n) for n in notas], index=0)
        if escolha != "Todas":
            df_base = df_base[df_base["__nota_num__"] == int(escolha)]

        counts, percs, total = pct_vector(df_base)
        if total == 0:
            st.info("Sem dados para o filtro selecionado."); st.stop()

        (st.columns(1)[0]).metric("Total no filtro", f"{total:,}".replace(",", "."))

        st.dataframe(
            pd.DataFrame({
                "Sentimento": [PT_LABEL[s] for s in SENT_ORDER],
                "Quantidade": [int(counts[s]) for s in SENT_ORDER],
                "Percentual (%)": [float(percs[s]) for s in SENT_ORDER],
            }),
            use_container_width=True, hide_index=True
        )
        fig = plot_perc_barh(percs, f"Distribuição por sentimento — Nota: {escolha}", "% no subconjunto")
        st.pyplot(fig); plt.close(fig)
