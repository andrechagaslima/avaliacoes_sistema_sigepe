import pandas as pd

df = pd.read_csv(
    "avaliacoes_sougov_preprocessado.csv",
    encoding="utf-8",
    sep=","
)

# Descobre as colunas de id e comentário
id_col = "ID" if "ID" in df.columns else "id"
comment_col = (
    "comments" if "comments" in df.columns
    else ("Comentário" if "Comentário" in df.columns else "Comentários")
)

# Remove linhas sem comentário (opcional)
df = df.dropna(subset=[comment_col])

k = min(20, len(df))
amostra = df.sample(n=k, random_state=None)[[id_col, comment_col]]

# Imprime no formato desejado
for _, row in amostra.iterrows():
    print(f"ID: {row[id_col]} | Comentário: {row[comment_col]}")
