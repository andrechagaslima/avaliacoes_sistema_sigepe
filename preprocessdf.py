import pandas as pd

# Nome exato da coluna no seu CSV
COL_COMENTARIOS = "Comentário"  # troque aqui se for diferente (ex.: "comentários")

df = pd.read_csv(
    "avaliacoes_sistema_sougov.csv",
    encoding="utf-8",
    sep=","
)

# Remove colunas indesejadas (ignora se não existirem)
df = df.drop(columns=["Nome", "CPF", "Informações adicionais"], errors="ignore")

# Adiciona 'id' sequencial a partir de 1 como primeira coluna
df = df.reset_index(drop=True)
df.insert(0, "id", range(1, len(df) + 1))

# Renomeia a coluna indicada para 'comments'
df = df.rename(columns={COL_COMENTARIOS: "comments"})

df.to_csv("avaliacoes_sougov_preprocessado.csv", index=False)
