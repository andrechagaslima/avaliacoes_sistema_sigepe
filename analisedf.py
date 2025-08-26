import pandas as pd

df = pd.read_csv(
    "avaliacoes_sigepe_preprocessado.csv",
    encoding="utf-8",   
    sep=","             
)

comment_col = "Comentário"

k = min(20, len(df))
indices_aleatorios = df.sample(n=k, random_state=None).index

# mostra o índice e o comentário correspondente
print(df.loc[indices_aleatorios, [comment_col]].to_string())