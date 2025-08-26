import pandas as pd

df = pd.read_csv(
    "avaliacoes_sistema_sigepe.csv",
    encoding="utf-8",   
    sep=","             
)

df = df.drop(columns=["Nome", "CPF", "Informações adicionais"], errors="ignore")

df.to_csv("avaliacoes_sigepe_preprocessado.csv", index=False)
