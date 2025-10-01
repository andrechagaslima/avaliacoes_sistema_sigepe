import pandas as pd
import os

# --- DEFINIÇÕES GLOBAIS ---
# Nome do arquivo CSV pré-processado que será lido
NOME_ARQUIVO = "avaliacoes_sougov_preprocessado.csv"

# Nomes exatos das colunas que vamos utilizar
COLUNA_ID = "id"
COLUNA_COMENTARIOS = "comments"


def encontrar_id_pelo_numero_do_comentario_base_zero(numero_comentario: int):
    """
    Localiza o N-ésimo comentário não-vazio (base 0) em um arquivo CSV e retorna seu ID.

    A contagem começa em 0. O primeiro comentário é o de número 0, o segundo é o 1, etc.

    Args:
        numero_comentario: A posição do comentário a ser encontrado (começando em 0).

    Returns:
        Uma string contendo o ID correspondente ou uma mensagem de erro.
    """
    # ETAPA 1: Verificar se o arquivo de dados existe.
    if not os.path.exists(NOME_ARQUIVO):
        return f"ERRO: O arquivo '{NOME_ARQUIVO}' não foi encontrado."

    # ETAPA 2: Ler o arquivo CSV com tratamento de erros.
    try:
        df = pd.read_csv(NOME_ARQUIVO)
    except Exception as e:
        return f"ERRO: Não foi possível ler o arquivo CSV. Detalhes: {e}"

    # ETAPA 3: Filtrar o DataFrame para manter apenas as linhas que contêm comentários reais.
    # Primeiro, removemos linhas onde o comentário é Nulo/NaN.
    df_com_comentarios = df.dropna(subset=[COLUNA_COMENTARIOS])
    # Em seguida, removemos linhas onde o comentário é apenas uma string vazia ou espaços em branco.
    df_com_comentarios = df_com_comentarios[df_com_comentarios[COLUNA_COMENTARIOS].str.strip() != ""]

    # ETAPA 4: Validar o número fornecido pelo usuário.
    total_comentarios = len(df_com_comentarios)
    if total_comentarios == 0:
        return "AVISO: Nenhum comentário foi encontrado no arquivo."

    # A contagem válida vai de 0 até (total de comentários - 1).
    if not 0 <= numero_comentario < total_comentarios:
        return f"ERRO: Número inválido. Forneça um número entre 0 e {total_comentarios - 1}."

    # ETAPA 5: Localizar a linha e extrair o ID.
    # Como a entrada do usuário já é base 0, ela corresponde diretamente ao índice do .iloc.
    linha_desejada = df_com_comentarios.iloc[numero_comentario]
    
    id_encontrado = linha_desejada[COLUNA_ID]

    # ETAPA 6: Retornar o resultado formatado.
    return f"O ID para o comentário de número {numero_comentario} (base 0) é: {id_encontrado}"


# --- BLOCO PRINCIPAL PARA EXECUÇÃO INTERATIVA ---
# Este trecho só será executado quando você rodar o script diretamente.
if __name__ == "__main__":
    print("--- Ferramenta para Buscar ID de Comentário (Base 0) ---")
    
    # Loop infinito para permitir múltiplas buscas
    while True:
        # Pede a entrada do usuário
        input_usuario = input("\nDigite o número do comentário (começando em 0) ou 'sair' para terminar: ")
        
        if input_usuario.lower() == 'sair':
            print("Encerrando o programa.")
            break

        try:
            # Tenta converter a entrada para um número inteiro
            numero = int(input_usuario)
            
            # Chama a função principal com o número fornecido
            resultado = encontrar_id_pelo_numero_do_comentario_base_zero(numero)
            
            # Imprime o resultado de forma clara
            print("-" * 50)
            print(f"RESULTADO: {resultado}")
            print("-" * 50)

        except ValueError:
            # Se o usuário não digitar um número
            print("ERRO: Entrada inválida. Por favor, digite um número inteiro.")
        except Exception as e:
            # Para qualquer outro erro inesperado
            print(f"Ocorreu um erro inesperado: {e}")