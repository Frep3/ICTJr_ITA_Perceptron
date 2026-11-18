"""Separa os arquivos temporais nos grupos de ensaios definidos pelo projeto."""

from pathlib import Path

import pandas as pd

# ============================================================
# CONFIGURAÇÕES DE PASTAS
# ============================================================

PASTA_ENTRADA_SEPARATED = Path("./dataset/separated")
PASTA_SAIDA_GRUPOS = Path("./dataset/grupos")


# ============================================================
# FUNÇÃO PARA SEPARAR EM DOIS GRUPOS
# ============================================================


def separar_em_dois_grupos(caminho_entrada, pasta_saida) -> None:
    df = pd.read_csv(caminho_entrada)

    # Garante que a pasta de saída para o tipo (train/validation/test) exista
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # Extrai o número do trial
    df["trial_num"] = df["trial"].astype(str).str.extract(r"(\d+)")[0].astype(int)

    # Separa os grupos
    grupo1 = df[df["trial_num"] <= 3].copy()
    grupo2 = df[df["trial_num"] > 3].copy()

    nome_base = caminho_entrada.stem  # Ex: "cola_train"

    # --------------------------------------------------------
    # GRUPO 1 (trials <= 3)
    # --------------------------------------------------------
    if not grupo1.empty:
        grupo1 = grupo1.sort_values(by=["trial_num", "t"]).reset_index(drop=True)
        grupo1 = grupo1.drop(columns=["trial_num"])

        path_g1 = pasta_saida / f"{nome_base}_grupo1.csv"
        grupo1.to_csv(path_g1, index=False)
        print(
            f"  Grupo 1 (trials <= 3): {len(grupo1)} linhas -> {path_g1.relative_to(PASTA_SAIDA_GRUPOS)}"
        )
    else:
        print(f"  Grupo 1 (trials <= 3): (Nenhum trial encontrado)")

    # --------------------------------------------------------
    # GRUPO 2 (trials > 3)
    # --------------------------------------------------------
    if not grupo2.empty:
        grupo2 = grupo2.sort_values(["trial_num", "t"]).reset_index(drop=True)
        grupo2 = grupo2.drop(columns=["trial_num"])

        path_g2 = pasta_saida / f"{nome_base}_grupo2.csv"
        grupo2.to_csv(path_g2, index=False)
        print(
            f"  Grupo 2 (trials > 3):  {len(grupo2)} linhas -> {path_g2.relative_to(PASTA_SAIDA_GRUPOS)}"
        )
    else:
        print(f"  Grupo 2 (trials > 3):  (Nenhum trial encontrado)")


# ============================================================
# EXECUÇÃO
# ============================================================

tipos = ["train", "validation", "test"]
materiais = ["cola", "gesso", "vazia"]

for tipo in tipos:
    pasta_origem = PASTA_ENTRADA_SEPARATED / tipo
    pasta_destino = PASTA_SAIDA_GRUPOS / tipo

    for mat in materiais:
        nome_arquivo = f"{mat}_{tipo}.csv"
        caminho_arquivo = pasta_origem / nome_arquivo

        if not caminho_arquivo.exists():
            print(f"\nArquivo não encontrado: {caminho_arquivo}")
            continue

        print("=" * 60)
        print(f"Processando: {tipo.upper()} -> {nome_arquivo}")
        separar_em_dois_grupos(caminho_arquivo, pasta_destino)

print("\nSeparação por grupos concluída.")
