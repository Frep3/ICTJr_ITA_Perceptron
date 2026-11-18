"""Divide cada ensaio cronologicamente em treino, validação e teste."""

import math
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_ENTRADA = Path("./dataset/processed")

PASTA_SAIDA_TRAIN = Path("./dataset/separated/train")
PASTA_SAIDA_VALIDATION = Path("./dataset/separated/validation")
PASTA_SAIDA_TEST = Path("./dataset/separated/test")

for pasta in [PASTA_SAIDA_TRAIN, PASTA_SAIDA_VALIDATION, PASTA_SAIDA_TEST]:
    pasta.mkdir(parents=True, exist_ok=True)


# ============================================================
# SEPARAÇÃO POR GRUPO E TEMPO RELATIVO
# ============================================================


def separar_por_grupo_e_tempo(df):
    df = df.copy()

    # Extrai apenas os números da coluna 'trial'
    df["trial_num"] = df["trial"].astype(str).str.extract(r"(\d+)")[0].astype(int)

    # Grupo 1: trials <= 3 (ex: Vazia)
    # Grupo 2: trials > 3 (ex: Cola, Gesso, Vazia)
    df["grupo"] = np.where(df["trial_num"] <= 3, 1, 2)

    train_list = []
    val_list = []
    test_list = []

    # Processa apenas os grupos presentes no arquivo
    grupos_presentes = sorted(df["grupo"].unique())

    for g_id in grupos_presentes:
        df_grupo = df[df["grupo"] == g_id].copy()

        g_train_parts = []
        g_val_parts = []
        g_test_parts = []

        # Processa cada trial individualmente
        for trial in df_grupo["trial"].unique():
            d_trial = df_grupo[df_grupo["trial"] == trial].copy()
            d_trial = d_trial.sort_values("t").reset_index(drop=True)

            # Tempo relativo para funcionar com offsets (ex: t + 4) e durações curtas (< 2s)
            t_min = d_trial["t"].min()
            t_max = d_trial["t"].max()
            tempo_total = t_max - t_min

            if tempo_total > 0:
                limite_train = t_min + 0.70 * tempo_total
                limite_val = t_min + 0.85 * tempo_total

                train_part = d_trial[d_trial["t"] <= limite_train].copy()
                val_part = d_trial[
                    (d_trial["t"] > limite_train) & (d_trial["t"] <= limite_val)
                ].copy()
                test_part = d_trial[d_trial["t"] > limite_val].copy()
            else:
                train_part = d_trial.copy()
                val_part = pd.DataFrame()
                test_part = pd.DataFrame()

            # Adiciona as partes se contiverem dados (sem restrição arbitrária de len >= 10)
            if not train_part.empty:
                g_train_parts.append(train_part)
            if not val_part.empty:
                g_val_parts.append(val_part)
            if not test_part.empty:
                g_test_parts.append(test_part)

        # Concatena e ordena por tempo dentro de cada grupo
        if g_train_parts:
            g_train = pd.concat(g_train_parts, ignore_index=True).sort_values(
                ["trial_num", "t"]
            )
            train_list.append(g_train)

        if g_val_parts:
            g_val = pd.concat(g_val_parts, ignore_index=True).sort_values(
                ["trial_num", "t"]
            )
            val_list.append(g_val)

        if g_test_parts:
            g_test = pd.concat(g_test_parts, ignore_index=True).sort_values(
                ["trial_num", "t"]
            )
            test_list.append(g_test)

    # Junta os grupos garantindo a ordem (Grupo 1 primeiro se existir, depois Grupo 2)
    train = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    validation = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    # Limpa colunas auxiliares
    cols_aux = ["trial_num", "grupo"]
    train = train.drop(
        columns=[c for c in cols_aux if c in train.columns], errors="ignore"
    )
    validation = validation.drop(
        columns=[c for c in cols_aux if c in validation.columns], errors="ignore"
    )
    test = test.drop(
        columns=[c for c in cols_aux if c in test.columns], errors="ignore"
    )

    return train, validation, test


# ============================================================
# PROCESSAR UM ARQUIVO
# ============================================================


def processar_arquivo(caminho):
    df = pd.read_csv(caminho)

    train, validation, test = separar_por_grupo_e_tempo(df)

    nome = caminho.stem.replace("_derivado", "")

    train.to_csv(PASTA_SAIDA_TRAIN / f"{nome}_train.csv", index=False)
    validation.to_csv(PASTA_SAIDA_VALIDATION / f"{nome}_validation.csv", index=False)
    test.to_csv(PASTA_SAIDA_TEST / f"{nome}_test.csv", index=False)

    print(f"\n{nome}")
    print(f"  Train:      {len(train)} linhas")
    print(f"  Validation: {len(validation)} linhas")
    print(f"  Test:       {len(test)} linhas")


# ============================================================
# EXECUÇÃO
# ============================================================

arquivos = [
    "cola_derivado.csv",
    "gesso_derivado.csv",
    "vazia_derivado.csv",
]

for arquivo in arquivos:
    caminho = PASTA_ENTRADA / arquivo

    if not caminho.exists():
        print(f"Arquivo não encontrado: {caminho}")
        continue

    print("=" * 60)
    print(f"Processando: {arquivo}")

    processar_arquivo(caminho)

print("\nSeparação concluída.")
