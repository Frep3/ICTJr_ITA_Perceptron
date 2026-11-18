"""Suaviza trajetórias e estima velocidade e aceleração por Savitzky–Golay."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_LIMPOS = Path("./dataset/cleaned")
PASTA_AUDITED = Path("./dataset/audited")
PASTA_SAIDA = Path("./dataset/processed")

WINDOW_LENGTH = 15
POLY_ORDER = 3

assert WINDOW_LENGTH % 2 == 1, "a janela precisa ser ímpar"
assert POLY_ORDER > 2, "para deriv=2 é preciso polyorder >= 3"


# ============================================================
# SAVITZKY-GOLAY
# ============================================================


def derivar_ensaio(d, window=WINDOW_LENGTH, poly=POLY_ORDER):

    d = d.reset_index(drop=True).copy()

    # Verifica se há quadros suficientes
    if len(d) < window:
        raise ValueError(
            f"Ensaio com {len(d)} quadros, mas a janela exige pelo menos {window}."
        )

    # Intervalo temporal
    dt = np.median(np.diff(d["t"].values))

    if dt <= 0:
        raise ValueError("O tempo não é estritamente crescente.")

    # --------------------------------------------------------
    # Velocidade vertical
    # --------------------------------------------------------

    d["v"] = savgol_filter(d["y"].values, window, poly, deriv=1, delta=dt)

    # --------------------------------------------------------
    # Aceleração vertical
    # --------------------------------------------------------

    d["a"] = savgol_filter(d["y"].values, window, poly, deriv=2, delta=dt)

    # --------------------------------------------------------
    # Posição horizontal suavizada
    # --------------------------------------------------------

    if "x" in d.columns:
        xh = d["x"].interpolate(limit_direction="both").fillna(0.0).values

        d["x_h"] = savgol_filter(xh, window, poly)

    else:
        d["x_h"] = 0.0

    # --------------------------------------------------------
    # Remove bordas menos confiáveis
    # --------------------------------------------------------

    borda = window // 2 + 2

    if len(d) <= 2 * borda:
        raise ValueError("Ensaio ficou pequeno demais após remover as bordas.")

    return d.iloc[borda:-borda].reset_index(drop=True)


# ============================================================
# PROCESSAMENTO DOS ARQUIVOS
# ============================================================

arquivos = ["cola", "gesso", "vazia"]

derivados = {}


for nome in arquivos:
    print()
    print("=" * 60)
    print(f"Processando: {nome}")
    print("=" * 60)

    caminho_auditado = PASTA_AUDITED / f"{nome}_auditado.csv"

    caminho_limpo = PASTA_LIMPOS / f"{nome}_limpo.csv"

    # --------------------------------------------------------
    # Lê auditoria
    # --------------------------------------------------------

    if not caminho_auditado.is_file() or not caminho_limpo.is_file():
        continue

    auditado = pd.read_csv(caminho_auditado)

    # --------------------------------------------------------
    # Seleciona somente os aprovados
    # --------------------------------------------------------

    aprovados = auditado[auditado["aprovado"] == True]

    trials_aprovados = aprovados["trial"].astype(str).tolist()

    print(f"Trials aprovados: {trials_aprovados}")

    # --------------------------------------------------------
    # Lê dados limpos
    # --------------------------------------------------------

    dados = pd.read_csv(caminho_limpo)

    # --------------------------------------------------------
    # Processa cada trial aprovado
    # --------------------------------------------------------

    resultados = []

    for trial in trials_aprovados:
        d = dados[dados["trial"].astype(str) == trial].copy()

        if d.empty:
            print(f"  AVISO: {trial} não encontrado no arquivo limpo.")
            continue
        # Evita manter uma coluna de velocidade antiga; ela será recalculada.
        if "v" in d.columns:
            d = d.drop(columns="v")
        try:
            derivado = derivar_ensaio(d)

            resultados.append(derivado)

            print(f"  {trial}: OK (n={len(derivado)})")

        except ValueError as erro:
            print(f"  {trial}: ERRO - {erro}")

    # --------------------------------------------------------
    # Junta todos os trials aprovados
    # --------------------------------------------------------

    if resultados:
        df_final = pd.concat(resultados, ignore_index=True)

        # ----------------------------------------------------
        # Salva
        # ----------------------------------------------------

        caminho_saida = PASTA_SAIDA / f"{nome}_derivado.csv"

        df_final.to_csv(caminho_saida, index=False)

        derivados[nome] = df_final

        print()
        print(f"Salvo em: {caminho_saida}")

    else:
        print("Nenhum trial aprovado para processar.")


# ============================================================
# RESUMO
# ============================================================

print()
print("=" * 60)
print("RESUMO DA MODELAGEM")
print("=" * 60)

for nome, d in derivados.items():
    print(
        f"{nome:8s} "
        f"n={len(d):4d}  "
        f"a ∈ [{d['a'].min():6.2f}, "
        f"{d['a'].max():6.2f}]  "
        f"σ_a={d['a'].std():.3f} m/s²"
    )
