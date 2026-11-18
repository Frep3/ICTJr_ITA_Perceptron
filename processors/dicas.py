"""Cria atributos derivados (quadrado e inverso) para os dados de cada grupo."""

import math
from pathlib import Path

import numpy as np
import pandas as pd

ENTRADA = Path("./dataset/grupos")
SAIDA = Path("./dataset/dicas")

for pasta in ENTRADA.iterdir():
    if not pasta.is_dir():
        continue
    destino_pasta = SAIDA / pasta.name
    destino_pasta.mkdir(parents=True, exist_ok=True)

    for arquivo in pasta.glob("*.csv"):
        df = pd.read_csv(arquivo)
        for coluna in df.columns:
            if (
                np.issubdtype(df[coluna].to_numpy().dtype, np.number)
                and coluna != "a"
                and coluna != "x"
            ):
                df = df.assign(**{coluna + "**2": df[coluna] ** 2})
                df = df.assign(**{"1/" + coluna: 1 / df[coluna]})
                # Inversos de zero geram infinito; remova-os antes de salvar.
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)
        df.to_csv(destino_pasta / arquivo.name, index=False)
