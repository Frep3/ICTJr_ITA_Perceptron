"""Agrupa, normaliza e exporta os conjuntos com atributos derivados."""

from pathlib import Path

import numpy as np
import pandas as pd
from networkx import sigma

test = Path("./dataset/dicas/test")
train = Path("./dataset/dicas/train")
validation = Path("./dataset/dicas/validation")

tab1 = pd.DataFrame()
tab2 = pd.DataFrame()

arquivos1 = []
arquivos2 = []
for arquivo in train.glob("*.csv"):
    if arquivo.name.endswith("1.csv"):
        SPara = pd.read_csv(arquivo)
        arquivos1.append(SPara)
        print(f"Arquivo {arquivo.name} lido com {len(SPara)} linhas")
        print(f"heading:\n{SPara.head()}\n")
    if arquivo.name.endswith("2.csv"):
        CPara = pd.read_csv(arquivo)
        arquivos2.append(CPara)
        print(f"Arquivo {arquivo.name} lido com {len(CPara)} linhas")
        print("Shape:", CPara.shape)
        print("Colunas:", CPara.columns.tolist())
        print("Primeira linha:", CPara.iloc[0].to_dict())
        print(f"heading:\n{CPara.head()}\n")

for i, arquivos in enumerate(arquivos1):
    tab1 = pd.concat([tab1, arquivos], ignore_index=True)

for i, arquivos in enumerate(arquivos2):
    tab2 = pd.concat([tab2, arquivos], ignore_index=True)

tab1 = tab1.drop(columns=["trial"], errors="ignore")
tab2 = tab2.drop(columns=["trial"], errors="ignore")

mu1 = tab1.mean()
sigma1 = tab1.std().replace(0, np.nan)

mu2 = tab2.mean()
sigma2 = tab2.std().replace(0, np.nan)

for column in tab1.columns:
    tab1[column] = (tab1[column] - mu1[column]) / sigma1[column]

tab1.replace([np.inf, -np.inf], np.nan, inplace=True)
tab1.dropna(inplace=True)

tab1.to_csv("dataset/prepared/dicas/train_normal.csv", index=False)

for column in tab2.columns:
    tab2[column] = (tab2[column] - mu2[column]) / sigma2[column]

tab2.replace([np.inf, -np.inf], np.nan, inplace=True)
tab2.dropna(inplace=True)
tab2.to_csv("dataset/prepared/dicas/train_paraquedas.csv", index=False)

arquivos1.clear()
arquivos2.clear()

tab1 = tab1.iloc[0:0]
tab2 = tab2.iloc[0:0]

for arquivo in validation.glob("*.csv"):
    if arquivo.name.endswith("1.csv"):
        SPara = pd.read_csv(arquivo)
        arquivos1.append(SPara)
        print(f"Arquivo {arquivo.name} lido com {len(SPara)} linhas")
        print(f"heading:\n{SPara.head()}\n")
    if arquivo.name.endswith("2.csv"):
        CPara = pd.read_csv(arquivo)
        arquivos2.append(CPara)
        print(f"Arquivo {arquivo.name} lido com {len(CPara)} linhas")
        print(f"heading:\n{CPara.head()}\n")

for i, arquivos in enumerate(arquivos1):
    tab1 = pd.concat([tab1, arquivos], ignore_index=True)

for i, arquivos in enumerate(arquivos2):
    tab2 = pd.concat([tab2, arquivos], ignore_index=True)

tab1 = tab1.drop(columns=["trial"], errors="ignore")
tab2 = tab2.drop(columns=["trial"], errors="ignore")

for column in tab1.columns:
    tab1[column] = (tab1[column] - mu1[column]) / sigma1[column]

tab1.replace([np.inf, -np.inf], np.nan, inplace=True)
tab1.dropna(inplace=True)

tab1.to_csv("dataset/prepared/dicas/validation_normal.csv", index=False)

for column in tab2.columns:
    tab2[column] = (tab2[column] - mu2[column]) / sigma2[column]

tab2.dropna(inplace=True)

tab2.to_csv("dataset/prepared/dicas/validation_paraquedas.csv", index=False)

arquivos1.clear()
arquivos2.clear()

tab1 = tab1.iloc[0:0]
tab2 = tab2.iloc[0:0]

for arquivo in test.glob("*.csv"):
    if arquivo.name.endswith("1.csv"):
        SPara = pd.read_csv(arquivo)
        arquivos1.append(SPara)
        print(f"Arquivo {arquivo.name} lido com {len(SPara)} linhas")
        print(f"heading:\n{SPara.head()}\n")
    if arquivo.name.endswith("2.csv"):
        CPara = pd.read_csv(arquivo)
        arquivos2.append(CPara)
        print(f"Arquivo {arquivo.name} lido com {len(CPara)} linhas")
        print(f"heading:\n{CPara.head()}\n")

for i, arquivos in enumerate(arquivos1):
    tab1 = pd.concat([tab1, arquivos], ignore_index=True)

for i, arquivos in enumerate(arquivos2):
    tab2 = pd.concat([tab2, arquivos], ignore_index=True)

tab1 = tab1.drop(columns=["trial"], errors="ignore")
tab2 = tab2.drop(columns=["trial"], errors="ignore")

for column in tab1.columns:
    tab1[column] = (tab1[column] - mu1[column]) / sigma1[column]

tab1.replace([np.inf, -np.inf], np.nan, inplace=True)
tab1.dropna(inplace=True)

tab1.to_csv("dataset/prepared/dicas/test_normal.csv", index=False)

for column in tab2.columns:
    tab2[column] = (tab2[column] - mu2[column]) / sigma2[column]

tab2.dropna(inplace=True)

tab2.to_csv("dataset/prepared/dicas/test_paraquedas.csv", index=False)
