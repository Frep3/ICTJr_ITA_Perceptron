"""Extrai e organiza os ensaios dos CSVs brutos em tabelas padronizadas."""

import re
from pathlib import Path

import pandas as pd


def nome_base_coluna(nome):
    """
    Remove o sufixo que o pandas coloca em nomes repetidos.

    Exemplos:
        t     -> t
        t.1   -> t
        y.2   -> y
        vy.5  -> vy
        x.10  -> x
    """
    nome = str(nome).strip().lower()

    resultado = re.match(r"^(t|y|vy|x)(?:\.\d+)?$", nome)

    if resultado:
        return resultado.group(1)

    return None


def limpar_trials(arquivo_entrada, arquivo_saida=None):

    arquivo_entrada = Path(arquivo_entrada)

    # Lê como texto para preservar a estrutura do CSV
    df = pd.read_csv(arquivo_entrada, decimal=",", dtype=str, keep_default_na=False)

    trials = []

    # Descobre as colunas vazias que separam os trials
    separadores = []

    for i, coluna in enumerate(df.columns):
        if str(coluna).startswith("Unnamed"):
            separadores.append(i)

    # Início e fim do arquivo
    limites = [0] + separadores + [len(df.columns)]

    numero_trial = 0

    for inicio, fim in zip(limites[:-1], limites[1:]):
        bloco = df.iloc[:, inicio:fim]

        # Identifica t, y, vy e x pelo nome base
        posicoes = {}

        for i, coluna in enumerate(bloco.columns):
            nome = nome_base_coluna(coluna)

            if nome is not None and nome not in posicoes:
                posicoes[nome] = i

        # Ignora blocos que não possuem as quatro variáveis
        if not all(nome in posicoes for nome in ("t", "y", "vy", "x")):
            continue

        # Monta o trial na ordem padrão
        trial = pd.DataFrame(
            {
                "t": bloco.iloc[:, posicoes["t"]],
                "y": bloco.iloc[:, posicoes["y"]],
                "vy": bloco.iloc[:, posicoes["vy"]],
                "x": bloco.iloc[:, posicoes["x"]],
            }
        )

        # Converte os valores para números
        for coluna in ["t", "y", "vy", "x"]:
            trial[coluna] = (
                trial[coluna].astype(str).str.strip().str.replace(",", ".", regex=False)
            )

            trial[coluna] = pd.to_numeric(trial[coluna], errors="coerce")

        # Remove linhas incompletas
        trial = trial.dropna(subset=["t", "y", "vy", "x"]).reset_index(drop=True)

        trial = trial.sort_values("t").reset_index(drop=True)

        if trial.empty:
            continue

        # Normaliza cada trial individualmente
        trial["t"] = trial["t"] - trial["t"].iloc[0]
        trial["y"] = trial["y"] - trial["y"].iloc[0]
        trial["x"] = trial["x"] - trial["x"].iloc[0]

        # Identificação do trial
        numero_trial += 1

        nome_base = arquivo_entrada.stem
        trial["trial"] = nome_base + str(numero_trial)

        massas = {"vazia": 3, "cola": 32, "gesso": 53}

        trial["massa"] = massas[nome_base]

        trials.append(trial)

    # Nenhum trial encontrado
    if not trials:
        raise ValueError(f"Nenhum trial completo encontrado em {arquivo_entrada}")

    # Junta todos os trials verticalmente
    resultado = pd.concat(trials, ignore_index=True)

    # Ordem final
    resultado = resultado[["t", "y", "vy", "x", "trial", "massa"]]

    # Salva
    if arquivo_saida is not None:
        arquivo_saida = Path(arquivo_saida)

        arquivo_saida.parent.mkdir(parents=True, exist_ok=True)

        resultado.to_csv(arquivo_saida, index=False)

    return resultado


if __name__ == "__main__":
    pasta_raw = Path("./dataset/raw")
    pasta_processed = Path("./dataset/processed")

    arquivos = [
        pasta_raw / "cola.csv",
        pasta_raw / "gesso.csv",
        pasta_raw / "vazia.csv",
    ]

    for arquivo in arquivos:
        if not arquivo.exists():
            print(f"Arquivo não encontrado: {arquivo}")
            continue

        saida = pasta_processed / f"{arquivo.stem}_limpo.csv"

        print(f"\nProcessando: {arquivo}")

        try:
            resultado = limpar_trials(arquivo, saida)

            print(f"Trials encontrados: {resultado['trial'].nunique()}")

            print(f"Linhas geradas: {len(resultado)}")

            print(f"Salvo em: {saida}")

        except Exception as erro:
            print(f"ERRO: {erro}")
