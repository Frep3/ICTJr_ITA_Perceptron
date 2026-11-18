"""Treina o modelo linear com o conjunto de dados normalizado."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Parâmetros deste experimento. Permanecem locais a este perceptron.
SEED = 42
TAXA_APRENDIZADO = 0.0001
MAX_EPOCAS = 500
TAMANHO_LOTE = 32
ERRO_ALVO = 0.05


def prever(entradas: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    """Calcula a saída linear para uma ou mais amostras."""
    return entradas @ pesos


def treinar_epoca(lotes_x, lotes_y, pesos, taxa_aprendizado):
    """Atualiza os pesos em mini-lotes e calcula o MSE da época."""
    pesos = pesos.copy()

    for entradas_lote, alvos_lote in zip(lotes_x, lotes_y):
        erros = alvos_lote - prever(entradas_lote, pesos)
        gradiente = entradas_lote.T @ erros / len(alvos_lote)
        pesos += taxa_aprendizado * gradiente

    erros_finais = np.concatenate(
        [alvos_lote - prever(entradas_lote, pesos)
         for entradas_lote, alvos_lote in zip(lotes_x, lotes_y)]
    )
    return pesos, float(np.mean(erros_finais ** 2))


def treinar_modelo(lotes_x, lotes_y):
    """Treina este modelo com inicialização e histórico próprios."""
    gerador = np.random.default_rng(SEED)
    pesos = gerador.uniform(-1, 1, size=lotes_x[0].shape[1])
    historico_erros = []
    historico_pesos = [pesos.copy()]

    for _ in range(MAX_EPOCAS):
        pesos, mse = treinar_epoca(
            lotes_x, lotes_y, pesos, TAXA_APRENDIZADO
        )
        historico_erros.append(mse)
        historico_pesos.append(pesos.copy())
        if mse <= ERRO_ALVO:
            break

    return pesos, historico_erros, historico_pesos


def carregar_dados(caminho, coluna_alvo="a"):
    """Carrega o CSV, separa o alvo e acrescenta o termo bias."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {caminho}")

    dados = pd.read_csv(caminho)
    if coluna_alvo not in dados.columns:
        raise ValueError(f"A coluna-alvo '{coluna_alvo}' não existe no dataset.")

    numericos = dados.select_dtypes(include=[np.number]).copy()
    if numericos.empty:
        raise ValueError("O dataset não contém colunas numéricas.")

    alvos = numericos.pop(coluna_alvo).to_numpy(dtype=float)
    entradas = numericos.to_numpy(dtype=float)
    validas = np.isfinite(entradas).all(axis=1) & np.isfinite(alvos)
    entradas, alvos = entradas[validas], alvos[validas]
    if len(alvos) == 0:
        raise ValueError("Não há amostras numéricas válidas para treinamento.")

    entradas = np.column_stack((entradas, np.ones(len(entradas))))
    nomes_atributos = list(numericos.columns) + ["bias"]
    return entradas, alvos, nomes_atributos


def criar_lotes(entradas, alvos):
    """Divide as amostras em mini-lotes; o último pode ser menor."""
    lotes_x, lotes_y = [], []
    for inicio in range(0, len(entradas), TAMANHO_LOTE):
        fim = inicio + TAMANHO_LOTE
        lotes_x.append(entradas[inicio:fim])
        lotes_y.append(alvos[inicio:fim])
    return lotes_x, lotes_y


def gerar_graficos(historico_erros, historico_pesos, nomes_atributos):
    """Plota o erro e a evolução dos pesos deste experimento."""
    plt.rcParams["figure.figsize"] = (9, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    plt.figure()
    plt.plot(range(1, len(historico_erros) + 1), historico_erros, marker="o")
    plt.xlabel("Época")
    plt.ylabel("Erro quadrático médio (MSE)")
    plt.title("MSE durante o treinamento — Dados normalizados")
    plt.tight_layout()

    pesos = np.asarray(historico_pesos)
    plt.figure()
    for indice, nome in enumerate(nomes_atributos):
        plt.plot(range(len(pesos)), pesos[:, indice], label=f"w_{nome}")
    plt.xlabel("Época")
    plt.ylabel("Valor do peso")
    plt.title("Evolução dos pesos — Dados normalizados")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    raiz_projeto = Path(__file__).resolve().parent
    caminho_dados = raiz_projeto / "dataset/prepared/normal/train_paraquedas.csv"
    entradas, alvos, nomes_atributos = carregar_dados(caminho_dados)

    # Embaralha as amostras de forma reprodutível neste experimento.
    gerador = np.random.default_rng(SEED)
    indices = gerador.permutation(len(alvos))
    entradas, alvos = entradas[indices], alvos[indices]

    lotes_x, lotes_y = criar_lotes(entradas, alvos)
    pesos, historico_erros, historico_pesos = treinar_modelo(lotes_x, lotes_y)

    previsoes = entradas @ pesos
    mse_final = float(np.mean((alvos - previsoes) ** 2))
    print(f"MSE final (Dados normalizados): {mse_final:.4f}")
    gerar_graficos(historico_erros, historico_pesos, nomes_atributos)


if __name__ == "__main__":
    main()
