"""Executa, em ordem, as etapas de preparação dos dados."""

import subprocess
import sys
from pathlib import Path


RAIZ_PROJETO = Path(__file__).resolve().parent
PASTA_PROCESSADORES = RAIZ_PROJETO / "processors"

# A ordem é importante: cada etapa utiliza arquivos produzidos pela anterior.
ETAPAS = [
    "limpar_dados.py",
    "auditar_dados.py",
    "data_process.py",
    "data_separate.py",
    "split_separate.py",
    "dicas.py",
    "data_prepare.py",
    "data_prepare_dicas.py",
]


def main():
    for etapa in ETAPAS:
        caminho = PASTA_PROCESSADORES / etapa
        if not caminho.is_file():
            raise FileNotFoundError(f"Etapa não encontrada: {caminho.name}")

        # check=True interrompe o pipeline se uma etapa falhar.
        subprocess.run(
            [sys.executable, str(caminho)],
            cwd=RAIZ_PROJETO,
            stdout=subprocess.DEVNULL,  # Oculta mensagens rotineiras de cada etapa.
            check=True,
        )


if __name__ == "__main__":
    main()
