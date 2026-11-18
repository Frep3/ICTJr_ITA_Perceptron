"""Aplica critérios de qualidade e registra a aprovação de cada ensaio."""

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_ENTRADA = Path("./dataset/cleaned")
PASTA_SAIDA = Path("./dataset/audited")

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

ARQUIVOS = [
    "cola_limpo.csv",
    "gesso_limpo.csv",
    "vazia_limpo.csv",
]

VELOCIDADE_MAXIMA = 25.0
MINIMO_QUADROS = 60


# ============================================================
# FUNÇÃO PARA CRIAR ASSINATURA DA TRAJETÓRIA
# ============================================================


def assinatura_trajetoria(df):
    """
    Cria uma assinatura para identificar trajetórias
    iguais ou invertidas no tempo.
    """

    dados = df[["t", "y", "vy", "x"]].round(6)

    return tuple(map(tuple, dados.to_numpy()))


# ============================================================
# AUDITORIA DE UM ARQUIVO
# ============================================================


def auditar_arquivo(caminho_entrada):

    print()
    print("=" * 60)
    print(f"Processando: {caminho_entrada}")
    print("=" * 60)

    df = pd.read_csv(caminho_entrada)

    # --------------------------------------------------------
    # Verificação das colunas
    # --------------------------------------------------------

    colunas_obrigatorias = ["t", "y", "vy", "x", "trial", "massa"]

    colunas_faltantes = [
        coluna for coluna in colunas_obrigatorias if coluna not in df.columns
    ]

    if colunas_faltantes:
        print("ERRO: colunas ausentes:", ", ".join(colunas_faltantes))
        return

    resultados = []

    # Guarda as trajetórias já encontradas
    assinaturas = {}

    # --------------------------------------------------------
    # Analisa cada trial
    # --------------------------------------------------------

    for trial, grupo in df.groupby("trial", sort=False):
        grupo = grupo.reset_index(drop=True)

        motivos = []

        # ====================================================
        # CRITÉRIO 1 — MÍNIMO DE QUADROS
        # ====================================================

        numero_quadros = len(grupo)

        quadros_ok = numero_quadros >= MINIMO_QUADROS

        if not quadros_ok:
            motivos.append("menos_de_60_quadros")

        # ====================================================
        # CRITÉRIO 2 — TEMPO ESTRITAMENTE CRESCENTE
        # ====================================================

        dt = grupo["t"].diff().dropna()

        tempo_ok = (dt > 0).all()

        if not tempo_ok:
            motivos.append("tempo_nao_estritamente_crescente")

        # ====================================================
        # CRITÉRIO 3 — VELOCIDADE MÁXIMA
        # ====================================================

        dy = grupo["y"].diff()
        dt = grupo["t"].diff()

        # Evita divisão por zero
        velocidade = (dy / dt).replace([np.inf, -np.inf], np.nan).abs()

        velocidade_maxima = velocidade.max()

        velocidade_ok = (
            pd.notna(velocidade_maxima) and velocidade_maxima <= VELOCIDADE_MAXIMA
        )

        if not velocidade_ok:
            motivos.append("velocidade_acima_de_25_m_s")

        # ====================================================
        # CRITÉRIO 4 — Y MONOTONICAMENTE DECRESCENTE
        # ====================================================

        dy_y = grupo["y"].diff().dropna()

        y_monotono_ok = not (dy_y > 0).any()

        if not y_monotono_ok:
            motivos.append("y_nao_monotonicamente_decrescente")

        # ====================================================
        # CRITÉRIO 5 — DUPLICATA
        # ====================================================

        assinatura = assinatura_trajetoria(grupo)

        duplicata = False
        tipo_duplicata = ""
        duplicata_de = ""

        # -----------------------------------------------
        # Verifica duplicata exata
        # -----------------------------------------------

        if assinatura in assinaturas:
            duplicata = True
            tipo_duplicata = "exata"
            duplicata_de = assinaturas[assinatura]

            motivos.append("duplicata_exata")

        else:
            # -------------------------------------------
            # Verifica trajetória invertida no tempo
            # -------------------------------------------

            assinatura_invertida = tuple(reversed(assinatura))

            if assinatura_invertida in assinaturas:
                duplicata = True
                tipo_duplicata = "invertida_no_tempo"
                duplicata_de = assinaturas[assinatura_invertida]

                motivos.append("duplicata_invertida_no_tempo")

        # Se não foi duplicata, registra a assinatura
        if not duplicata:
            assinaturas[assinatura] = trial

        # ====================================================
        # RESULTADO FINAL
        # ====================================================

        aprovado = len(motivos) == 0

        if aprovado:
            print(f"  {trial}: APROVADO")
        else:
            print(f"  {trial}: REJEITADO")
            print(f"    Motivo(s): {'; '.join(motivos)}")

        # ====================================================
        # SALVA RESULTADO DA AUDITORIA
        # ====================================================

        resultados.append(
            {
                "trial": trial,
                "massa": grupo["massa"].iloc[0],
                "quadros": numero_quadros,
                "quadros_ok": quadros_ok,
                "tempo_ok": tempo_ok,
                "velocidade_maxima": velocidade_maxima,
                "velocidade_ok": velocidade_ok,
                "y_monotono_ok": y_monotono_ok,
                "duplicata": duplicata,
                "tipo_duplicata": tipo_duplicata,
                "duplicata_de": duplicata_de,
                "aprovado": aprovado,
                "motivos_rejeicao": "; ".join(motivos),
            }
        )

    # ========================================================
    # CRIA DATAFRAME DA AUDITORIA
    # ========================================================

    df_auditoria = pd.DataFrame(resultados)

    # ========================================================
    # SALVA
    # ========================================================

    nome_saida = caminho_entrada.stem.replace("_limpo", "_auditado") + ".csv"

    caminho_saida = PASTA_SAIDA / nome_saida

    df_auditoria.to_csv(caminho_saida, index=False)

    print()
    print(f"Salvo em: {caminho_saida}")

    # ========================================================
    # RESUMO
    # ========================================================

    total = len(df_auditoria)
    aprovados = df_auditoria["aprovado"].sum()
    rejeitados = total - aprovados

    print()
    print("Resumo:")
    print(f"  Total de trials: {total}")
    print(f"  Aprovados: {aprovados}")
    print(f"  Rejeitados: {rejeitados}")


# ============================================================
# EXECUÇÃO
# ============================================================

for nome_arquivo in ARQUIVOS:
    caminho = PASTA_ENTRADA / nome_arquivo

    if not caminho.exists():
        print()
        print(f"Arquivo não encontrado: {caminho}")

        continue

    auditar_arquivo(caminho)


print()
print("=" * 60)
print("AUDITORIA CONCLUÍDA")
print("=" * 60)
