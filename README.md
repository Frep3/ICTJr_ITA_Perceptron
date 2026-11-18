# Projeto de iniciação científica — Instituto Tecnológico de Aeronáutica

Este repositório reúne o processamento de dados experimentais e dois experimentos
com um modelo de um neurônio com saída linear. O projeto trabalha com os arquivos
de ensaios dos materiais **cola**, **gesso** e **vazia**.

> **Nota sobre o modelo:** apesar dos nomes históricos dos arquivos (`perceptron.py`
> e `perceptron_dicas.py`), a implementação atual usa uma saída linear e minimiza
> o erro quadrático médio (MSE). Portanto, matematicamente, trata-se de um modelo
> linear treinado por gradiente descendente, e não de um perceptron clássico de
> classificação com função degrau.

## Estrutura

```text
.
├── dataset/
│   ├── raw/          # Dados originais
│   ├── cleaned/      # Ensaios extraídos dos arquivos originais
│   ├── audited/      # Resultado da auditoria de qualidade
│   ├── processed/    # Dados com velocidade/aceleração derivadas
│   ├── separated/   # Divisão temporal: treino, validação e teste
│   ├── grupos/       # Separação dos grupos de ensaios
│   ├── dicas/        # Atributos derivados (ex.: quadrado e inverso)
│   └── prepared/     # Dados normalizados prontos para os modelos
├── processors/       # Etapas de preparação e validação dos dados
├── perceptron.py     # Experimento com os dados normalizados
├── perceptron_dicas.py
├── tratar_dados.py   # Executa o pipeline de preparação
└── requirements.txt
```

## Requisitos

- Python 3.10 ou superior
- NumPy
- pandas
- SciPy
- Matplotlib

Instale as dependências em um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No CachyOS/Linux, caso o comando `python` não esteja disponível, use `python3`
nos comandos acima.

## Preparação dos dados

Execute a partir de qualquer diretório:

```bash
python /caminho/para/o/projeto/tratar_dados.py
```

O script localiza os arquivos usando o caminho do próprio projeto e executa as
etapas na ordem necessária:

1. Limpeza dos arquivos brutos;
2. Auditoria dos ensaios;
3. Cálculo de velocidade e aceleração com o filtro de Savitzky–Golay;
4. Separação temporal em treino, validação e teste;
5. Separação dos grupos de ensaios;
6. Criação de atributos derivados;
7. Normalização e preparação dos conjuntos.

As mensagens rotineiras das etapas são ocultadas para manter o terminal limpo.
Se uma etapa falhar, o processo é interrompido e o erro é apresentado.

## Treinamento dos modelos

Depois de preparar os dados, execute na pasta do projeto:

```bash
python perceptron.py
```

Esse experimento usa `dataset/prepared/normal/train_paraquedas.csv`.

Para executar o experimento com os atributos derivados:

```bash
python perceptron_dicas.py
```

Esse experimento usa `dataset/prepared/dicas/train_normal.csv`.

Cada script apresenta o MSE final e abre gráficos com a evolução do erro e dos
pesos. A semente aleatória é fixa (`42`) para facilitar a reprodução dos
resultados.

## Como o modelo aprende

Para cada amostra, o modelo calcula uma combinação linear dos atributos:

\[
\hat{y} = \mathbf{x}^{T}\mathbf{w}
\]

O termo constante acrescentado às entradas representa o *bias*. Os pesos são
ajustados para reduzir o erro quadrático médio:

\[
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
\]

O treinamento utiliza gradiente descendente em mini-lotes. O histórico dos pesos
e do MSE é guardado para produzir os gráficos ao final.

## Observações para reprodutibilidade científica

- Os dados brutos em `dataset/raw/` devem ser preservados.
- Os diretórios intermediários são resultados do pipeline e podem ser recriados.
- A normalização de validação e teste utiliza as estatísticas calculadas no
  conjunto de treino, evitando ajustar a escala com os dados de avaliação.
- Os parâmetros do filtro, os critérios de auditoria, a taxa de aprendizado,
  o tamanho do lote e o limite de épocas estão definidos nos scripts.
- Antes de interpretar resultados, confira os critérios de aprovação dos ensaios
  e a distribuição dos exemplos em cada partição.

## Arquivos principais

- `perceptron.py`: contém de forma independente as funções de leitura, treinamento e visualização do experimento com dados normalizados.
- `perceptron_dicas.py`: contém de forma independente as funções de leitura, treinamento e visualização do experimento com atributos derivados.
- `processors/limpar_dados.py`: leitura e organização dos ensaios originais.
- `processors/auditar_dados.py`: critérios de qualidade e identificação de duplicatas.
- `processors/data_process.py`: cálculo das grandezas derivadas.
- `processors/data_separate.py` e `processors/split_separate.py`: divisão dos dados.
- `processors/data_prepare.py` e `processors/data_prepare_dicas.py`: normalização.
