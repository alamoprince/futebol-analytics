# Football Predictor Pro (BackDraw Model)

Ferramenta para análise e previsão de resultados de futebol, focada na estratégia "Back the Draw" (Apostar no Empate) utilizando Hiperparâmetros.

## Descrição

Este projeto utiliza dados históricos de partidas de futebol (incluindo odds) de um arquivo Excel e dados de jogos futuros (com odds) de uma fonte externa (CSV do GitHub) para treinar e avaliar múltiplos modelos de Machine Learning (Random Forest, Logistic Regression, SVC, Gaussian Naive Bayes, K-Nearest Neighbors, opcionalmente LightGBM).

O objetivo principal é identificar jogos com potencial de empate (modelo de classificação binária). O sistema calcula métricas de desempenho do modelo, incluindo Acurácia, F1-Score para Empates, e um cálculo de Profit/ROI simulado para a estratégia "Back Draw" no conjunto de teste.

O projeto oferece duas interfaces: uma gráfica (GUI) usando Tkinter e uma de linha de comando (CLI) usando Rich. Ele também inclui a funcionalidade opcional de salvar as previsões geradas em um repositório GitHub configurado.

## Funcionalidades Principais

*   Carregamento de dados históricos de arquivo Excel.
*   Busca de dados de jogos futuros (com odds) de um arquivo CSV hospedado online.
*   Filtragem de jogos futuros por ligas de interesse.
*   Pré-processamento de dados e engenharia de features (cálculo de médias móveis, CV de odds, etc.).
*   Treinamento e avaliação de múltiplos modelos de classificação (RF, LR, SVC, GNB, KNN, LGBM opcional).

* Definição de relevância de cada features usando Hiperparâmetros.

*   Cálculo de Profit/ROI simulado para a estratégia "Back Draw" no conjunto de teste.
*   Salvar e carregar o melhor modelo treinado e suas estatísticas.
*   Realização de previsões para jogos futuros (probabilidade de Não Empate vs. Empate).
*   Interface Gráfica (GUI via `main.py`) e Interface de Linha de Comando (CLI via `dashboard.py`).
*   (Opcional) Salvar previsões geradas em um repositório GitHub.

## Pré-requisitos

*   Python 3.8 ou superior.
*   Pip (gerenciador de pacotes Python).
*   **Tkinter:** Necessário para a interface gráfica (`main.py`). Geralmente incluído no Windows e macOS. Pode precisar ser instalado separadamente no Linux (ex: `sudo apt-get update && sudo apt-get install python3-tk`).
*   **Git:** Para clonar o repositório (se aplicável).
*   **(Opcional) Token do GitHub:** Necessário *apenas* se você quiser usar a funcionalidade de salvar previsões no seu repositório.

## Instalação

1.  **Clone o Repositório (se estiver no Git):**
    ```bash
    git clone <url_do_seu_repositorio>
    cd <nome_da_pasta_do_projeto>
    ```
2.  **Crie e Ative um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Opcional) Instale LightGBM:** Se quiser testar o modelo LightGBM:
    ```bash
    pip install lightgbm
    ```

## Configuração

1.  **Dados Históricos:**
    *   Certifique-se de que seu arquivo Excel com dados históricos (ex: `Brasileirao_A_e_B (1).xlsx`) está na pasta `data/`.
    *   Verifique se o nome do arquivo corresponde a `HISTORICAL_DATA_FILENAME` em `src/config.py`.
    *   Confirme que o Excel contém as colunas listadas em `EXCEL_EXPECTED_COLS` em `src/config.py` com os nomes exatos.
2.  **Fonte de Dados Futuros (CSV):**
    *   Verifique a `FIXTURE_CSV_URL_TEMPLATE` em `src/config.py`. Ela deve apontar para a fonte correta dos CSVs diários.
    *   **MUITO IMPORTANTE:** Verifique os nomes reais das colunas no CSV fonte e ajuste as **chaves** do dicionário `FIXTURE_CSV_COL_MAP` em `src/config.py` para corresponderem exatamente.
    *   Ajuste `FIXTURE_FETCH_DAY` ('today' ou 'tomorrow') em `src/config.py` conforme sua preferência.
3.  **Ligas Alvo:**
    *   Edite a lista `TARGET_LEAGUES` em `src/config.py` com os nomes **exatos** das ligas (como aparecem na coluna 'League' do CSV) que você deseja incluir nas previsões. Se a lista estiver vazia (`[]`), nenhum filtro será aplicado.
4.  **Features do Modelo:**
    *   Revise a lista `FEATURE_COLUMNS` em `src/config.py`. Estas são as features que o modelo usará. Certifique-se de que elas podem ser obtidas (diretamente do histórico/CSV) ou calculadas (pelo `data_handler.py`).
5.  **(Opcional) Configuração do GitHub:**
    *   Se desejar salvar as previsões no GitHub, crie um arquivo chamado `.env` na pasta **raiz** do projeto (fora da pasta `src`).
    *   Adicione as seguintes linhas ao arquivo `.env`, substituindo pelos seus valores:
        ```dotenv
        GITHUB_TOKEN=seu_token_pessoal_aqui
        GITHUB_REPO_NAME=seu_usuario_github/nome_do_repositorio_destino
        ```
    *   Gere um token de acesso pessoal no GitHub com permissões de `repo`.
    *   Ajuste `GITHUB_PREDICTIONS_PATH` em `src/config.py` para o caminho desejado dentro do seu repositório.

## Uso

Existem duas interfaces disponíveis:

1.  **Interface Gráfica (GUI):**
    *   Execute: `python src/main.py`
    *   Use os botões:
        *   **Carregar Histórico e Treinar:** Carrega dados do Excel, treina os modelos configurados, seleciona o melhor, salva-o e exibe as estatísticas.
        *   **Prever Jogos:** Busca o CSV do dia/amanhã, prepara as features usando o histórico e o CSV, usa o modelo treinado para prever, e exibe os resultados (incluindo odds e probabilidades).

2.  **Interface de Linha de Comando (CLI):**
    *   Execute: `python src/dashboard.py`
    *   Siga as opções do menu:
        *   **[1] Treinar/Retreinar Modelo:** Mesmo processo da GUI.
        *   **[2] Prever Próximos Jogos:** Mesmo processo da GUI.
        *   **[3] Exibir Estatísticas do Modelo:** Mostra as métricas e parâmetros do último modelo treinado e salvo.
        *   **[4] Sair:** Encerra a aplicação.

## Estrutura do Projeto

```plaintext
futebol-analytics/
├── data/
│   ├── Brasileirao_A_e_B (1).xlsx  # Seu arquivo histórico
│   └── best_model_backdraw_best_vX.joblib  # Modelo salvo (nome pode variar)
├── src/
│   ├── __init__.py
│   ├── config.py  # Configurações gerais, paths, features, modelos
│   ├── data_handler.py  # Carregamento, pré-processamento, cálculo de features
│   ├── model_trainer.py  # Treinamento, avaliação, seleção de modelo, cálculo ROI
│   ├── predictor.py  # Carregamento de modelo, geração de previsões
│   ├── github_manager.py  # (Opcional) Interação com API do GitHub
│   ├── dashboard.py  # Interface de Linha de Comando (CLI)
│   └── main.py  # Interface Gráfica (GUI)
├── .env  # (Opcional) Arquivo para segredos do GitHub (NÃO   versionar!)
├── requirements.txt  # Dependências Python
└── README.md  # Este arquivo
```

## Detalhes do Modelo

*   O foco atual é um modelo de **Classificação Binária** para prever **Empate (1)** vs. **Não Empate (0)**.
*   Múltiplos algoritmos modificados a partir de Hiperparâmetros são testados (SVC, GaussianNB por padrão, outros podem ser habilitados no `config.py`).
*   O melhor modelo é selecionado com base na métrica `BEST_MODEL_METRIC` (padrão: `f1_score_draw`).
*   As features usadas estão definidas em `FEATURE_COLUMNS` no `config.py`.
*   A avaliação inclui métricas de classificação padrão e Profit/ROI simulado da estratégia "Back the Draw".

---

## Aprimoramentos Futuros
* Próprio WebScrapping para abranger maiores quantidades de jogos e ter maior liberdade para escolha das ligas.
* Melhorar o método de escolha de features atravé do RandomForest.
* Adicionar método de correlação de Pearson para avaliar features.
* Aplicar a Eliminação Recursiva de Features (Recursive Feature Elimination - RFE) por meio do RandomForest.

