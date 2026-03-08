# 🤖 AutoML

Sistema de **AutoML com interface web em Streamlit** para ingestão de datasets tabulares, detecção automática do tipo de problema, processamento inteligente de dados, treinamento massivo de modelos de Machine Learning, comparação de desempenho, exportação de artefatos e geração de relatórios.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-red">
  <img alt="Scikit-Learn" src="https://img.shields.io/badge/Scikit--Learn-ML-orange">
  <img alt="Plotly" src="https://img.shields.io/badge/Plotly-Interactive%20Charts-6f42c1">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## Visão geral

O **AutoML** foi criado para reduzir o trabalho manual na construção de modelos supervisionados. Em vez de exigir scripts separados para limpeza, transformação, treinamento, avaliação e exportação, a aplicação centraliza todo o fluxo em uma experiência visual e orientada por etapas.

O sistema permite:

- carregar datasets tabulares;
- detectar automaticamente o tipo de problema;
- processar dados com limpeza, imputação, encoding, escalonamento e seleção de atributos;
- treinar dezenas de algoritmos em paralelo;
- comparar métricas com validação cruzada;
- selecionar e salvar o melhor modelo;
- exportar ranking, modelo treinado e relatório final.

---

## Principais funcionalidades

### Upload e leitura de dados
- Upload de arquivos via interface web.
- Suporte a dados tabulares em formatos como **CSV, TXT e Excel**.
- Pré-visualização do dataset antes do processamento.
- Seleção manual ou detecção automática da coluna alvo (*target*).

### Processamento inteligente
- Remoção de duplicatas.
- Tratamento de valores infinitos e ausentes.
- Remoção de colunas com alta taxa de dados faltantes.
- Tratamento de outliers para datasets menores.
- Codificação automática de variáveis categóricas.
- Escalonamento de atributos numéricos.
- Engenharia de atributos com combinações e estatísticas derivadas.
- Seleção automática das melhores features.

### Treinamento automatizado
- Detecção de **classificação** ou **regressão**.
- Treinamento massivo de múltiplos algoritmos.
- Validação cruzada configurável.
- Estratégias diferentes para classificação e regressão.
- Ranqueamento dos modelos com base em métricas objetivas.
- Criação de ensembles.
- Otimização de hiperparâmetros para modelos de topo.

### Resultados e exportação
- Ranking completo dos modelos treinados.
- Métricas detalhadas por algoritmo.
- Gráficos interativos com Plotly.
- Exportação do ranking em CSV.
- Salvamento do melhor modelo em `.pkl`.
- Geração de relatório em **PDF** ou fallback em **TXT**.

---

## Modelos suportados

O projeto implementa um conjunto amplo de algoritmos para problemas supervisionados.

### Classificação
- Logistic Regression
- Ridge Classifier
- SGD Classifier
- SVC / NuSVC / LinearSVC
- KNeighbors Classifier / RadiusNeighbors Classifier
- Decision Tree / Extra Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Bagging
- Extra Trees
- HistGradientBoosting
- GaussianNB / BernoulliNB / MultinomialNB
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- MLP Classifier
- XGBoost
- LightGBM
- CatBoost
- Voting Classifier

### Regressão
- Linear Regression
- Ridge
- Lasso
- ElasticNet
- SGD Regressor
- SVR / NuSVR / LinearSVR
- KNeighbors Regressor / RadiusNeighbors Regressor
- Decision Tree Regressor / Extra Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- Bagging Regressor
- Extra Trees Regressor
- HistGradientBoosting Regressor
- Kernel Ridge
- MLP Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- Voting Regressor

---

## Métricas avaliadas

### Classificação
- Accuracy
- F1 Score
- Precision
- Recall
- ROC AUC
- Desvio padrão dos folds
- Tempo médio de treino e score

### Regressão
- R² Score
- RMSE
- MAE
- Explained Variance
- Desvio padrão dos folds
- Tempo médio de treino e score

---

## Interface da aplicação

A aplicação principal foi desenvolvida em **Streamlit** e organizada em etapas, o que torna o fluxo mais claro para o usuário:

1. **Upload do dataset**  
   Envio do arquivo e inspeção inicial dos dados.

2. **Seleção/detecção do target**  
   Escolha manual da variável alvo ou uso da detecção automática.

3. **Processamento de dados**  
   Limpeza, transformação, engenharia e seleção de atributos.

4. **Treinamento com validação cruzada**  
   Configuração do número de folds, estratégia de CV e paralelismo.

5. **Resultados**  
   Exibição do melhor modelo, métricas, ranking e opções de exportação.

---

## 🏗️ Arquitetura do projeto

```bash
AutoML/
├── app.py
├── dashboard.py
├── data_processing.py
├── model_training.py
├── report_generator.py
├── requirements.txt
├── LICENSE
├── .gitignore
├── models/
│   └── *.pkl
├── reports/
│   └── relatorio_automl_*.txt / *.pdf
└── README.md
```

### Responsabilidade dos módulos

#### `app.py`
Arquivo principal da aplicação Streamlit. Reúne a interface, o fluxo da aplicação, o treinamento, o ranking e as exportações.

#### `data_processing.py`
Implementa o pipeline de preparação de dados, incluindo limpeza, imputação, encoding, escalonamento e seleção de features.

#### `model_training.py`
Contém o motor de treinamento automatizado, avaliação, ranking, ensemble e otimização de hiperparâmetros.

#### `dashboard.py`
Disponibiliza um dashboard avançado com **Dash + Plotly** para análise visual dos resultados.

#### `report_generator.py`
Responsável pela geração de relatórios detalhados em PDF com estrutura executiva.

---

## Tecnologias utilizadas

- **Python** — linguagem principal
- **Streamlit** — interface web interativa
- **Scikit-learn** — modelos, métricas e pré-processamento
- **Pandas** — manipulação de dados
- **NumPy** — operações numéricas
- **Plotly** — visualizações interativas
- **Joblib** — serialização de modelos
- **Optuna** — otimização de hiperparâmetros
- **XGBoost** — gradient boosting avançado
- **LightGBM** — boosting eficiente
- **CatBoost** — boosting para dados tabulares
- **Dash + Bootstrap Components** — dashboard adicional
- **FPDF / ReportLab / Matplotlib** — geração de relatórios

---

## Como executar localmente

### 1) Clone o repositório

```bash
git clone https://github.com/CostaPaiiva/AutoML.git
cd AutoML
```

### 2) Crie e ative um ambiente virtual

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Instale as dependências

```bash
pip install -r requirements.txt
```

### 4) Execute a aplicação

```bash
streamlit run app.py
```

### 5) Acesse no navegador

```text
http://localhost:8501
```

---

## 📥 Requisitos

O arquivo `requirements.txt` atual contém as dependências essenciais abaixo:

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.14.0
joblib>=1.3.0
```

### Dependências adicionais recomendadas

Como o código também usa bibliotecas extras para treinamento avançado, dashboard e relatórios, recomenda-se instalar adicionalmente:

```bash
pip install optuna xgboost lightgbm catboost dash dash-bootstrap-components fpdf2 reportlab matplotlib openpyxl
```

> **Observação:** o código-fonte importa mais bibliotecas do que as listadas atualmente em `requirements.txt`. Para rodar todos os recursos do projeto sem erros de importação, é importante complementar essas dependências.

---

## Fluxo de uso

```text
Upload do arquivo
   ↓
Seleção ou detecção do target
   ↓
Detecção do tipo de problema
   ↓
Pré-processamento automático
   ↓
Treinamento com validação cruzada
   ↓
Ranking dos modelos
   ↓
Exportação do melhor modelo e dos relatórios
```

---

## Saídas geradas

Durante o uso do sistema, podem ser gerados artefatos como:

- **Modelos serializados** em `.pkl`
- **Relatórios** em `.pdf` ou `.txt`
- **Ranking** de modelos em `.csv`

Esses artefatos facilitam auditoria, reaproveitamento e integração com outros fluxos de ML.

---

## Casos de uso

O AutoML pode ser aplicado em diferentes cenários, como:

- previsão de churn;
- classificação de clientes, produtos ou eventos;
- detecção de fraude;
- previsão de preços, vendas ou demanda;
- comparação rápida de modelos em projetos acadêmicos;
- prototipagem de soluções de Machine Learning para MVPs.

---

## Pontos fortes do projeto

- Interface amigável para usuários iniciantes e intermediários.
- Pipeline automatizado de ponta a ponta.
- Ampla cobertura de algoritmos.
- Comparação objetiva entre modelos.
- Exportação facilitada de resultados.
- Estrutura modular, com separação entre interface, processamento, treino e relatório.

---

## Limitações e melhorias sugeridas

Embora o projeto esteja bem estruturado para estudo e prototipagem, há oportunidades claras de evolução:

- consolidar todas as dependências no `requirements.txt`;
- adicionar testes automatizados;
- incluir exemplos de datasets e screenshots no repositório;
- versionar pipelines de inferência separadamente;
- documentar melhor os formatos esperados de entrada;
- adicionar deploy em nuvem;
- criar API para inferência em produção.

---

## Roadmap sugerido

- [ ] Criar arquivo `requirements` completo
- [ ] Adicionar inferência com novo dataset
- [ ] Incluir explicabilidade com SHAP ou LIME
- [ ] Publicar demo em Streamlit Cloud
- [ ] Criar endpoint REST para previsão
- [ ] Adicionar monitoramento de performance do modelo
- [ ] Incluir persistência de experimentos

---

## Contribuição

Contribuições são bem-vindas.

1. Faça um fork do projeto.
2. Crie uma branch para sua feature:

```bash
git checkout -b feature/minha-melhoria
```

3. Commit suas alterações:

```bash
git commit -m "feat: adiciona melhoria X"
```

4. Envie para o repositório remoto:

```bash
git push origin feature/minha-melhoria
```

5. Abra um Pull Request.

---

## Licença

Este projeto está licenciado sob os termos da **MIT License**. Consulte o arquivo `LICENSE` para mais detalhes.

---

## 👨‍💻 Autor

Desenvolvido por **CostaPaiiva**.

Se este projeto foi útil para você, deixe uma estrela no repositório.

---

## Aviso

Este projeto foi desenvolvido **para fins de estudo, aprendizado e experimentação**.
