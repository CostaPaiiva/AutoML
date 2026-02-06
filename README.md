🤖 AutoML - Sistema Automático de Machine Learning

Sistema completo e robusto para processamento automático de dados e treinamento de múltiplos modelos de Machine Learning com interface web intuitiva.

✨ Funcionalidades Principais
🔍 Processamento Inteligente
✅ Upload automático de arquivos CSV, TXT e Excel

✅ Detecção automática do tipo de problema (Classificação/Regressão)

✅ Limpeza inteligente de dados (missing values, outliers, duplicatas)

✅ Codificação automática de variáveis categóricas

✅ Normalização e escalonamento de features

🤖 Machine Learning Avançado
✅ 4+ algoritmos por tipo de problema

✅ Treinamento paralelo de múltiplos modelos

✅ Validação cruzada automática

✅ Seleção do melhor modelo baseado em métricas

✅ Ranking completo dos modelos treinados

📊 Dashboard Interativo
✅ Visualizações com Plotly

✅ Métricas detalhadas por modelo

✅ Gráficos comparativos

✅ Exportação de resultados (CSV, modelos, relatórios)

✅ Interface responsiva e amigável

🚀 Começando
Pré-requisitos
Python 3.8 ou superior

pip (gerenciador de pacotes Python)

Instalação
Clone o repositório ou baixe os arquivos

bash
git clone https://github.com/CostaPaiiva/AutoML.git
cd automl-system
Instale as dependências

bash
pip install -r requirements.txt
Execute o sistema

bash
streamlit run app_ultra_robust.py
Acesse no navegador

text
http://localhost:8501
Instalação Rápida (Windows)

powershell
# Execute o instalador automático
install.bat
📋 Como Usar
Passo 1: Upload do Dataset
Clique em "Escolha um arquivo CSV"

Selecione seu dataset (CSV, TXT ou Excel)

O sistema mostrará uma pré-visualização

Passo 2: Configuração
Selecione a coluna target (variável a ser prevista)

Ajuste configurações avançadas se necessário

Clique em "Processar Dados"

Passo 3: Treinamento
Revise as informações do processamento

Clique em "Iniciar Treinamento"

Aguarde enquanto os modelos são treinados

Passo 4: Resultados
Analise o ranking dos modelos

Visualize gráficos comparativos

Exporte os resultados

Baixe o melhor modelo treinado

🔧 Tecnologias Utilizadas
Python 3.8+ - Linguagem principal

Streamlit - Framework para aplicações web

Scikit-learn - Machine Learning

Pandas - Manipulação de dados

NumPy - Computação numérica

Plotly - Visualizações interativas

Joblib - Serialização de modelos

📊 Modelos Implementados:

Para Classificação:

✅ Logistic Regression

✅ Random Forest Classifier

✅ Decision Tree Classifier

✅ Naive Bayes (Gaussian)

Para Regressão:

✅ Linear Regression

✅ Ridge Regression

✅ Random Forest Regressor

✅ Decision Tree Regressor

📈 Métricas de Avaliação:

Classificação

Acurácia - Porcentagem de previsões corretas

F1-Score - Média harmônica entre precisão e recall

Validação Cruzada - Score médio em múltiplos folds

Regressão
R² Score - Qualidade do ajuste do modelo

RMSE - Raiz do erro quadrático médio

Validação Cruzada - Score médio em múltiplos folds

🎯 Casos de Uso
1. Análise Preditiva
text
- Previsão de churn de clientes
- Detecção de fraudes
- Classificação de sentimentos

2. Regressão de Valores
text
- Previsão de preços de imóveis
- Estimativa de vendas
- Previsão de demanda

3. Pesquisa Acadêmica
text
- Experimentos com diferentes algoritmos
- Comparação de modelos
- Análise exploratória de dados

4. Prototipagem Rápida
text
- MVP de soluções de ML
- Testes com novos datasets
- Validação de hipóteses
🔍 Exemplos de Datasets
Dataset de Demonstração (Iris)
python
# Características: 4 features numéricas
# Target: 3 classes de flores
# Tamanho: 150 amostras
# Tipo: Classificação Multiclasse
Para Testar:
Iris Dataset - Classificação de flores

Diabetes Dataset - Regressão (valores contínuos)

Titanic Dataset - Classificação binária

Boston Housing - Regressão de preços

⚙️ Configurações Avançadas
Opções Disponíveis:
Escalonamento de Features - Ativar/desativar normalização

Tamanho do Teste - 20% padrão (ajustável no código)

Número de Folds - Validação cruzada com 3 folds

Paralelismo - Usa todos os núcleos da CPU disponíveis

Personalização:
python
# No arquivo app.py, você pode modificar:

# 1. Adicionar novos modelos
models['Novo Modelo'] = SeuModelo(parametros)

# 2. Alterar métricas de avaliação
scoring = 'f1'  # Em vez de 'accuracy'

# 3. Ajustar tamanho do split
test_size = 0.3  # 30% para teste
📤 Exportação de Resultados
1. CSV do Ranking
csv
Posição,Modelo,Score
1,Random Forest,0.95
2,Logistic Regression,0.92
3,Decision Tree,0.89
2. Modelo Treinado
Formato: .pkl (Joblib)

Pode ser carregado em produção

Inclui todos os parâmetros otimizados

3. Relatório de Análise
Métricas detalhadas

Configurações usadas

Recomendações

🚨 Solução de Problemas
Erro Comum 1: "No columns to parse from file"
Solução: Verifique se o arquivo CSV está bem formatado e tem delimitadores corretos.

Erro Comum 2: "Memory Error"
Solução:

Reduza o tamanho do dataset

Use .sample() para testar com menos dados

Aumente a memória disponível

Erro Comum 3: "ValueError with stratify"
Solução: O sistema detecta automaticamente e usa split sem stratify quando necessário.

Erro Comum 4: "ImportError"
Solução: Instale todas as dependências:

bash
pip install -r requirements.txt --upgrade
📊 Benchmark de Performance
Dataset Size	Tempo de Processamento	Tempo de Treinamento
1,000 linhas	2-5 segundos	10-20 segundos
10,000 linhas	5-10 segundos	30-60 segundos
100,000 linhas	15-30 segundos	2-5 minutos
Testado em CPU Intel i7 com 16GB RAM

🔮 Roadmap de Melhorias
Versão 2.0 (Planejada)
Deep Learning - Redes neurais integradas

AutoML Avançado - Otimização automática de hiperparâmetros

Explainable AI - Explicabilidade dos modelos (SHAP/LIME)

Big Data - Suporte a datasets muito grandes

Deploy Cloud - Integração com AWS/GCP/Azure

Versão 1.5 (Em Desenvolvimento)
Mais Modelos - XGBoost, LightGBM, CatBoost

Balanceamento - Técnicas para dados desbalanceados

Feature Engineering - Automático avançado

API REST - Para integração com outros sistemas

🤝 Contribuindo
Contribuições são bem-vindas! Siga estes passos:

Fork o projeto

Crie uma branch para sua feature (git checkout -b feature/AmazingFeature)

Commit suas mudanças (git commit -m 'Add some AmazingFeature')

Push para a branch (git push origin feature/AmazingFeature)

Abra um Pull Request

Diretrizes de Código
Siga o padrão PEP 8

Adicione docstrings para novas funções

Inclua testes quando possível

Mantenha a compatibilidade com versões anteriores

📝 Licença
Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.


⚠️ NOTA: ESTE É UM SISTEMA EDUCACIONAL PARA FINS DE ESTUDO



##  **Próximos passos:**

1. **Teste com diferentes datasets** para ver como se comporta
2. **Adicione mais modelos** para expandir
3. **Crie uma versão para deploy** no Streamlit Cloud
4. **Adicione mais visualizações** de dados