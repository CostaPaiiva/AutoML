# Importa a biblioteca Streamlit para criar aplicações web interativas
import streamlit as st
# Importa o pandas para manipulação de dados em tabelas
import pandas as pd
# Importa o numpy para cálculos numéricos e vetoriais
import numpy as np
# Importa o matplotlib para criação de gráficos estáticos
import matplotlib.pyplot as plt
# Importa o seaborn para visualizações estatísticas mais sofisticadas
import seaborn as sns
# Importa o Plotly Express para gráficos interativos simplificados
import plotly.express as px
# Importa objetos gráficos do Plotly para gráficos mais customizados
import plotly.graph_objects as go
# Importa função para criar subplots (gráficos compostos) no Plotly
from plotly.subplots import make_subplots
# Importa a biblioteca time para manipulação de tempo e pausas
import time
# Importa base64 para codificação de arquivos em texto
import base64
# Importa io para manipulação de streams de dados/arquivos
import io
# Importa joblib para salvar e carregar modelos de Machine Learning
import joblib
# Importa datetime para trabalhar com datas e horários
from datetime import datetime
# Importa warnings para controlar mensagens de aviso
import warnings
# Ignora todos os warnings para não poluir a saída
warnings.filterwarnings('ignore')

# Importa módulos internos do sistema (arquivos Python criados pelo usuário)
from data_processing import AdvancedDataProcessor   # módulo para processamento de dados
from model_training import AdvancedModelTrainer     # módulo para treinamento de modelos
from dashboard import AdvancedDashboard             # módulo para visualização em dashboard
from report_generator import PDFReportGenerator     # módulo para gerar relatórios em PDF

# Configura a página do Streamlit
st.set_page_config(
    page_title="Sistema Avançado de ML",   # título da aba do navegador
    page_icon="🚀",                        # ícone da aba
    layout="wide",                         # layout em tela cheia (wide)
    initial_sidebar_state="expanded"       # barra lateral expandida por padrão
)

# Insere CSS personalizado para estilizar a aplicação
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;          /* tamanho da fonte do cabeçalho principal */
        color: #1E88E5;             /* cor azul */
        text-align: center;         /* centraliza o texto */
        margin-bottom: 2rem;        /* espaçamento inferior */
    }
    .sub-header {
        font-size: 1.8rem;          /* tamanho da fonte do subtítulo */
        color: #0D47A1;             /* cor azul escura */
        margin-top: 1.5rem;         /* espaçamento superior */
        margin-bottom: 1rem;        /* espaçamento inferior */
    }
    .highlight-box {
        background-color: #E3F2FD;  /* cor de fundo clara */
        padding: 1rem;              /* espaçamento interno */
        border-radius: 10px;        /* bordas arredondadas */
        border-left: 5px solid #1E88E5; /* borda lateral azul */
        margin: 1rem 0;             /* espaçamento vertical */
    }
    .model-card {
        background-color: #F5F5F5;  /* cor de fundo cinza claro */
        padding: 1rem;              /* espaçamento interno */
        border-radius: 8px;         /* bordas arredondadas */
        margin: 0.5rem 0;           /* espaçamento vertical */
        border: 1px solid #E0E0E0;  /* borda cinza */
    }
    .best-model {
        background-color: #FFF8E1;  /* cor de fundo amarela clara */
        border: 2px solid #FFB300;  /* borda dourada */
        animation: pulse 2s infinite; /* animação pulsante */
    }
    @keyframes pulse {
        0% { border-color: #FFB300; }   /* início da animação */
        50% { border-color: #FFD54F; }  /* meio da animação */
        100% { border-color: #FFB300; } /* fim da animação */
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;  /* cor da barra de progresso */
    }
</style>
""", unsafe_allow_html=True)  # permite inserir HTML/CSS diretamente

# Define a classe principal do sistema de ML
class AdvancedMLSystem:
    # Método construtor da classe
    def __init__(self):
        # Inicializa variáveis internas como None
        self.data = None                # dados brutos
        self.processed_data = None      # dados processados
        self.results = None             # resultados dos modelos
        self.models = None              # modelos treinados
        self.best_model = None          # melhor modelo encontrado
        self.problem_type = None        # tipo de problema (classificação/regressão)
        self.feature_importance = None  # importância das variáveis

        # Inicializa variáveis de estado da sessão do Streamlit
        if 'processing_done' not in st.session_state:
            st.session_state.processing_done = False  # indica se o processamento foi concluído
        if 'training_done' not in st.session_state:
            st.session_state.training_done = False    # indica se o treinamento foi concluído
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1         # controla o passo atual do fluxo
    def run(self):
        """Executar o sistema completo"""   # Docstring explicando a função

        # Exibe o título principal do sistema com HTML customizado
        st.markdown('<h1 class="main-header">🚀 Sistema Avançado de Machine Learning</h1>', 
                   unsafe_allow_html=True)

        # Exibe uma caixa de destaque com descrição do sistema
        st.markdown("""
        <div class="highlight-box">
        <strong>Sistema Premium de IA:</strong> Carregue seu dataset CSV, o sistema automaticamente detectará 
        o tipo de problema (classificação ou regressão), realizará limpeza e tratamento avançado dos dados, 
        treinará mais de 30 modelos de machine learning com otimização automática, e apresentará um dashboard 
        completo com ranking dos modelos e relatório final em PDF.
        </div>
        """, unsafe_allow_html=True)
        
        # Renderiza a barra de progresso personalizada
        self.render_progress_bar()
        
        # Cria a barra lateral (sidebar) da aplicação
        with st.sidebar:
            # Exibe uma imagem na barra lateral
            st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", 
                    width=100)
            # Exibe o título da seção de configurações
            st.title("Configurações")
            
            # Componente para upload de arquivo CSV
            uploaded_file = st.file_uploader("📂 Upload do Dataset CSV", type=['csv'])
            
            # Se um arquivo foi carregado
            if uploaded_file:
                # Lê o arquivo CSV em um DataFrame
                self.data = pd.read_csv(uploaded_file)
                # Mensagem de sucesso com nome do arquivo
                st.success(f"Dataset carregado: {uploaded_file.name}")
                # Exibe o shape (linhas, colunas) do dataset
                st.info(f"Shape: {self.data.shape}")
                
                # Seleciona a coluna target (variável dependente)
                target_column = st.selectbox(
                    "🎯 Selecione a coluna target:",
                    options=self.data.columns.tolist(),   # lista de colunas do dataset
                    index=len(self.data.columns)-1        # por padrão, última coluna
                )
                
                # Seção de configurações avançadas
                with st.expander("⚙️ Configurações Avançadas"):
                    # Checkbox para detecção automática do tipo de problema
                    auto_detect = st.checkbox("Detecção automática do tipo de problema", value=True)
                    # Se não for automático, usuário escolhe manualmente
                    if not auto_detect:
                        problem_type = st.selectbox("Tipo de problema:", 
                                                   ["classification", "regression"])
                    else:
                        problem_type = "auto"
                    
                    # Checkbox para otimizar modelos com Optuna
                    optimize_models = st.checkbox("Otimizar modelos com Optuna", value=True)
                    # Slider para definir número de otimizações
                    n_optimizations = st.slider("Número de otimizações", 5, 50, 20)
                    
                    # Checkbox para criar ensemble dos melhores modelos
                    create_ensemble = st.checkbox("Criar ensemble dos melhores modelos", value=True)
                    
                # Botão para iniciar processamento completo
                if st.button("🚀 Iniciar Processamento Completo", 
                           type="primary", 
                           use_container_width=True):
                    # Exibe spinner de carregamento
                    with st.spinner("Iniciando processamento..."):
                        # Chama método para processar os dados
                        self.process_data(uploaded_file.name, target_column, problem_type)
                        # Atualiza estado da sessão indicando que processamento foi concluído
                        st.session_state.processing_done = True
                        # Atualiza passo atual para 2 (análise de dados)
                        st.session_state.current_step = 2
                        # Recarrega a aplicação para refletir mudanças
                        st.rerun()
        
        # Renderiza conteúdo principal dependendo do passo atual
        if st.session_state.current_step == 1:
            self.render_welcome_screen()      # Tela inicial
        elif st.session_state.current_step == 2:
            self.render_data_analysis()       # Análise dos dados
        elif st.session_state.current_step == 3:
            self.render_model_training()      # Treinamento dos modelos
        elif st.session_state.current_step == 4:
            self.render_results_dashboard()   # Dashboard de resultados
    
    def render_progress_bar(self):
        """Renderizar barra de progresso"""   # Docstring explicando a função

        # Lista de etapas do fluxo
        steps = ["Upload", "Análise", "Treino", "Resultados"]
        # Calcula etapa atual (ajustando índice)
        current = st.session_state.current_step - 1
        
        # Cria colunas para cada etapa
        cols = st.columns(len(steps))
        # Itera sobre as etapas e colunas
        for i, col in enumerate(cols):
            with col:
                # Se etapa já concluída
                if i < current:
                    st.success(f"✅ {steps[i]}")
                # Se etapa atual em andamento
                elif i == current:
                    st.info(f"⏳ {steps[i]}")
                # Se etapa futura ainda não iniciada
                else:
                    st.warning(f"⏳ {steps[i]}")
        
        # Calcula progresso percentual
        progress = current / (len(steps) - 1)
        # Exibe barra de progresso
        st.progress(progress)
    def render_welcome_screen(self):
        """Tela inicial"""   # Docstring descreve que este método renderiza a tela inicial

        # Cria três colunas na página, com proporções 1:2:1
        col1, col2, col3 = st.columns([1, 2, 1])
        
        # Usa a coluna do meio (col2) para centralizar o conteúdo
        with col2:
            # Exibe um bloco de texto em Markdown com as funcionalidades do sistema
            st.markdown("""
            ## 📋 Funcionalidades do Sistema
            
            ### 🔍 **Análise e Processamento de Dados**
            - Detecção automática do tipo de problema
            - Limpeza avançada: outliers, missing values, duplicatas
            - Engenharia de features automática
            - Codificação inteligente de variáveis categóricas
            - Normalização e padronização
            
            ### 🤖 **Machine Learning Avançado**
            - **30+ Modelos** incluindo:
            - XGBoost, LightGBM, CatBoost
            - Random Forest, Gradient Boosting
            - SVM, Redes Neurais, KNN
            - Ensemble personalizado
            - Otimização automática com **Optuna**
            - Validação cruzada avançada
            - Seleção de features
            
            ### 📊 **Visualização e Relatórios**
            - Dashboard interativo completo
            - Ranking dos modelos
            - Análise de importância das features
            - Relatório PDF profissional
            - Exportação de resultados
            """)
            
            # Exibe instruções de uso do sistema em formato de lista
            st.markdown("""
            ### 🎯 **Como usar:**
            1. Faça upload do seu dataset CSV
            2. Selecione a coluna target
            3. Ajuste configurações (opcional)
            4. Clique em "Iniciar Processamento"
            5. Explore os resultados no dashboard
            6. Baixe o relatório PDF
            """)

    # Método para processar os dados
    def process_data(self, filename, target_column, problem_type):
        """Processar dados"""   # Docstring descreve que este método processa os dados

        # Exibe uma notificação (toast) informando que o processamento começou
        st.toast("🚀 Iniciando processamento dos dados...")
        
        # Inicializa o processador de dados, passando coluna alvo e tipo de problema
        processor = AdvancedDataProcessor(target_column=target_column, 
                                        problem_type=problem_type)
        
        # Processa os dados convertendo o DataFrame em CSV e depois em bytes
        X, y, detected_problem_type = processor.process(io.BytesIO(self.data.to_csv().encode()))
        
        # Armazena os dados processados e informações adicionais em um dicionário
        self.processed_data = {
            'X': X,                                # Features processadas
            'y': y,                                # Target processado
            'problem_type': detected_problem_type, # Tipo de problema detectado
            'processor': processor                 # Objeto processador usado
        }
        
        # Atualiza o atributo da classe com o tipo de problema detectado
        self.problem_type = detected_problem_type
        
        # Exibe uma notificação (toast) informando que os dados foram processados
        st.toast(f"✅ Dados processados! Tipo detectado: {detected_problem_type}")
    def render_data_analysis(self):
        """Mostrar análise dos dados"""   # Docstring explicando que este método mostra a análise exploratória dos dados

        # Exibe um título em HTML customizado para a seção de análise
        st.markdown('<h2 class="sub-header">📈 Análise Exploratória dos Dados</h2>', 
                unsafe_allow_html=True)
        
        # Verifica se os dados foram carregados
        if self.data is not None:
            # Cria abas (tabs) para diferentes tipos de análise
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Visão Geral", 
                "📊 Estatísticas", 
                "🔍 Distribuições", 
                "📈 Correlações"
            ])
            
            # Conteúdo da aba "Visão Geral"
            with tab1:
                # Cria duas colunas lado a lado
                col1, col2 = st.columns(2)
                with col1:
                    # Exibe as primeiras linhas do dataset
                    st.write("**Primeiras linhas:**")
                    st.dataframe(self.data.head(), use_container_width=True)
                with col2:
                    # Exibe as últimas linhas do dataset
                    st.write("**Últimas linhas:**")
                    st.dataframe(self.data.tail(), use_container_width=True)
                
                # Cria mais duas colunas lado a lado
                col3, col4 = st.columns(2)
                with col3:
                    # Exibe informações gerais do dataset (tipos, memória, etc.)
                    st.write("**Informações do Dataset:**")
                    buffer = io.StringIO()              # Cria um buffer de texto
                    self.data.info(buf=buffer)          # Captura saída do método info()
                    st.text(buffer.getvalue())          # Exibe o conteúdo capturado
                with col4:
                    # Exibe valores ausentes por coluna
                    st.write("**Valores ausentes:**")
                    missing_df = pd.DataFrame({
                        'Coluna': self.data.columns,
                        'Valores Ausentes': self.data.isnull().sum().values,
                        'Percentual': (self.data.isnull().sum() / len(self.data) * 100).values
                    })
                    st.dataframe(missing_df, use_container_width=True)
            
            # Conteúdo da aba "Estatísticas"
            with tab2:
                # Exibe estatísticas descritivas (média, desvio padrão, etc.)
                st.write("**Estatísticas Descritivas:**")
                st.dataframe(self.data.describe(), use_container_width=True)
                
                # Exibe tipos de dados presentes no dataset
                st.write("**Tipos de Dados:**")
                dtype_df = pd.DataFrame(self.data.dtypes.value_counts()).reset_index()
                dtype_df.columns = ['Tipo', 'Quantidade']
                st.dataframe(dtype_df, use_container_width=True)
            
            # Conteúdo da aba "Distribuições"
            with tab3:
                # Seleciona colunas numéricas
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # Permite escolher uma coluna numérica para histograma
                    selected_col = st.selectbox("Selecione coluna para histograma:", numeric_cols)
                    fig = px.histogram(self.data, x=selected_col, title=f"Distribuição de {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Seleciona colunas categóricas
                categorical_cols = self.data.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    # Permite escolher uma coluna categórica para gráfico de barras
                    selected_cat = st.selectbox("Selecione coluna categórica:", categorical_cols)
                    value_counts = self.data[selected_cat].value_counts().head(10)   # Top 10 valores
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                                title=f"Top 10 valores em {selected_cat}")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Conteúdo da aba "Correlação"
            with tab4:
                # Seleciona apenas colunas numéricas
                numeric_data = self.data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    # Calcula matriz de correlação
                    corr_matrix = numeric_data.corr()
                    # Cria heatmap da matriz de correlação
                    fig = px.imshow(corr_matrix, 
                                title="Matriz de Correlação",
                                color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Exibe correlações mais fortes
                    st.write("**Correlações mais fortes:**")
                    corr_pairs = corr_matrix.unstack()   # Transforma matriz em pares
                    sorted_pairs = corr_pairs.sort_values(key=abs, ascending=False)  # Ordena por valor absoluto
                    # Remove auto-correlações (variável com ela mesma)
                    sorted_pairs = sorted_pairs[sorted_pairs.index.get_level_values(0) != 
                                            sorted_pairs.index.get_level_values(1)]
                    # Seleciona top 10 correlações
                    top_corr = pd.DataFrame(sorted_pairs.head(10)).reset_index()
                    top_corr.columns = ['Variável 1', 'Variável 2', 'Correlação']
                    st.dataframe(top_corr, use_container_width=True)
            
            # Botão para iniciar treinamento dos modelos
            if st.button("▶️ Iniciar Treinamento dos Modelos", 
                        type="primary",
                        use_container_width=True):
                st.session_state.current_step = 3   # Atualiza passo atual para "treinamento"
                st.rerun()                          # Recarrega a aplicação para refletir mudança
    def render_model_training(self):
        """Interface de treinamento dos modelos"""   # Docstring explicando que este método renderiza a interface de treinamento

        # Exibe o título da seção de treinamento com HTML customizado
        st.markdown('<h2 class="sub-header">🤖 Treinamento Avançado de Modelos</h2>', 
                unsafe_allow_html=True)
        
        # Verifica se os dados já foram processados
        if self.processed_data:
            # Cria um bloco expansível para mostrar informações do processamento
            with st.expander("📋 Informações do Processamento", expanded=True):
                # Exibe o tipo de problema detectado (classificação ou regressão)
                st.write(f"**Tipo de problema:** {self.problem_type}")
                # Exibe o número de features (colunas) do dataset processado
                st.write(f"**Número de features:** {self.processed_data['X'].shape[1]}")
                # Exibe o número de amostras (linhas) do dataset processado
                st.write(f"**Número de amostras:** {self.processed_data['X'].shape[0]}")
                
                # Se o target tiver o método nunique (para contar classes distintas)
                if hasattr(self.processed_data['y'], 'nunique'):
                    # Exibe o número de classes no target
                    st.write(f"**Classes no target:** {self.processed_data['y'].nunique()}")
            
            # Cria uma barra de progresso inicializada em 0
            progress_bar = st.progress(0)
            # Cria um espaço vazio para exibir mensagens de status
            status_text = st.empty()
            
            # Botão para iniciar o treinamento completo dos modelos
            if st.button("🎯 Executar Treinamento Completo", 
                        type="primary",
                        use_container_width=True):
                
                # Simula progresso de 0 a 100%
                for i in range(100):
                    progress_bar.progress(i + 1)   # Atualiza barra de progresso
                    status_text.text(f"Treinando modelos... {i+1}%")  # Atualiza texto de status
                    time.sleep(0.01)  # Pausa curta para simulação visual
                
                # Executa o treinamento real dos modelos com spinner de carregamento
                with st.spinner("Treinando modelos..."):
                    # Inicializa o treinador de modelos passando o tipo de problema
                    trainer = AdvancedModelTrainer(self.problem_type)
                    # Treina os modelos e obtém resultados e melhor modelo
                    self.results, self.best_model = trainer.train_models(
                        self.processed_data['X'], 
                        self.processed_data['y']
                    )
                    # Armazena todos os modelos treinados
                    self.models = trainer.models
                    # Armazena a importância das features calculada
                    self.feature_importance = trainer.feature_importance
                    
                    # Salva os modelos treinados em uma pasta
                    trainer.save_models('saved_models/')
                
                # Atualiza estado da sessão indicando que treinamento foi concluído
                st.session_state.training_done = True
                # Atualiza passo atual para 4 (resultados)
                st.session_state.current_step = 4
                # Recarrega a aplicação para refletir mudanças
                st.rerun()
    # Método responsável por renderizar todo o dashboard de resultados no Streamlit
    def render_results_dashboard(self):

            # Docstring explicativa
            """Dashboard de resultados"""

            # Exibe um título HTML estilizado no dashboard
            st.markdown(
                '<h2 class="sub-header">📊 Dashboard de Resultados</h2>', 
                unsafe_allow_html=True
            )
            
            # Verifica se existem resultados calculados e modelos treinados
            if self.results and self.models:

                # Calcula qual é o melhor modelo com base na métrica principal
                best_model_name = max(
                    self.results.items(), 
                    key=lambda x: self._get_primary_metric(x[1])
                )[0]

                # Recupera o dicionário de métricas do melhor modelo
                best_metrics = self.results[best_model_name]
                
                # Cria quatro colunas para exibir métricas em formato de cards
                col1, col2, col3, col4 = st.columns(4)
                
                # Primeira coluna: nome do melhor modelo
                with col1:
                    st.metric("🏆 Melhor Modelo", best_model_name)
                
                # Segunda coluna: métrica principal (F1 ou R²)
                with col2:
                    if self.problem_type == 'classification':
                        st.metric("📈 F1-Score", f"{best_metrics.get('f1', 0):.4f}")
                    else:
                        st.metric("📈 R² Score", f"{best_metrics.get('r2', 0):.4f}")
                
                # Terceira coluna: acurácia ou RMSE
                with col3:
                    if self.problem_type == 'classification':
                        st.metric("🎯 Acurácia", f"{best_metrics.get('accuracy', 0):.4f}")
                    else:
                        st.metric("🎯 RMSE", f"{best_metrics.get('rmse', 0):.4f}")
                
                # Quarta coluna: total de modelos treinados
                with col4:
                    st.metric("🤖 Total Modelos", len(self.models))
                
                # Cria abas para organizar diferentes visualizações
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🏆 Ranking", 
                    "📊 Comparação", 
                    "🔍 Detalhes", 
                    "📥 Exportar"
                ])
                
                # Aba de ranking dos modelos
                with tab1:

                    # Título da seção de ranking
                    st.markdown("### Ranking dos Modelos (do melhor para o pior)")
                    
                    # Ordena os modelos com base na métrica principal
                    sorted_results = sorted(
                        self.results.items(), 
                        key=lambda x: self._get_primary_metric(x[1]), 
                        reverse=True
                    )
                    
                    # Lista que armazenará os dados do ranking
                    ranking_data = []

                    # Percorre os modelos ordenados e cria o ranking
                    for i, (model_name, metrics) in enumerate(sorted_results, 1):
                        ranking_data.append({
                            'Posição': i,
                            'Modelo': model_name,
                            'Métrica Principal': self._get_primary_metric(metrics),
                            'Status': '⭐ RECOMENDADO' if model_name == best_model_name else ''
                        })
                    
                    # Converte os dados do ranking em DataFrame
                    ranking_df = pd.DataFrame(ranking_data)

                    # Exibe a tabela de ranking
                    st.dataframe(ranking_df, use_container_width=True)
                    
                    # Cria gráfico de barras horizontal para o ranking
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[d['Métrica Principal'] for d in ranking_data],
                            y=[d['Modelo'] for d in ranking_data],
                            orientation='h',
                            marker_color=[
                                '#FFD700' if d['Status'] == '⭐ RECOMENDADO' 
                                else '#1E88E5' 
                                for d in ranking_data
                            ]
                        )
                    ])
                    
                    # Configura o layout do gráfico
                    fig.update_layout(
                        title="Ranking dos Modelos",
                        xaxis_title="Métrica Principal",
                        yaxis_title="Modelo",
                        height=500
                    )
                    
                    # Exibe o gráfico no Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                
                # Aba de comparação entre métricas
                with tab2:

                    # Título da seção de comparação
                    st.markdown("### Comparação de Métricas por Modelo")
                    
                    # Converte os resultados em DataFrame (modelos nas linhas)
                    metrics_df = pd.DataFrame(self.results).T
                    
                    # Seleção de métricas para classificação
                    if self.problem_type == 'classification':
                        selected_metrics = st.multiselect(
                            "Selecione métricas:",
                            options=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                            default=['accuracy', 'f1']
                        )
                    else:
                        # Seleção de métricas para regressão
                        selected_metrics = st.multiselect(
                            "Selecione métricas:",
                            options=['r2', 'rmse', 'mae', 'mape'],
                            default=['r2', 'rmse']
                        )
                    
                    # Só cria gráficos se houver métricas selecionadas
                    if selected_metrics:

                        # Inicializa o gráfico
                        fig = go.Figure()
                        
                        # Adiciona uma barra para cada métrica selecionada
                        for metric in selected_metrics:
                            fig.add_trace(go.Bar(
                                x=metrics_df.index,
                                y=metrics_df[metric],
                                name=metric.upper()
                            ))
                        
                        # Configura layout do gráfico
                        fig.update_layout(
                            title="Comparação de Métricas",
                            barmode='group',
                            height=500
                        )
                        
                        # Exibe o gráfico
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Título do heatmap
                        st.markdown("### Heatmap de Similaridade entre Modelos")

                        # Seleciona apenas métricas numéricas
                        numeric_metrics = metrics_df.select_dtypes(include=[np.number])

                        # Calcula correlação entre modelos
                        corr_matrix = numeric_metrics.T.corr()
                        
                        # Cria heatmap de correlação
                        fig2 = px.imshow(
                            corr_matrix,
                            color_continuous_scale='RdBu_r',
                            title="Correlação entre Desempenhos dos Modelos"
                        )

                        # Exibe o heatmap
                        st.plotly_chart(fig2, use_container_width=True)
                
                # Aba de detalhes individuais
                with tab3:

                    # Título da seção
                    st.markdown("### Detalhes por Modelo")
                    
                    # Dropdown para selecionar modelo
                    selected_model = st.selectbox(
                        "Selecione um modelo para detalhes:",
                        options=list(self.results.keys())
                    )
                    
                    # Verifica se um modelo foi selecionado
                    if selected_model:

                        # Recupera métricas do modelo selecionado
                        metrics = self.results[selected_model]
                        
                        # Cria duas colunas
                        col1, col2 = st.columns(2)
                        
                        # Coluna de métricas
                        with col1:
                            # Adiciona um cabeçalho para a seção de métricas
                            st.markdown("#### Métricas")
                            # Itera sobre as métricas disponíveis no dicionário de métricas
                            for metric_name, value in metrics.items():
                                # Verifica se a métrica não é a matriz de confusão
                                if metric_name != 'confusion_matrix':
                                    # Exibe a métrica no formato de cartão com o nome e valor formatado
                                    st.metric(metric_name.upper(), f"{value:.4f}")
                        
                        # Coluna de informações do modelo
                        with col2:
                            # Verifica se o modelo selecionado está na lista de modelos treinados
                            if selected_model in self.models:
                                # Recupera o modelo selecionado da lista de modelos treinados
                                model = self.models[selected_model]
                                # Exibe um título para a seção de informações do modelo
                                st.markdown("#### Informações do Modelo")

                                # Exibe os parâmetros do modelo selecionado
                                st.write("**Parâmetros:**")
                                # Obtém os parâmetros do modelo como um dicionário
                                params = model.get_params()
                                # Itera sobre os primeiros 10 parâmetros do modelo e os exibe
                                for param, value in list(params.items())[:10]:
                                    st.text(f"{param}: {value}")
                
                # Aba de exportação
                with tab4:

                    # Título da aba
                    st.markdown("### 📥 Exportação de Resultados")
                    
                    # Cria três colunas
                    col1, col2, col3 = st.columns(3)
                    
                    # Botão para gerar PDF
                    with col1:
                        if st.button("📄 Gerar Relatório PDF", use_container_width=True):

                            # Instancia o gerador de relatório PDF com os parâmetros necessários
                            report_gen = PDFReportGenerator(
                                self.results,  # Resultados dos modelos treinados
                                self.models,  # Modelos treinados
                                best_model_name,  # Nome do melhor modelo
                                self.problem_type,  # Tipo de problema (classificação ou regressão)
                                {
                                    'dataset_name': 'Dataset Processado',  # Nome do dataset processado
                                    'n_samples': self.processed_data['X'].shape[0],  # Número de amostras no dataset
                                    'n_features': self.processed_data['X'].shape[1]  # Número de features no dataset
                                }
                            )
                            
                            # Gera o arquivo PDF
                            report_file = report_gen.generate_report("relatorio_final.pdf")
                            
                            # Lê o PDF em binário
                            with open(report_file, "rb") as f:
                                pdf_data = f.read()
                            
                            # Converte o PDF para base64
                            b64 = base64.b64encode(pdf_data).decode()

                            # Cria link de download
                            href = f'<a href="data:application/pdf;base64,{b64}" download="relatorio_ml.pdf">Clique para baixar o relatório PDF</a>'
                            st.markdown(href, unsafe_allow_html=True)

                            # Mensagem de sucesso
                            st.success("✅ Relatório PDF gerado com sucesso!")
                    
                    # Botão para exportar CSV
                    with col2:
                        if st.button("💾 Exportar Resultados CSV", use_container_width=True):

                            # Converte resultados em DataFrame
                            results_df = pd.DataFrame(self.results).T

                            # Converte DataFrame em CSV
                            csv = results_df.to_csv()
                            
                            # Converte CSV para base64
                            b64 = base64.b64encode(csv.encode()).decode()

                            # Cria link de download
                            href = f'<a href="data:file/csv;base64,{b64}" download="resultados_ml.csv">Clique para baixar o CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)

                            # Mensagem de sucesso
                            st.success("✅ CSV exportado com sucesso!")
                    
                    # Botão para salvar o melhor modelo
                    with col3:
                        if st.button("🤖 Salvar Melhor Modelo", use_container_width=True):

                            # Verifica se o melhor modelo existe
                            if best_model_name in self.models:

                                # Recupera o modelo
                                model = self.models[best_model_name]

                                # Salva o modelo em arquivo pickle
                                joblib.dump(model, 'melhor_modelo.pkl')
                                
                                # Lê o modelo salvo
                                with open('melhor_modelo.pkl', "rb") as f:
                                    model_data = f.read()
                                
                                # Converte modelo para base64
                                b64 = base64.b64encode(model_data).decode()

                                # Cria link de download
                                href = f'<a href="data:application/octet-stream;base64,{b64}" download="melhor_modelo.pkl">Clique para baixar o modelo</a>'
                                st.markdown(href, unsafe_allow_html=True)

                                # Mensagem de sucesso
                                st.success("✅ Modelo salvo com sucesso!")
                
                # Botão para reiniciar o sistema
                if st.button(
                    "🔄 Reiniciar Sistema",  # Texto do botão
                    type="secondary",       # Tipo do botão (secundário)
                    use_container_width=True # Define que o botão usa a largura do container
                ):
                    # Reseta o passo atual para o primeiro (tela inicial)
                    st.session_state.current_step = 1
                    # Marca que o processamento não foi concluído
                    st.session_state.processing_done = False
                    # Marca que o treinamento não foi concluído
                    st.session_state.training_done = False
                    # Recarrega a aplicação para refletir as mudanças
                    st.rerun()
    

    # Método auxiliar para definir qual métrica usar na ordenação
    def _get_primary_metric(self, metrics):

        # Docstring explicativa
        """Obter métrica principal para ordenação"""

        # Para classificação, usa F1-score
        if self.problem_type == 'classification':
            return metrics.get('f1', 0)
        else:
            # Para regressão, usa RMSE negativo (menor é melhor)
            return -metrics.get('rmse', 0)

# Ponto de entrada do script
if __name__ == "__main__":

    # Cria a instância principal do sistema
    system = AdvancedMLSystem()

    # Executa o sistema
    system.run()
