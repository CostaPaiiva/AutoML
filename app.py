import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import base64
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURAÇÃO ==========
st.set_page_config(
    page_title="AutoML Completo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS PERSONALIZADO ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stButton > button {
        width: 100%;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========== PROCESSAMENTO DE DADOS SIMPLES ==========
class SimpleDataProcessor:
    def __init__(self, target_column=None):
        self.target_column = target_column
    
    def process(self, data):
        """Processamento simples e à prova de erros"""
        try:
            # Se não tiver target, usar última coluna
            if self.target_column is None:
                self.target_column = data.columns[-1]
            
            # Verificar se target existe
            if self.target_column not in data.columns:
                st.error(f"Coluna '{self.target_column}' não encontrada no dataset.")
                # Usar última coluna como fallback
                self.target_column = data.columns[-1]
            
            # Separar X e y
            X = data.drop(columns=[self.target_column]).copy()
            y = data[self.target_column].copy()
            
            # Detectar tipo de problema
            if y.dtype == 'object' or len(y.unique()) <= 10:
                problem_type = 'classification'
            else:
                problem_type = 'regression'
            
            st.info(f"✅ Tipo de problema detectado: **{problem_type.upper()}**")
            
            # 1. Limpeza básica
            X_clean = self.clean_data(X)
            
            # 2. Codificar categóricas
            X_encoded = self.encode_categorical(X_clean)
            
            # 3. Lidar com missing values
            X_final = self.handle_missing(X_encoded)
            
            # 4. Escalar features (opcional, apenas se solicitado)
            if st.session_state.get('scale_features', True):
                X_final = self.scale_features(X_final)
            
            return X_final, y, problem_type
            
        except Exception as e:
            st.error(f"❌ Erro no processamento: {str(e)}")
            # Fallback: processamento mínimo
            return self.minimal_process(data)
    
    def clean_data(self, X):
        """Limpeza básica dos dados"""
        # Remover colunas com muitos missing (>50%)
        missing_threshold = 0.5
        missing_pct = X.isnull().mean()
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            st.info(f"📉 Removidas {len(cols_to_drop)} colunas com muitos valores faltantes")
        
        # Remover colunas constantes
        constant_cols = [col for col in X.columns if X[col].nunique() == 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)
            st.info(f"⚡ Removidas {len(constant_cols)} colunas constantes")
        
        return X
    
    def encode_categorical(self, X):
        """Codificação de variáveis categóricas"""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Se tiver poucas categorias, usar one-hot
            if X[col].nunique() <= 10:
                # One-hot encoding
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            else:
                # Label encoding para muitas categorias
                X[col] = pd.factorize(X[col])[0]
        
        return X
    
    def handle_missing(self, X):
        """Tratamento de valores faltantes"""
        # Para colunas numéricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Usar mediana (mais robusta que média)
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Para colunas não numéricas (após encoding)
        other_cols = X.select_dtypes(exclude=[np.number]).columns
        for col in other_cols:
            X[col] = X[col].fillna(0)
        
        return X
    
    def scale_features(self, X):
        """Escalonamento simples de features"""
        from sklearn.preprocessing import StandardScaler
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Apenas scale colunas com desvio padrão > 0
            cols_to_scale = [col for col in numeric_cols if X[col].std() > 0]
            
            if cols_to_scale:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X[cols_to_scale])
                X[cols_to_scale] = X_scaled
                st.info(f"📊 {len(cols_to_scale)} features escaladas")
        
        return X
    
    def minimal_process(self, data):
        """Processamento mínimo de fallback"""
        # Target é a última coluna
        target_col = data.columns[-1]
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Detectar tipo
        if y.dtype == 'object' or len(y.unique()) <= 10:
            problem_type = 'classification'
        else:
            problem_type = 'regression'
        
        # Apenas fillna
        X = X.fillna(0)
        
        return X, y, problem_type

# ========== TREINAMENTO DE MODELOS ==========
class SimpleModelTrainer:
    def __init__(self, problem_type):
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = ""
    
    def train_models(self, X, y):
        """Treina vários modelos de ML"""
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
        
        st.info("🤖 Iniciando treinamento de modelos...")
        
        # Split dos dados
        if self.problem_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Obter modelos
        models = self.get_models()
        
        # Barra de progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_list = []
        
        for i, (name, model) in enumerate(models.items()):
            try:
                status_text.text(f"📊 Treinando {name}...")
                
                # Validação cruzada simples
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=3, 
                    scoring='accuracy' if self.problem_type == 'classification' else 'r2',
                    n_jobs=-1
                )
                
                # Treinar modelo
                model.fit(X_train, y_train)
                
                # Previsões
                y_pred = model.predict(X_test)
                
                # Métricas
                if self.problem_type == 'classification':
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred, average='weighted'),
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                else:
                    metrics = {
                        'r2': r2_score(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                
                # Salvar
                self.models[name] = model
                self.results[name] = metrics
                results_list.append((name, metrics))
                
                # Atualizar progresso
                progress_bar.progress((i + 1) / len(models))
                
            except Exception as e:
                st.warning(f"⚠️ {name}: {str(e)}")
                continue
        
        # Determinar melhor modelo
        if self.results:
            self.determine_best_model()
            st.success(f"✅ Treinamento completo! {len(self.results)} modelos treinados")
            st.success(f"🏆 Melhor modelo: **{self.best_model_name}**")
        else:
            st.error("❌ Nenhum modelo foi treinado com sucesso!")
        
        return self.results, self.best_model_name
    
    def get_models(self):
        """Retorna lista de modelos para treinar"""
        if self.problem_type == 'classification':
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import (
                RandomForestClassifier, GradientBoostingClassifier,
                AdaBoostClassifier
            )
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.naive_bayes import GaussianNB
            
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'K-Neighbors': KNeighborsClassifier(n_jobs=-1),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'Naive Bayes': GaussianNB()
            }
        else:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.ensemble import (
                RandomForestRegressor, GradientBoostingRegressor,
                AdaBoostRegressor
            )
            from sklearn.svm import SVR
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.tree import DecisionTreeRegressor
            
            models = {
                'Linear Regression': LinearRegression(n_jobs=-1),
                'Ridge Regression': Ridge(random_state=42),
                'Lasso Regression': Lasso(random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'K-Neighbors': KNeighborsRegressor(n_jobs=-1),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'AdaBoost': AdaBoostRegressor(random_state=42)
            }
        
        return models
    
    def determine_best_model(self):
        """Determina o melhor modelo baseado nas métricas"""
        if not self.results:
            return
        
        if self.problem_type == 'classification':
            # Ordenar por accuracy
            sorted_models = sorted(self.results.items(), 
                                  key=lambda x: x[1]['accuracy'], 
                                  reverse=True)
        else:
            # Ordenar por r2
            sorted_models = sorted(self.results.items(), 
                                  key=lambda x: x[1]['r2'], 
                                  reverse=True)
        
        self.best_model_name = sorted_models[0][0]
        self.best_model = self.models[self.best_model_name]
    
    def get_ranking(self):
        """Retorna ranking dos modelos como DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        ranking_data = []
        for name, metrics in self.results.items():
            if self.problem_type == 'classification':
                ranking_data.append({
                    'Modelo': name,
                    'Acurácia': f"{metrics['accuracy']:.4f}",
                    'F1-Score': f"{metrics['f1_score']:.4f}",
                    'CV Score': f"{metrics['cv_mean']:.4f}"
                })
            else:
                ranking_data.append({
                    'Modelo': name,
                    'R²': f"{metrics['r2']:.4f}",
                    'RMSE': f"{metrics['rmse']:.4f}",
                    'CV Score': f"{metrics['cv_mean']:.4f}"
                })
        
        df = pd.DataFrame(ranking_data)
        
        # Ordenar
        sort_col = 'Acurácia' if self.problem_type == 'classification' else 'R²'
        df = df.sort_values(sort_col, ascending=False)
        df.insert(0, 'Posição', range(1, len(df) + 1))
        
        return df

# ========== APLICAÇÃO PRINCIPAL ==========
class AutoMLApp:
    def __init__(self):
        # Inicializar estado da sessão
        if 'step' not in st.session_state:
            st.session_state.step = 1
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'processed' not in st.session_state:
            st.session_state.processed = False
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'scale_features' not in st.session_state:
            st.session_state.scale_features = True
    
    def run(self):
        """Executa a aplicação completa"""
        # Cabeçalho
        st.markdown('<h1 class="main-header">🚀 AutoML Completo - Sistema Inteligente</h1>', 
                   unsafe_allow_html=True)
        
        # Barra de progresso
        self.show_progress()
        
        # Conteúdo por passo
        if st.session_state.step == 1:
            self.step_upload()
        elif st.session_state.step == 2:
            self.step_process()
        elif st.session_state.step == 3:
            self.step_train()
        elif st.session_state.step == 4:
            self.step_results()
    
    def show_progress(self):
        """Mostra barra de progresso"""
        steps = ["📥 Upload", "🔧 Processar", "🤖 Treinar", "📊 Resultados"]
        current = st.session_state.step - 1
        
        cols = st.columns(len(steps))
        for i, col in enumerate(cols):
            with col:
                if i < current:
                    st.success(f"✅ {steps[i]}")
                elif i == current:
                    st.info(f"⏳ {steps[i]}")
                else:
                    st.write(f"📌 {steps[i]}")
        
        st.progress(current / (len(steps) - 1))
    
    def step_upload(self):
        """Passo 1: Upload do dataset"""
        st.markdown("## 📥 Upload do Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Escolha um arquivo CSV", 
                type=['csv'],
                help="Faça upload do seu dataset em formato CSV"
            )
            
            if uploaded_file:
                try:
                    # Ler o arquivo
                    st.session_state.data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Dataset carregado com sucesso!")
                    
                    # Mostrar informações
                    st.write(f"**Formato:** {st.session_state.data.shape[0]} linhas × {st.session_state.data.shape[1]} colunas")
                    
                    # Mostrar preview
                    with st.expander("📋 Visualizar primeiras linhas"):
                        st.dataframe(st.session_state.data.head(), use_container_width=True)
                    
                    # Informações do dataset
                    with st.expander("📊 Informações do dataset"):
                        buffer = io.StringIO()
                        st.session_state.data.info(buf=buffer)
                        st.text(buffer.getvalue())
                        
                        # Valores faltantes
                        missing = st.session_state.data.isnull().sum()
                        if missing.sum() > 0:
                            st.warning(f"⚠️ {missing.sum()} valores faltantes encontrados")
                    
                    # Selecionar target
                    target_col = st.selectbox(
                        "🎯 Selecione a coluna target (variável a ser prevista):",
                        options=st.session_state.data.columns.tolist(),
                        index=len(st.session_state.data.columns) - 1,
                        help="Esta é a variável que os modelos vão tentar prever"
                    )
                    
                    st.session_state.target_col = target_col
                    
                    # Configurações opcionais
                    with st.expander("⚙️ Configurações avançadas"):
                        st.session_state.scale_features = st.checkbox(
                            "Escalar features automaticamente", 
                            value=True,
                            help="Normaliza as features para melhor performance dos modelos"
                        )
                    
                    # Botão para continuar
                    if st.button("▶️ Processar Dados", type="primary", use_container_width=True):
                        st.session_state.step = 2
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Erro ao ler arquivo: {str(e)}")
        
        with col2:
            st.markdown("""
            ### 📋 Como Funciona
            
            1. **Upload CSV**
               - Qualquer dataset em formato CSV
               - Processamento automático
            
            2. **Processamento**
               - Limpeza de dados
               - Codificação automática
               - Tratamento de valores faltantes
            
            3. **Treinamento**
               - 7+ algoritmos de ML
               - Validação cruzada
               - Seleção do melhor modelo
            
            4. **Resultados**
               - Ranking completo
               - Dashboard interativo
               - Exportação de resultados
            
            ### 🎯 Tipos Suportados
            
            • **Classificação**
              - Previsão de categorias
              - Ex: spam/não spam
            
            • **Regressão**
              - Previsão de valores numéricos
              - Ex: preços, temperaturas
            """)
    
    def step_process(self):
        """Passo 2: Processamento dos dados"""
        st.markdown("## 🔧 Processamento de Dados")
        
        if st.session_state.data is None:
            st.warning("⚠️ Nenhum dataset carregado.")
            if st.button("⬅️ Voltar para Upload"):
                st.session_state.step = 1
                st.rerun()
            return
        
        # Processar dados
        with st.spinner("Processando dados..."):
            processor = SimpleDataProcessor(target_column=st.session_state.target_col)
            
            # Adicionar configuração de scaling
            processor.scale_features_enabled = st.session_state.scale_features
            
            X, y, problem_type = processor.process(st.session_state.data)
            
            # Salvar no estado
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.problem_type = problem_type
            st.session_state.processed = True
        
        # Mostrar resultados do processamento
        st.success("✅ Processamento concluído!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tipo de Problema", problem_type.upper())
        
        with col2:
            st.metric("Features", X.shape[1])
        
        with col3:
            st.metric("Amostras", X.shape[0])
        
        # Mostrar informações dos dados processados
        with st.expander("📊 Dados Processados"):
            tab1, tab2 = st.tabs(["📋 Amostra", "📈 Estatísticas"])
            
            with tab1:
                st.write("**Primeiras 5 linhas das features:**")
                st.dataframe(X.head(), use_container_width=True)
                
                st.write("**Primeiras 5 valores do target:**")
                st.dataframe(y.head().to_frame(), use_container_width=True)
            
            with tab2:
                st.write("**Estatísticas das features:**")
                st.dataframe(X.describe(), use_container_width=True)
        
        # Botões de navegação
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Voltar", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        
        with col3:
            if st.button("🤖 Iniciar Treinamento", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
    
    def step_train(self):
        """Passo 3: Treinamento dos modelos"""
        st.markdown("## 🤖 Treinamento de Modelos")
        
        if not st.session_state.processed:
            st.warning("⚠️ Dados não processados.")
            st.session_state.step = 2
            st.rerun()
            return
        
        X = st.session_state.X
        y = st.session_state.y
        problem_type = st.session_state.problem_type
        
        # Informações sobre o treinamento
        st.info(f"""
        **Configuração do Treinamento:**
        - Tipo: {problem_type.upper()}
        - Features: {X.shape[1]}
        - Amostras: {X.shape[0]}
        - Modelos: 7 algoritmos diferentes
        - Validação: 3-fold cross-validation
        """)
        
        # Iniciar treinamento
        if st.button("🚀 Iniciar Treinamento Completo", type="primary", use_container_width=True):
            with st.spinner("Treinando modelos... Isso pode levar alguns minutos"):
                # Criar e treinar modelos
                trainer = SimpleModelTrainer(problem_type)
                results, best_model = trainer.train_models(X, y)
                
                # Salvar resultados
                st.session_state.results = results
                st.session_state.trainer = trainer
                st.session_state.best_model = best_model
                
                # Ir para resultados
                st.session_state.step = 4
                st.rerun()
        
        # Botão para voltar
        if st.button("⬅️ Voltar", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    
    def step_results(self):
        """Passo 4: Resultados"""
        st.markdown("## 📊 Resultados do Treinamento")
        
        if st.session_state.results is None:
            st.warning("⚠️ Nenhum resultado disponível.")
            if st.button("⬅️ Voltar para Treinamento"):
                st.session_state.step = 3
                st.rerun()
            return
        
        results = st.session_state.results
        trainer = st.session_state.trainer
        problem_type = st.session_state.problem_type
        
        # Métricas do melhor modelo
        best_model_name = trainer.best_model_name
        best_metrics = results[best_model_name]
        
        # Cartões de métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏆 Melhor Modelo", best_model_name)
        
        with col2:
            if problem_type == 'classification':
                st.metric("🎯 Acurácia", f"{best_metrics['accuracy']:.3f}")
            else:
                st.metric("🎯 R² Score", f"{best_metrics['r2']:.3f}")
        
        with col3:
            if problem_type == 'classification':
                st.metric("📈 F1-Score", f"{best_metrics['f1_score']:.3f}")
            else:
                st.metric("📈 RMSE", f"{best_metrics['rmse']:.3f}")
        
        with col4:
            st.metric("🤖 Modelos Treinados", len(results))
        
        # Ranking dos modelos
        st.markdown("### 🏆 Ranking dos Modelos")
        
        ranking_df = trainer.get_ranking()
        st.dataframe(ranking_df, use_container_width=True)
        
        # Gráfico do ranking
        st.markdown("### 📈 Visualização do Ranking")
        
        if problem_type == 'classification':
            fig = px.bar(
                ranking_df,
                x='Modelo',
                y='Acurácia',
                title='Acurácia por Modelo',
                color='Acurácia',
                color_continuous_scale='Viridis',
                text='Acurácia'
            )
        else:
            fig = px.bar(
                ranking_df,
                x='Modelo',
                y='R²',
                title='R² Score por Modelo',
                color='R²',
                color_continuous_scale='Viridis',
                text='R²'
            )
        
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Abas de detalhes
        tab1, tab2, tab3 = st.tabs(["📋 Detalhes", "💾 Exportar", "🔄 Novo"])
        
        with tab1:
            st.markdown("#### 📊 Métricas Detalhadas")
            
            # Tabela completa de métricas
            metrics_df = pd.DataFrame(results).T
            st.dataframe(metrics_df, use_container_width=True)
            
            # Comparação visual
            st.markdown("#### 📈 Comparação entre Modelos")
            
            models = list(results.keys())
            
            if problem_type == 'classification':
                scores = [results[m]['accuracy'] for m in models]
                metric_name = 'Acurácia'
            else:
                scores = [results[m]['r2'] for m in models]
                metric_name = 'R²'
            
            fig2 = go.Figure(data=[
                go.Bar(
                    x=models, 
                    y=scores,
                    marker_color=['#FF6B6B' if m == best_model_name else '#4ECDC4' for m in models],
                    text=[f'{s:.3f}' for s in scores],
                    textposition='auto'
                )
            ])
            
            fig2.update_layout(
                title=f'{metric_name} - Comparação',
                xaxis_title='Modelo',
                yaxis_title=metric_name,
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.markdown("#### 💾 Exportar Resultados")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Exportar ranking
                ranking_csv = ranking_df.to_csv(index=False).encode()
                st.download_button(
                    label="📊 Ranking CSV",
                    data=ranking_csv,
                    file_name="ranking_modelos.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Exportar métricas completas
                metrics_csv = pd.DataFrame(results).T.to_csv().encode()
                st.download_button(
                    label="📈 Métricas CSV",
                    data=metrics_csv,
                    file_name="metricas_completas.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # Salvar melhor modelo
                if st.button("🤖 Salvar Modelo", use_container_width=True):
                    model_path = "melhor_modelo.pkl"
                    joblib.dump(trainer.best_model, model_path)
                    
                    with open(model_path, "rb") as f:
                        model_bytes = f.read()
                    
                    st.download_button(
                        label="⬇️ Baixar .pkl",
                        data=model_bytes,
                        file_name="melhor_modelo.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
            
            # Relatório de análise
            st.markdown("---")
            st.markdown("#### 📄 Relatório de Análise")
            
            report = f"""
            # Relatório de AutoML
            Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            
            ## Resumo
            - Tipo de problema: {problem_type.upper()}
            - Melhor modelo: {best_model_name}
            - Total de modelos treinados: {len(results)}
            
            ## Métricas do Melhor Modelo
            """
            
            if problem_type == 'classification':
                report += f"""
                - Acurácia: {best_metrics['accuracy']:.4f}
                - F1-Score: {best_metrics['f1_score']:.4f}
                - CV Score: {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}
                """
            else:
                report += f"""
                - R² Score: {best_metrics['r2']:.4f}
                - RMSE: {best_metrics['rmse']:.4f}
                - CV Score: {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}
                """
            
            report += "\n\n## Ranking Completo\n" + ranking_df.to_markdown()
            
            st.download_button(
                label="📄 Baixar Relatório",
                data=report.encode(),
                file_name="relatorio_automl.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with tab3:
            st.markdown("#### 🔄 Novo Treinamento")
            
            st.info("""
            Clique no botão abaixo para:
            1. Limpar todos os resultados atuais
            2. Voltar à tela inicial
            3. Começar um novo treinamento
            """)
            
            if st.button("🔄 Iniciar Novo Projeto", type="primary", use_container_width=True):
                # Limpar estado da sessão
                keys_to_keep = ['scale_features']
                keys_to_delete = [k for k in st.session_state.keys() if k not in keys_to_keep]
                
                for key in keys_to_delete:
                    del st.session_state[key]
                
                st.session_state.step = 1
                st.rerun()
        
        # Botão para voltar
        if st.button("⬅️ Voltar para Treinamento", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ========== IMPORTS ADICIONAIS ==========
import io

# ========== EXECUÇÃO ==========
if __name__ == "__main__":
    app = AutoMLApp()
    app.run()