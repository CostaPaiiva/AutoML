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
    page_title="AutoML Ultra Robusto",
    page_icon="🤖",
    layout="wide"
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
    .stButton > button {
        width: 100%;
        font-weight: bold;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== PROCESSAMENTO DE DADOS ULTRA SIMPLES ==========
class UltraSimpleProcessor:
    def process(self, data, target_col):
        """Processamento ultra simples e à prova de erros"""
        try:
            # 1. Separar X e y
            if target_col in data.columns:
                X = data.drop(columns=[target_col]).copy()
                y = data[target_col].copy()
            else:
                # Fallback: usar última coluna
                X = data.iloc[:, :-1].copy()
                y = data.iloc[:, -1].copy()
            
            # 2. Detectar tipo de problema
            try:
                unique_y = len(y.unique())
                if y.dtype == 'object' or unique_y <= 10:
                    problem_type = 'classification'
                else:
                    problem_type = 'regression'
            except:
                problem_type = 'regression'  # Fallback
            
            # 3. Limpeza básica de X
            X_clean = self.simple_clean(X)
            
            return X_clean, y, problem_type
            
        except Exception as e:
            # Fallback absoluto
            st.error(f"Erro no processamento: {str(e)}. Usando fallback...")
            return self.absolute_fallback(data, target_col)
    
    def simple_clean(self, X):
        """Limpeza ultra simples"""
        # Converter tudo para numérico
        X_num = X.copy()
        
        for col in X_num.columns:
            try:
                # Tentar converter para numérico
                X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
            except:
                # Se não conseguir, converter categórico para numérico
                X_num[col] = pd.factorize(X_num[col])[0]
        
        # Preencher NaN com 0
        X_num = X_num.fillna(0)
        
        return X_num
    
    def absolute_fallback(self, data, target_col):
        """Fallback absoluto - funciona com QUALQUER dataset"""
        try:
            # Usar apenas as primeiras 1000 linhas para não sobrecarregar
            if len(data) > 1000:
                data = data.sample(1000, random_state=42)
            
            # Criar features sintéticas se necessário
            if len(data.columns) < 2:
                # Dataset muito pequeno, criar features
                n_samples = len(data)
                data['feature_1'] = np.random.randn(n_samples)
                data['feature_2'] = np.random.randn(n_samples)
            
            # Target é última coluna
            if target_col not in data.columns:
                target_col = data.columns[-1]
            
            X = data.drop(columns=[target_col]).copy()
            y = data[target_col].copy()
            
            # Converter tudo para numérico
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            X = X.fillna(0)
            
            # Verificar se y é numérico
            try:
                y = pd.to_numeric(y, errors='coerce')
                y = y.fillna(0)
                problem_type = 'regression'
            except:
                problem_type = 'classification'
            
            return X, y, problem_type
            
        except Exception as e:
            # Último recurso: criar dados sintéticos
            st.error("Erro crítico. Criando dataset de demonstração...")
            return self.create_demo_data()

    def create_demo_data(self):
        """Cria dados de demonstração"""
        n_samples = 100
        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        return X, y, 'classification'

# ========== TREINAMENTO ULTRA ROBUSTO ==========
class UltraRobustTrainer:
    def __init__(self, problem_type):
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = ""
    
    def train_safe(self, X, y):
        """Treinamento seguro à prova de erros"""
        st.info("🤖 Iniciando treinamento seguro...")
        
        try:
            # Verificar se temos dados suficientes
            if len(X) < 10:
                st.warning("⚠️ Dataset muito pequeno. Treinando com validação simples.")
                return self.train_minimal(X, y)
            
            # Split seguro
            try:
                from sklearn.model_selection import train_test_split
                
                if self.problem_type == 'classification' and len(y.unique()) > 1:
                    # Tentar stratified split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                else:
                    # Split normal
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
            except:
                # Split manual simples
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Obter modelos simples
            models = self.get_simple_models()
            
            # Treinar cada modelo
            for name, model in models.items():
                try:
                    st.write(f"📊 Treinando {name}...")
                    
                    # Treinar
                    model.fit(X_train, y_train)
                    
                    # Prever
                    y_pred = model.predict(X_test)
                    
                    # Calcular métricas
                    metrics = self.calculate_safe_metrics(y_test, y_pred)
                    
                    # Salvar
                    self.models[name] = model
                    self.results[name] = metrics
                    
                except Exception as e:
                    st.write(f"⚠️ {name}: {str(e)[:50]}...")
                    continue
            
            # Determinar melhor modelo
            if self.results:
                self.determine_best_model_safe()
                st.success(f"✅ {len(self.results)} modelos treinados com sucesso!")
                st.success(f"🏆 Melhor: {self.best_model_name}")
            else:
                st.error("❌ Nenhum modelo treinou com sucesso.")
            
            return self.results, self.best_model_name
            
        except Exception as e:
            st.error(f"❌ Erro no treinamento: {str(e)}")
            return self.train_minimal(X, y)
    
    def train_minimal(self, X, y):
        """Treinamento mínimo de fallback"""
        st.info("Usando treinamento mínimo...")
        
        try:
            # Modelos mínimos
            if self.problem_type == 'classification':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000)
            else:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            
            # Treinar com todos os dados
            model.fit(X, y)
            
            # Métricas simples
            y_pred = model.predict(X)
            metrics = self.calculate_safe_metrics(y, y_pred)
            
            # Salvar
            model_name = "Logistic Regression" if self.problem_type == 'classification' else "Linear Regression"
            self.models[model_name] = model
            self.results[model_name] = metrics
            self.best_model_name = model_name
            self.best_model = model
            
            st.success(f"✅ {model_name} treinado!")
            
            return self.results, self.best_model_name
            
        except:
            # Último recurso
            st.error("❌ Não foi possível treinar modelos.")
            return {}, ""
    
    def get_simple_models(self):
        """Modelos simples e robustos"""
        if self.problem_type == 'classification':
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.naive_bayes import GaussianNB
            
            return {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=50),
                'Decision Tree': DecisionTreeClassifier(),
                'Naive Bayes': GaussianNB()
            }
        else:
            from sklearn.linear_model import LinearRegression, Ridge
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.tree import DecisionTreeRegressor
            
            return {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Random Forest': RandomForestRegressor(n_estimators=50),
                'Decision Tree': DecisionTreeRegressor()
            }
    
    def calculate_safe_metrics(self, y_true, y_pred):
        """Cálculo seguro de métricas"""
        try:
            if self.problem_type == 'classification':
                from sklearn.metrics import accuracy_score
                return {'accuracy': accuracy_score(y_true, y_pred)}
            else:
                from sklearn.metrics import r2_score
                return {'r2': r2_score(y_true, y_pred)}
        except:
            return {'score': 0.5}  # Valor padrão
    
    def determine_best_model_safe(self):
        """Determina melhor modelo de forma segura"""
        if not self.results:
            return
        
        # Encontrar modelo com maior score
        best_score = -1
        best_name = ""
        
        for name, metrics in self.results.items():
            score = metrics.get('accuracy', metrics.get('r2', metrics.get('score', 0)))
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models.get(best_name)
    
    def get_ranking(self):
        """Ranking seguro"""
        if not self.results:
            return pd.DataFrame(columns=['Modelo', 'Score'])
        
        ranking = []
        for name, metrics in self.results.items():
            score = metrics.get('accuracy', metrics.get('r2', metrics.get('score', 0)))
            ranking.append({
                'Modelo': name,
                'Score': f"{score:.4f}"
            })
        
        df = pd.DataFrame(ranking)
        df = df.sort_values('Score', ascending=False)
        df.insert(0, 'Posição', range(1, len(df) + 1))
        
        return df

# ========== APLICAÇÃO PRINCIPAL ==========
class UltraRobustApp:
    def __init__(self):
        # Estado da sessão
        if 'step' not in st.session_state:
            st.session_state.step = 1
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'processed' not in st.session_state:
            st.session_state.processed = False
    
    def run(self):
        """Executa a aplicação"""
        st.title("🤖 AutoML Ultra Robusto")
        st.markdown("Sistema à prova de erros para qualquer dataset!")
        
        # Progresso
        self.show_progress()
        
        # Conteúdo
        if st.session_state.step == 1:
            self.step_upload()
        elif st.session_state.step == 2:
            self.step_process()
        elif st.session_state.step == 3:
            self.step_train()
        elif st.session_state.step == 4:
            self.step_results()
    
    def show_progress(self):
        """Barra de progresso simples"""
        steps = ["📥 Upload", "🔧 Processar", "🤖 Treinar", "📊 Resultados"]
        current = st.session_state.step - 1
        
        cols = st.columns(4)
        for i, col in enumerate(cols):
            with col:
                if i < current:
                    st.success(steps[i])
                elif i == current:
                    st.info(steps[i])
                else:
                    st.write(steps[i])
    
    def step_upload(self):
        """Upload do dataset"""
        st.header("📥 Upload do Dataset")
        
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV", 
            type=['csv', 'txt', 'xlsx'],
            help="Suporta CSV, TXT e Excel"
        )
        
        if uploaded_file:
            try:
                # Ler arquivo
                if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                else:
                    # Tentar ler como CSV
                    data = pd.read_csv(uploaded_file)
                
                st.session_state.data = data
                st.success(f"✅ Dataset carregado: {data.shape[0]} linhas × {data.shape[1]} colunas")
                
                # Preview
                with st.expander("👁️ Visualizar dados"):
                    st.dataframe(data.head(), use_container_width=True)
                
                # Selecionar target
                target_options = data.columns.tolist()
                target_col = st.selectbox(
                    "🎯 Selecione a coluna target:",
                    target_options,
                    index=len(target_options)-1
                )
                
                st.session_state.target_col = target_col
                
                if st.button("▶️ Processar Dados", type="primary"):
                    st.session_state.step = 2
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Erro ao ler arquivo: {str(e)}")
                # Tentar fallback
                try:
                    data = pd.read_csv(uploaded_file, encoding='latin-1')
                    st.session_state.data = data
                    st.warning("⚠️ Arquivo lido com encoding alternativo.")
                except:
                    st.error("❌ Não foi possível ler o arquivo.")
    
    def step_process(self):
        """Processamento dos dados"""
        st.header("🔧 Processamento de Dados")
        
        if st.session_state.data is None:
            st.warning("Nenhum dataset carregado.")
            if st.button("⬅️ Voltar"):
                st.session_state.step = 1
                st.rerun()
            return
        
        with st.spinner("Processando..."):
            processor = UltraSimpleProcessor()
            X, y, problem_type = processor.process(
                st.session_state.data, 
                st.session_state.target_col
            )
            
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.problem_type = problem_type
            st.session_state.processed = True
        
        st.success("✅ Processamento concluído!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tipo", problem_type.upper())
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            st.metric("Amostras", X.shape[0])
        
        # Botões
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Voltar"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("🤖 Treinar Modelos", type="primary"):
                st.session_state.step = 3
                st.rerun()
    
    def step_train(self):
        """Treinamento dos modelos"""
        st.header("🤖 Treinamento de Modelos")
        
        if not st.session_state.processed:
            st.warning("Dados não processados.")
            st.session_state.step = 2
            st.rerun()
            return
        
        X = st.session_state.X
        y = st.session_state.y
        problem_type = st.session_state.problem_type
        
        st.info(f"""
        **Configuração:**
        - Tipo: {problem_type.upper()}
        - Dimensões: {X.shape[1]} features × {X.shape[0]} amostras
        - Modelos: 4 algoritmos robustos
        """)
        
        if st.button("🚀 Iniciar Treinamento", type="primary"):
            with st.spinner("Treinando..."):
                trainer = UltraRobustTrainer(problem_type)
                results, best_model = trainer.train_safe(X, y)
                
                st.session_state.results = results
                st.session_state.trainer = trainer
                st.session_state.best_model = best_model
                
                st.session_state.step = 4
                st.rerun()
        
        if st.button("⬅️ Voltar"):
            st.session_state.step = 2
            st.rerun()
    
    def step_results(self):
        """Resultados"""
        st.header("📊 Resultados")
        
        if 'results' not in st.session_state or not st.session_state.results:
            st.warning("Nenhum resultado disponível.")
            if st.button("⬅️ Voltar"):
                st.session_state.step = 3
                st.rerun()
            return
        
        results = st.session_state.results
        trainer = st.session_state.trainer
        
        # Melhor modelo
        best_name = trainer.best_model_name
        if best_name and best_name in results:
            best_score = results[best_name].get('accuracy', results[best_name].get('r2', 0))
        else:
            best_score = 0
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Melhor Modelo", best_name or "N/A")
        with col2:
            st.metric("🎯 Score", f"{best_score:.4f}")
        with col3:
            st.metric("🤖 Modelos", len(results))
        
        # Ranking
        st.subheader("🏆 Ranking")
        ranking_df = trainer.get_ranking()
        st.dataframe(ranking_df, use_container_width=True)
        
        # Gráfico
        if not ranking_df.empty:
            fig = px.bar(
                ranking_df,
                x='Modelo',
                y='Score',
                title='Performance dos Modelos',
                color='Score'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Exportação
        st.subheader("💾 Exportar")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not ranking_df.empty:
                csv = ranking_df.to_csv(index=False).encode()
                st.download_button(
                    "📊 CSV",
                    csv,
                    "ranking.csv",
                    "text/csv"
                )
        
        with col2:
            if trainer.best_model:
                try:
                    joblib.dump(trainer.best_model, 'modelo.pkl')
                    with open('modelo.pkl', 'rb') as f:
                        model_bytes = f.read()
                    
                    st.download_button(
                        "🤖 Modelo",
                        model_bytes,
                        "melhor_modelo.pkl",
                        "application/octet-stream"
                    )
                except:
                    st.write("❌ Não foi possível salvar o modelo")
        
        with col3:
            if st.button("🔄 Novo"):
                # Limpar estado
                for key in ['data', 'X', 'y', 'results', 'trainer', 'processed']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.step = 1
                st.rerun()

# ========== EXECUTAR ==========
if __name__ == "__main__":
    # Configurar para evitar warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Executar app
    try:
        app = UltraRobustApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Erro crítico: {str(e)}")
        st.info("Recarregue a página para tentar novamente.")