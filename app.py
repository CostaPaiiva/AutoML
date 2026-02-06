import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import base64
import joblib
from datetime import datetime
import os
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

# ========== GERADOR DE RELATÓRIO PDF ==========
class PDFReportGenerator:
    """Gera relatório PDF dos resultados"""
    
    @staticmethod
    def generate_report(results, trainer, problem_type, data_info=None):
        """Gera relatório PDF com todos os resultados"""
        try:
            # Tenta usar fpdf2
            try:
                from fpdf import FPDF
                
                # Criar PDF
                pdf = FPDF()
                pdf.add_page()
                
                # Configurações - SEM EMOJIS
                pdf.set_font("Arial", 'B', 16)
                
                # Título SEM EMOJIS
                pdf.cell(0, 10, "RELATORIO AUTOML", ln=True, align='C')
                pdf.ln(5)
                
                # Data
                pdf.set_font("Arial", '', 10)
                pdf.cell(0, 10, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
                pdf.ln(10)
                
                # Informações do projeto
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "INFORMACOES DO PROJETO", ln=True)
                pdf.set_font("Arial", '', 10)
                
                if data_info:
                    pdf.cell(0, 10, f"Dataset: {data_info.get('dataset_name', 'N/A')}", ln=True)
                    pdf.cell(0, 10, f"Amostras: {data_info.get('n_samples', 'N/A')}", ln=True)
                    pdf.cell(0, 10, f"Features: {data_info.get('n_features', 'N/A')}", ln=True)
                
                pdf.cell(0, 10, f"Tipo de problema: {problem_type.upper()}", ln=True)
                pdf.cell(0, 10, f"Total de modelos treinados: {len(results)}", ln=True)
                pdf.ln(10)
                
                # Melhor modelo
                best_name = trainer.best_model_name
                if best_name and best_name in results:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "MELHOR MODELO", ln=True)
                    pdf.set_font("Arial", '', 10)
                    
                    best_metrics = results[best_name]
                    pdf.cell(0, 10, f"Modelo: {best_name}", ln=True)
                    
                    if problem_type == 'classification':
                        score = best_metrics.get('accuracy', best_metrics.get('score', 0))
                        pdf.cell(0, 10, f"Acurácia: {score:.4f}", ln=True)
                    else:
                        score = best_metrics.get('r2', best_metrics.get('score', 0))
                        pdf.cell(0, 10, f"R² Score: {score:.4f}", ln=True)
                    
                    pdf.ln(10)
                
                # Ranking completo
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "RANKING DOS MODELOS", ln=True)
                pdf.set_font("Arial", '', 10)
                
                ranking_df = trainer.get_ranking()
                
                # Adicionar tabela
                pdf.set_fill_color(240, 240, 240)
                pdf.cell(40, 10, "Posicao", border=1, fill=True)
                pdf.cell(80, 10, "Modelo", border=1, fill=True)
                pdf.cell(40, 10, "Score", border=1, fill=True, ln=True)
                
                for i, row in ranking_df.iterrows():
                    pos = str(row['Posição'])
                    model = str(row['Modelo'])
                    score = str(row['Score'])
                    
                    pdf.cell(40, 10, pos, border=1)
                    pdf.cell(80, 10, model, border=1)
                    pdf.cell(40, 10, score, border=1, ln=True)
                
                pdf.ln(10)
                
                # Métricas detalhadas
                if len(results) > 0:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "METRICAS DETALHADAS", ln=True)
                    pdf.set_font("Arial", '', 10)
                    
                    for model_name, metrics in results.items():
                        pdf.set_font("Arial", 'B', 10)
                        pdf.cell(0, 10, f"Modelo: {model_name}", ln=True)
                        pdf.set_font("Arial", '', 9)
                        
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                pdf.cell(0, 8, f"  {metric_name}: {value:.4f}", ln=True)
                        pdf.ln(5)
                    
                    pdf.ln(10)
                
                # Recomendações SEM EMOJIS
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "RECOMENDACOES", ln=True)
                pdf.set_font("Arial", '', 10)
                
                recommendations = [
                    "1. Implemente o melhor modelo em producao",
                    "2. Monitore performance periodicamente",
                    "3. Re-treine com novos dados regularmente",
                    "4. Considere tecnicas de ensemble",
                    "5. Valide com testes A/B antes de deploy"
                ]
                
                for rec in recommendations:
                    pdf.cell(0, 8, rec, ln=True)
                
                # Salvar PDF
                os.makedirs('reports', exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'reports/relatorio_automl_{timestamp}.pdf'
                
                try:
                    pdf.output(filename)
                    
                    # Verificar se arquivo foi criado
                    if os.path.exists(filename):
                        return filename
                    else:
                        st.error("PDF não foi criado")
                        return None
                        
                except Exception as e:
                    st.error(f"Erro ao salvar PDF: {str(e)}")
                    return PDFReportGenerator.generate_txt_report(results, trainer, problem_type, data_info)
                
            except ImportError as e:
                st.warning("fpdf2 não encontrado. Gerando relatório TXT...")
                return PDFReportGenerator.generate_txt_report(results, trainer, problem_type, data_info)
            except Exception as e:
                st.error(f"Erro no fpdf: {str(e)}")
                return PDFReportGenerator.generate_txt_report(results, trainer, problem_type, data_info)
                
        except Exception as e:
            # Fallback final
            st.error(f"Erro ao gerar relatório: {str(e)}")
            return PDFReportGenerator.generate_txt_report(results, trainer, problem_type, data_info)
    
    @staticmethod
    def generate_txt_report(results, trainer, problem_type, data_info=None):
        """Gera relatório em texto (fallback)"""
        try:
            
            # Criar pasta
            os.makedirs('reports', exist_ok=True)
            
            # Gerar nome único
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'reports/relatorio_automl_{timestamp}.txt'
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("RELATORIO AUTOML - TODOS OS MODELOS\n")
                f.write("=" * 60 + "\n\n")
                
                # Informações básicas
                f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
                f.write(f"Tipo de problema: {problem_type.upper()}\n")
                f.write(f"Total de modelos: {len(results)}\n")
                
                if data_info:
                    f.write(f"Amostras: {data_info.get('n_samples', 'N/A')}\n")
                    f.write(f"Features: {data_info.get('n_features', 'N/A')}\n")
                
                # Melhor modelo
                f.write("\n" + "=" * 60 + "\n")
                f.write("MELHOR MODELO\n")
                f.write("=" * 60 + "\n\n")
                
                best_name = trainer.best_model_name
                if best_name and best_name in results:
                    f.write(f"Modelo: {best_name}\n")
                    best_metrics = results[best_name]
                    
                    for metric, value in best_metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"{metric}: {value:.4f}\n")
                
                # Ranking
                f.write("\n" + "=" * 60 + "\n")
                f.write("RANKING COMPLETO\n")
                f.write("=" * 60 + "\n\n")
                
                ranking_df = trainer.get_ranking()
                for i, row in ranking_df.iterrows():
                    f.write(f"{row['Posição']}. {row['Modelo']} - Score: {row['Score']}\n")
                
                # Métricas detalhadas
                f.write("\n" + "=" * 60 + "\n")
                f.write("METRICAS POR MODELO\n")
                f.write("=" * 60 + "\n\n")
                
                for model_name, metrics in results.items():
                    f.write(f"{model_name}:\n")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n")
                
                # Recomendações
                f.write("=" * 60 + "\n")
                f.write("RECOMENDACOES\n")
                f.write("=" * 60 + "\n\n")
                
                recs = [
                    "• Use o melhor modelo em producao",
                    "• Monitore performance",
                    "• Re-treine regularmente",
                    "• Valide com novos dados"
                ]
                
                for rec in recs:
                    f.write(f"{rec}\n")
            
            return filename
            
        except Exception as e:
            st.error(f"Erro ao gerar TXT: {str(e)}")
            return None

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
        problem_type = st.session_state.problem_type
        
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
        
        # ⭐⭐ COMEÇO DA SEÇÃO DE EXPORTAÇÃO ⭐⭐
        # EXPORTAÇÃO COMPLETA - Com PDF
        st.subheader("💾 Exportar Resultados")
        
        # Criar 4 colunas em vez de 3
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # CSV do ranking
            if not ranking_df.empty:
                try:
                    csv_data = ranking_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 CSV",
                        csv_data,
                        "ranking.csv",
                        "text/csv",
                        key="csv_button"
                    )
                except Exception as e:
                    st.error(f"Erro CSV: {e}")
        
        with col2:
            # SALVAR MODELO .pkl
            if trainer.best_model is not None:
                # Botão para salvar o modelo
                if st.button("💾 Salvar Modelo", key="save_model_btn"):
                    try:
                        # Criar pasta se não existir
                        os.makedirs('models', exist_ok=True)
                        
                        # Criar nome único com timestamp
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        model_filename = f"modelo_{best_name.replace(' ', '_')}_{timestamp}.pkl"
                        model_path = f"models/{model_filename}"
                        
                        # Salvar modelo
                        joblib.dump(trainer.best_model, model_path)
                        
                        # Verificar se salvou
                        if os.path.exists(model_path):
                            file_size = os.path.getsize(model_path)
                            st.success(f"✅ Modelo salvo: {model_filename} ({file_size} bytes)")
                            
                            # Botão para download
                            with open(model_path, 'rb') as f:
                                model_bytes = f.read()
                            
                            st.download_button(
                                "⬇️ Baixar Modelo",
                                model_bytes,
                                model_filename,
                                "application/octet-stream",
                                key=f"download_{timestamp}"
                            )
                        else:
                            st.error("❌ Arquivo não foi criado")
                            
                    except Exception as e:
                        st.error(f"❌ Erro ao salvar: {str(e)}")
            else:
                st.info("Nenhum modelo disponível")
        
        # ⭐⭐ COLUNA 3 - ONDE VOCÊ COLOCA O CÓDIGO DO PDF ⭐⭐
        with col3:
            # GERAR RELATÓRIO PDF - VERSÃO SIMPLES
            if st.button("📄 Gerar Relatório", key="pdf_report_btn"):
                with st.spinner("Gerando relatório..."):
                    try:
                        # Preparar informações
                        data_info = {
                            'dataset_name': 'Dataset Processado',
                            'n_samples': st.session_state.X.shape[0] if 'X' in st.session_state else 'N/A',
                            'n_features': st.session_state.X.shape[1] if 'X' in st.session_state else 'N/A'
                        }
                        
                        # Tentar PDF primeiro
                        pdf_path = PDFReportGenerator.generate_report(
                            results, 
                            trainer, 
                            problem_type,
                            data_info
                        )
                        
                        # Se PDF falhou, usar TXT
                        if not pdf_path or not os.path.exists(pdf_path):
                            st.warning("PDF falhou, gerando TXT...")
                            pdf_path = PDFReportGenerator.generate_txt_report(
                                results, trainer, problem_type, data_info
                            )
                        
                        if pdf_path and os.path.exists(pdf_path):
                            # Ler arquivo
                            with open(pdf_path, 'rb') as f:
                                file_bytes = f.read()
                            
                            # Determinar tipo MIME
                            if pdf_path.endswith('.pdf'):
                                mime_type = "application/pdf"
                                btn_label = "⬇️ Baixar PDF"
                                file_ext = "pdf"
                            else:
                                mime_type = "text/plain"
                                btn_label = "⬇️ Baixar TXT"
                                file_ext = "txt"
                            
                            # Botão de download
                            st.download_button(
                                btn_label,
                                file_bytes,
                                os.path.basename(pdf_path),
                                mime_type,
                                key="download_report"
                            )
                            
                            st.success(f"✅ Relatório gerado: {os.path.basename(pdf_path)}")
                            
                        else:
                            st.error("❌ Falha ao gerar relatório")
                            
                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")
        # ⭐⭐ FIM DA COLUNA 3 ⭐⭐
        
        with col4:
            # Botão para novo treinamento
            if st.button("🔄 Novo", type="primary"):
                # Limpar estado
                for key in ['data', 'X', 'y', 'results', 'trainer', 'processed']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.step = 1
                st.rerun()
        
        # Seção de arquivos gerados
        st.markdown("---")
        with st.expander("📁 Arquivos Gerados"):
            try:
                
                # Listar modelos
                if os.path.exists('models'):
                    st.write("**📦 Modelos salvos:**")
                    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
                    if model_files:
                        for f in sorted(model_files, reverse=True)[:5]:  # Mostrar últimos 5
                            st.write(f"- `{f}`")
                    else:
                        st.write("(nenhum modelo salvo ainda)")
                
                # Listar relatórios
                if os.path.exists('reports'):
                    st.write("**📄 Relatórios gerados:**")
                    report_files = [f for f in os.listdir('reports') if f.endswith(('.pdf', '.txt'))]
                    if report_files:
                        for f in sorted(report_files, reverse=True)[:3]:  # Mostrar últimos 3
                            filepath = os.path.join('reports', f)
                            filesize = os.path.getsize(filepath)
                            st.write(f"- `{f}` ({filesize} bytes)")
                    else:
                        st.write("(nenhum relatório gerado ainda)")
                        
            except Exception as e:
                st.write(f"Erro ao listar arquivos: {e}")

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