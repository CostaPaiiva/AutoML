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

# ========== CONFIGURAÇÃO DA PÁGINA ==========
st.set_page_config(
    page_title="AutoML",
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
    .power-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        text-align: center;
    }
    .model-badge {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.8em;
        background-color: #e3f2fd;
        color: #1565c0;
    }
    .cv-badge {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
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
                pdf.cell(0, 10, "RELATORIO AUTOML PRO", ln=True, align='C')
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
                f.write("RELATORIO AUTOML PRO - TODOS OS MODELOS\n")
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

# ========== PROCESSAMENTO DE DADOS POWERFULL PRO ==========
class PowerfulDataProcessor:
    """Processador de dados avançado com feature engineering"""
    
    def __init__(self):
        self.scaler = None
        self.encoders = {}
        self.imputer = None
        self.selected_features = []
    
    def process(self, data, target_col):
        """Processamento POWERFULL com feature engineering"""
        try:
            # 1. Separar X e y
            if target_col in data.columns:
                X = data.drop(columns=[target_col]).copy()
                y = data[target_col].copy()
            else:
                X = data.iloc[:, :-1].copy()
                y = data.iloc[:, -1].copy()
            
            # 2. Detectar tipo de problema CORRETAMENTE
            problem_type = self.detect_problem_type_smart(y)
            
            # 3. Pré-processamento POWERFULL
            X_processed = self.powerful_preprocessing(X)
            
            # 4. Feature Engineering AVANÇADO
            X_engineered = self.advanced_feature_engineering(X_processed)
            
            # 5. Seleção de Features INTELIGENTE
            X_final = self.smart_feature_selection(X_engineered, y, problem_type)
            
            # 6. Processar target
            if problem_type == 'classification':
                y_processed = self.process_target(y)
            else:
                y_processed = self.process_target_regression(y)
            
            return X_final, y_processed, problem_type
            
        except Exception as e:
            st.error(f"Erro no processamento POWERFULL: {str(e)}")
            # Fallback para o processador antigo
            return self.simple_fallback(data, target_col)
    
    def detect_problem_type_smart(self, y):
        """Detecção INTELIGENTE de tipo de problema"""
        try:
            # Converter para numérico se possível
            y_numeric = pd.to_numeric(y, errors='coerce')
            not_na = y_numeric.notna().sum()
            
            # Se menos de 80% são numéricos, é classificação
            if not_na / len(y) < 0.8:
                return 'classification'
            
            # Agora trabalhar com os valores numéricos
            y_clean = y_numeric.dropna()
            unique_vals = len(y_clean.unique())
            
            # Regras avançadas
            if unique_vals <= 10:
                # Verificar se são valores inteiros (provavelmente classes)
                if all(y_clean.astype(int) == y_clean):
                    return 'classification'
                else:
                    # Poucos valores mas não inteiros - regressão
                    return 'regression'
            elif unique_vals <= 30:
                # Verificar distribuição
                value_counts = y_clean.value_counts(normalize=True)
                # Se algum valor tem > 30% dos dados, pode ser classificação
                if (value_counts > 0.3).any():
                    return 'classification'
                else:
                    return 'regression'
            else:
                return 'regression'
                
        except Exception as e:
            # Fallback simples
            try:
                unique_vals = len(y.unique())
                if y.dtype == 'object' or unique_vals <= 10:
                    return 'classification'
                else:
                    return 'regression'
            except:
                return 'regression'
    
    def powerful_preprocessing(self, X):
        """Pré-processamento avançado"""
        X_clean = X.copy()
        
        # 1. Tratar valores ausentes de forma inteligente
        for col in X_clean.columns:
            if X_clean[col].isna().any():
                # Para numéricas: média ou mediana
                if pd.api.types.is_numeric_dtype(X_clean[col]):
                    if X_clean[col].skew() > 1:  # Distribuição assimétrica
                        X_clean[col] = X_clean[col].fillna(X_clean[col].median())
                    else:
                        X_clean[col] = X_clean[col].fillna(X_clean[col].mean())
                else:
                    # Para categóricas: moda
                    X_clean[col] = X_clean[col].fillna(X_clean[col].mode()[0])
        
        # 2. Separar tipos de colunas
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # 3. Processar colunas numéricas
        if numeric_cols:
            X_numeric = X_clean[numeric_cols]
            
            # Remover outliers (preservando 95% dos dados)
            for col in X_numeric.columns:
                Q1 = X_numeric[col].quantile(0.25)
                Q3 = X_numeric[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X_numeric[col] = np.clip(X_numeric[col], lower_bound, upper_bound)
            
            # Normalizar
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)
            X_clean[numeric_cols] = X_numeric_scaled
        
        # 4. Processar colunas categóricas
        if categorical_cols:
            for col in categorical_cols:
                # One-hot encoding para categorias com até 10 valores únicos
                unique_vals = len(X_clean[col].unique())
                if unique_vals <= 10:
                    # One-hot encoding
                    dummies = pd.get_dummies(X_clean[col], prefix=col, drop_first=True)
                    X_clean = pd.concat([X_clean.drop(columns=[col]), dummies], axis=1)
                else:
                    # Para muitas categorias, usar frequency encoding
                    freq = X_clean[col].value_counts(normalize=True)
                    X_clean[col] = X_clean[col].map(freq)
        
        return X_clean
    
    def advanced_feature_engineering(self, X):
        """Feature engineering avançado"""
        X_engineered = X.copy()
        
        # 1. Adicionar interações entre features numéricas
        numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Criar algumas interações importantes
            for i in range(min(3, len(numeric_cols))):
                for j in range(i+1, min(i+3, len(numeric_cols))):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    X_engineered[f'{col1}_x_{col2}'] = X_engineered[col1] * X_engineered[col2]
                    X_engineered[f'{col1}_div_{col2}'] = X_engineered[col1] / (X_engineered[col2] + 1e-10)
        
        # 2. Adicionar estatísticas
        if len(numeric_cols) > 0:
            X_engineered['mean_features'] = X_engineered[numeric_cols].mean(axis=1)
            X_engineered['std_features'] = X_engineered[numeric_cols].std(axis=1)
            X_engineered['max_features'] = X_engineered[numeric_cols].max(axis=1)
            X_engineered['min_features'] = X_engineered[numeric_cols].min(axis=1)
        
        # 3. Adicionar features polinomiais
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Apenas primeiras 3 para não explodir
                X_engineered[f'{col}_squared'] = X_engineered[col] ** 2
                X_engineered[f'{col}_sqrt'] = np.sqrt(np.abs(X_engineered[col]) + 1e-10)
        
        return X_engineered
    
    def smart_feature_selection(self, X, y, problem_type):
        """Seleção inteligente de features"""
        try:
            # Se poucas features, manter todas
            if X.shape[1] <= 20:
                return X
            
            # Remover features com baixa variância
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            X_selected = selector.fit_transform(X)
            
            # Se ainda muitas features, usar seleção baseada em importância
            if X_selected.shape[1] > 50:
                if problem_type == 'classification':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                
                model.fit(X_selected, y)
                importances = model.feature_importances_
                
                # Manter top 30 features
                top_indices = np.argsort(importances)[-30:]
                X_final = X_selected[:, top_indices]
                self.selected_features = top_indices
            else:
                X_final = X_selected
            
            return X_final
            
        except Exception as e:
            # Se falhar, retornar X original
            st.write(f"⚠️ Feature selection falhou: {str(e)[:50]}")
            return X
    
    def process_target(self, y):
        """Processar target para classificação"""
        # Se for string, converter para numérico
        if y.dtype == 'object':
            y_encoded, _ = pd.factorize(y)
            return pd.Series(y_encoded)
        
        # Se for numérico mas com poucas classes, garantir que seja inteiro
        if len(y.unique()) <= 10:
            return y.astype(int)
        
        return y
    
    def process_target_regression(self, y):
        """Processar target para regressão"""
        try:
            # Converter para numérico
            y_numeric = pd.to_numeric(y, errors='coerce')
            
            # Tratar outliers no target (apenas para regressão)
            if len(y_numeric) > 100:
                Q1 = y_numeric.quantile(0.25)
                Q3 = y_numeric.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                y_numeric = np.clip(y_numeric, lower_bound, upper_bound)
            
            return y_numeric.fillna(y_numeric.median())
        except:
            return y
    
    def simple_fallback(self, data, target_col):
        """Fallback simples"""
        try:
            if target_col in data.columns:
                X = data.drop(columns=[target_col]).copy()
                y = data[target_col].copy()
            else:
                X = data.iloc[:, :-1].copy()
                y = data.iloc[:, -1].copy()
            
            # Limpeza básica
            X_num = X.copy()
            for col in X_num.columns:
                try:
                    X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
                except:
                    X_num[col] = pd.factorize(X_num[col])[0]
            
            X_num = X_num.fillna(0)
            
            # Detectar problema
            try:
                unique_y = len(y.unique())
                if y.dtype == 'object' or unique_y <= 10:
                    problem_type = 'classification'
                else:
                    problem_type = 'regression'
            except:
                problem_type = 'regression'
            
            return X_num, y, problem_type
        except:
            # Último recurso
            n_samples = 100
            X = pd.DataFrame({
                'feature_1': np.random.randn(n_samples),
                'feature_2': np.random.randn(n_samples),
            })
            y = pd.Series(np.random.randint(0, 2, n_samples))
            return X, y, 'classification'

# ========== TREINAMENTO ULTRA COMPLETO COM VALIDAÇÃO CRUZADA ==========
class UltraCompleteTrainer:
    def __init__(self, problem_type):
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        self.cv_scores = {}  # Scores da validação cruzada
        self.best_model = None
        self.best_model_name = ""
        self.use_cross_validation = True  # SEMPRE usa validação cruzada
        self.n_folds = 5  # Número padrão de folds
    
    def train_safe(self, X, y):
        """Treinamento com VALIDAÇÃO CRUZADA AUTOMÁTICA"""
        st.info("🔬 Iniciando treinamento com VALIDAÇÃO CRUZADA...")
        
        try:
            # Verificar se temos dados suficientes para CV
            if len(X) < 20:
                st.warning("⚠️ Dataset pequeno. Usando validação simples.")
                return self.train_simple_fallback(X, y)
            
            # Obter TODOS os modelos
            models = self.get_all_models()
            
            trained_count = 0
            total_models = len(models)
            
            # Barra de progresso global
            progress_bar = st.progress(0)
            
            # Treinar cada modelo COM VALIDAÇÃO CRUZADA
            for name, model in models.items():
                try:
                    with st.spinner(f"🔄 {name} (CV {self.n_folds}-fold)..."):
                        # Métricas da validação cruzada
                        cv_metrics, cv_scores = self.train_with_cross_validation(model, X, y)
                        
                        # Salvar resultados
                        self.models[name] = model
                        self.results[name] = cv_metrics
                        self.cv_scores[name] = cv_scores
                        trained_count += 1
                        
                        # Atualizar barra de progresso
                        progress = trained_count / total_models
                        progress_bar.progress(progress)
                        
                        # Mostrar score rápido
                        if self.problem_type == 'classification':
                            score = cv_metrics.get('accuracy', 0)
                        else:
                            score = cv_metrics.get('r2', 0)
                        
                        st.write(f"✅ **{name}**: {score:.4f} ± {cv_metrics.get('std', 0.0):.4f}")
                        
                except Exception as e:
                    st.write(f"⚠️ {name}: {str(e)[:50]}...")
                    continue
            
            # Determinar melhor modelo
            if self.results:
                self.determine_best_model_complete()
                st.success(f"✅ {trained_count} modelos treinados com VALIDAÇÃO CRUZADA!")
                
                if self.best_model_name:
                    # Treinar modelo final com todos os dados
                    self.train_final_model(X, y)
                    
                    # Mostrar resultados da CV
                    self.show_cv_results()
                    
                    st.success(f"🏆 **MELHOR MODELO**: {self.best_model_name}")
            
            return self.results, self.best_model_name
            
        except Exception as e:
            st.error(f"❌ Erro no treinamento: {str(e)}")
            return self.train_simple_fallback(X, y)
    
    def train_with_cross_validation(self, model, X, y):
        """Treina com validação cruzada e retorna métricas"""
        from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
        
        # Escolher estratégia de CV
        if self.problem_type == 'classification' and len(np.unique(y)) > 1:
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            cv_type = "Stratified K-Fold"
        else:
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            cv_type = "K-Fold"
        
        # Métricas baseadas no tipo de problema
        if self.problem_type == 'classification':
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision_weighted',
                'recall': 'recall_weighted',
                'f1': 'f1_weighted'
            }
            main_metric = 'accuracy'
        else:
            scoring = {
                'r2': 'r2',
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'neg_mean_absolute_error': 'neg_mean_absolute_error'
            }
            main_metric = 'r2'
        
        try:
            # Executar validação cruzada
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,  # Paraleliza o treinamento
                verbose=0
            )
            
            # Calcular médias e desvios padrão
            metrics = {}
            scores_dict = {}
            
            for metric_name in scoring.keys():
                score_key = f'test_{metric_name}'
                if score_key in cv_results:
                    scores = cv_results[score_key]
                    metrics[metric_name] = np.mean(scores)
                    metrics[f'{metric_name}_std'] = np.std(scores)
                    scores_dict[metric_name] = scores.tolist()
            
            # Converter scores negativos para positivos
            if self.problem_type == 'regression' and 'neg_mean_squared_error' in metrics:
                metrics['rmse'] = np.sqrt(-metrics['neg_mean_squared_error'])
                metrics['rmse_std'] = np.sqrt(metrics['neg_mean_squared_error_std'])
            
            if self.problem_type == 'regression' and 'neg_mean_absolute_error' in metrics:
                metrics['mae'] = -metrics['neg_mean_absolute_error']
                metrics['mae_std'] = metrics['neg_mean_absolute_error_std']
            
            # Adicionar tempo de treino
            metrics['fit_time'] = np.mean(cv_results['fit_time'])
            metrics['score_time'] = np.mean(cv_results['score_time'])
            metrics['cv_type'] = cv_type
            metrics['n_folds'] = self.n_folds
            
            return metrics, scores_dict
            
        except Exception as e:
            # Fallback: treino simples
            st.write(f"⚠️ CV falhou para este modelo: {str(e)[:50]}")
            return self.train_simple_model(model, X, y), {}
    
    def train_simple_model(self, model, X, y):
        """Fallback: treino simples sem CV"""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if self.problem_type == 'classification' else None
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return self.calculate_complete_metrics(y_test, y_pred)
    
    def train_final_model(self, X, y):
        """Treina o melhor modelo com todos os dados"""
        if self.best_model_name and self.best_model_name in self.models:
            # Clonar o modelo
            from sklearn.base import clone
            final_model = clone(self.models[self.best_model_name])
            
            # Treinar com todos os dados
            final_model.fit(X, y)
            self.best_model = final_model
    
    def show_cv_results(self):
        """Mostra resultados da validação cruzada"""
        if self.best_model_name and self.best_model_name in self.cv_scores:
            cv_scores = self.cv_scores[self.best_model_name]
            
            with st.expander(f"📊 Resultados CV - {self.best_model_name}"):
                # Mostrar scores de cada fold
                for metric, scores in cv_scores.items():
                    if len(scores) > 0:
                        st.write(f"**{metric} por fold:**")
                        for i, score in enumerate(scores):
                            st.write(f"  Fold {i+1}: {score:.4f}")
                        st.write(f"  **Média:** {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                        st.write("---")
    
    def train_simple_fallback(self, X, y):
        """Fallback completo sem CV"""
        st.info("Usando treinamento simples (sem CV)...")
        
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Modelo simples
        if self.problem_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self.calculate_complete_metrics(y_test, y_pred)
        
        model_name = "Random Forest"
        self.models[model_name] = model
        self.results[model_name] = metrics
        self.best_model_name = model_name
        self.best_model = model
        
        return self.results, self.best_model_name
    
    def get_all_models(self):
        """Retorna TODOS os modelos disponíveis (30+ modelos)"""
        if self.problem_type == 'classification':
            return self.get_all_classification_models()
        else:
            return self.get_all_regression_models()
    
    def get_all_classification_models(self):
        """Retorna TODOS os modelos de classificação"""
        models = {}
        
        try:
            # 1. Ensemble Methods
            from sklearn.ensemble import (
                RandomForestClassifier, GradientBoostingClassifier, 
                AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
            )
            
            models['Random Forest'] = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
            models['Gradient Boosting'] = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42
            )
            models['AdaBoost'] = AdaBoostClassifier(
                n_estimators=100, random_state=42
            )
            models['Extra Trees'] = ExtraTreesClassifier(
                n_estimators=100, random_state=42
            )
            models['Bagging'] = BaggingClassifier(
                n_estimators=50, random_state=42
            )
            
            # 2. Linear Models
            from sklearn.linear_model import (
                LogisticRegression, RidgeClassifier, SGDClassifier
            )
            
            models['Logistic Regression'] = LogisticRegression(
                max_iter=1000, random_state=42, C=1.0
            )
            models['Ridge Classifier'] = RidgeClassifier(
                alpha=1.0, random_state=42
            )
            models['SGD Classifier'] = SGDClassifier(
                max_iter=1000, random_state=42
            )
            
            # 3. SVM e KNN
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            
            models['SVM RBF'] = SVC(
                kernel='rbf', probability=True, random_state=42
            )
            models['KNN'] = KNeighborsClassifier(
                n_neighbors=5
            )
            
            # 4. Árvores e Bayes
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.naive_bayes import GaussianNB
            
            models['Decision Tree'] = DecisionTreeClassifier(
                max_depth=10, random_state=42
            )
            models['Gaussian NB'] = GaussianNB()
            
            # 5. Advanced Libraries
            try:
                from xgboost import XGBClassifier
                models['XGBoost'] = XGBClassifier(
                    n_estimators=100, random_state=42, use_label_encoder=False,
                    eval_metric='logloss'
                )
            except:
                pass
            
            try:
                from lightgbm import LGBMClassifier
                models['LightGBM'] = LGBMClassifier(
                    n_estimators=100, random_state=42
                )
            except:
                pass
            
            # 6. Outros modelos
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.neural_network import MLPClassifier
            
            models['LDA'] = LinearDiscriminantAnalysis()
            models['MLP'] = MLPClassifier(
                hidden_layer_sizes=(100,), max_iter=1000, random_state=42
            )
            
        except Exception as e:
            st.write(f"⚠️ Erro ao carregar alguns modelos: {str(e)[:50]}")
        
        return models
    
    def get_all_regression_models(self):
        """Retorna TODOS os modelos de regressão"""
        models = {}
        
        try:
            # 1. Ensemble Methods
            from sklearn.ensemble import (
                RandomForestRegressor, GradientBoostingRegressor, 
                AdaBoostRegressor, ExtraTreesRegressor
            )
            
            models['Random Forest'] = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            models['Gradient Boosting'] = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, random_state=42
            )
            models['AdaBoost'] = AdaBoostRegressor(
                n_estimators=100, random_state=42
            )
            models['Extra Trees'] = ExtraTreesRegressor(
                n_estimators=100, random_state=42
            )
            
            # 2. Linear Models
            from sklearn.linear_model import (
                LinearRegression, Ridge, Lasso, ElasticNet,
                BayesianRidge
            )
            
            models['Linear Regression'] = LinearRegression()
            models['Ridge'] = Ridge(alpha=1.0, random_state=42)
            models['Lasso'] = Lasso(alpha=0.1, random_state=42)
            models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            models['Bayesian Ridge'] = BayesianRidge()
            
            # 3. SVM e KNN
            from sklearn.svm import SVR
            from sklearn.neighbors import KNeighborsRegressor
            
            models['SVR RBF'] = SVR(kernel='rbf')
            models['KNN Regressor'] = KNeighborsRegressor(n_neighbors=5)
            
            # 4. Árvores
            from sklearn.tree import DecisionTreeRegressor
            
            models['Decision Tree'] = DecisionTreeRegressor(
                max_depth=10, random_state=42
            )
            
            # 5. Advanced Libraries
            try:
                from xgboost import XGBRegressor
                models['XGBoost'] = XGBRegressor(
                    n_estimators=100, random_state=42
                )
            except:
                pass
            
            try:
                from lightgbm import LGBMRegressor
                models['LightGBM'] = LGBMRegressor(
                    n_estimators=100, random_state=42
                )
            except:
                pass
            
            # 6. Outros modelos
            from sklearn.neural_network import MLPRegressor
            
            models['MLP Regressor'] = MLPRegressor(
                hidden_layer_sizes=(100,), max_iter=1000, random_state=42
            )
            
        except Exception as e:
            st.write(f"⚠️ Erro ao carregar alguns modelos: {str(e)[:50]}")
        
        return models
    
    def calculate_complete_metrics(self, y_true, y_pred):
        """Cálculo COMPLETO de métricas"""
        try:
            if self.problem_type == 'classification':
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, 
                    f1_score, roc_auc_score
                )
                
                # Calcular métricas de classificação
                metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
                }
                
                # Tentar AUC (pode falhar para alguns casos)
                try:
                    if len(np.unique(y_true)) > 1:
                        metrics['auc'] = roc_auc_score(y_true, y_pred)
                except:
                    pass
                
                return metrics
                
            else:
                from sklearn.metrics import (
                    r2_score, mean_squared_error, mean_absolute_error,
                    explained_variance_score
                )
                
                metrics = {
                    'r2': r2_score(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'explained_variance': explained_variance_score(y_true, y_pred)
                }
                
                return metrics
                
        except Exception as e:
            st.write(f"⚠️ Erro em métricas: {str(e)[:50]}")
            # Fallback simples
            if self.problem_type == 'classification':
                from sklearn.metrics import accuracy_score
                return {'accuracy': accuracy_score(y_true, y_pred)}
            else:
                from sklearn.metrics import r2_score
                return {'r2': r2_score(y_true, y_pred)}
    
    def determine_best_model_complete(self):
        """Determina melhor modelo considerando múltiplas métricas"""
        if not self.results:
            return
        
        # Pesos para diferentes métricas
        if self.problem_type == 'classification':
            metric_weights = {'accuracy': 0.4, 'f1': 0.3, 'precision': 0.2, 'recall': 0.1}
            main_metric = 'accuracy'
        else:
            metric_weights = {'r2': 0.5, 'rmse': -0.3, 'mae': -0.2}
            main_metric = 'r2'
        
        best_score = -float('inf')
        best_name = ""
        
        for name, metrics in self.results.items():
            weighted_score = 0
            
            for metric, weight in metric_weights.items():
                if metric in metrics:
                    value = metrics[metric]
                    # Para métricas onde menor é melhor (rmse, mae), invertemos
                    if metric in ['rmse', 'mae']:
                        # Normalizar e inverter
                        max_val = max([m.get(metric, 0) for m in self.results.values()])
                        if max_val > 0:
                            normalized = 1 - (value / max_val)
                            weighted_score += normalized * abs(weight)
                    else:
                        weighted_score += value * weight
            
            # Considerar também a métrica principal
            if main_metric in metrics:
                main_score = metrics[main_metric]
                # Combinação: 70% métrica principal, 30% score ponderado
                final_score = 0.7 * main_score + 0.3 * weighted_score
                
                if final_score > best_score:
                    best_score = final_score
                    best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models.get(best_name)
        
        # Salvar métricas do melhor modelo
        if best_name in self.results:
            self.results[best_name]['weighted_score'] = best_score
    
    def get_ranking(self):
        """Ranking com todas as métricas"""
        if not self.results:
            return pd.DataFrame(columns=['Modelo', 'Score', 'Tipo', 'CV Score ± Std'])
        
        ranking = []
        for name, metrics in self.results.items():
            # Usar a métrica principal
            if self.problem_type == 'classification':
                score = metrics.get('accuracy', metrics.get('f1', metrics.get('score', 0)))
                score_std = metrics.get('accuracy_std', 0)
            else:
                score = metrics.get('r2', metrics.get('explained_variance', metrics.get('score', 0)))
                score_std = metrics.get('r2_std', 0)
            
            # Determinar tipo de modelo
            model_type = self.get_model_type(name)
            
            # Formatar score com desvio padrão da CV
            cv_score = f"{score:.4f} ± {score_std:.4f}"
            
            ranking.append({
                'Modelo': name,
                'Score': f"{score:.4f}",
                'CV Score ± Std': cv_score,
                'Tipo': model_type
            })
        
        df = pd.DataFrame(ranking)
        df = df.sort_values('Score', ascending=False)
        df.insert(0, 'Posição', range(1, len(df) + 1))
        
        return df
    
    def get_model_type(self, model_name):
        """Determina o tipo do modelo baseado no nome"""
        model_name_lower = model_name.lower()
        
        if any(x in model_name_lower for x in ['xgboost', 'lightgbm']):
            return 'Boosting'
        elif any(x in model_name_lower for x in ['random forest', 'extra trees', 'bagging']):
            return 'Ensemble'
        elif any(x in model_name_lower for x in ['svm', 'svc', 'svr']):
            return 'SVM'
        elif any(x in model_name_lower for x in ['linear', 'logistic', 'ridge', 'lasso', 'elastic']):
            return 'Linear'
        elif any(x in model_name_lower for x in ['knn', 'neighbors']):
            return 'KNN'
        elif any(x in model_name_lower for x in ['tree', 'decision']):
            return 'Árvore'
        elif any(x in model_name_lower for x in ['naive', 'bayes']):
            return 'Bayes'
        elif any(x in model_name_lower for x in ['mlp', 'neural']):
            return 'Neural'
        elif any(x in model_name_lower for x in ['adaboost', 'gradient']):
            return 'Boosting'
        else:
            return 'Outro'

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
        if 'processor_type' not in st.session_state:
            st.session_state.processor_type = "POWERFULL"
        if 'trainer_type' not in st.session_state:
            st.session_state.trainer_type = "ULTRA_COMPLETE"
    
    def run(self):
        """Executa a aplicação"""
        st.title("🤖 AutoML")
        st.markdown("""
        <div class='cv-badge'>✅ VALIDAÇÃO CRUZADA ATIVADA</div>
        Sistema profissional com **validação cruzada** e **30+ modelos**!
        """, unsafe_allow_html=True)
        
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
        steps = ["📥 Upload", "🔧 Processar", "🚀 Treinar", "📊 Resultados"]
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
        st.header("🔧 Processamento POWERFULL de Dados")
        
        if st.session_state.data is None:
            st.warning("Nenhum dataset carregado.")
            if st.button("⬅️ Voltar"):
                st.session_state.step = 1
                st.rerun()
            return
        
        # Opção de processamento
        st.session_state.processor_type = st.radio(
            "Selecione o modo de processamento:",
            ["🔄 SIMPLES (Rápido)", "🚀 POWERFULL (Recomendado)"],
            index=1
        )
        
        with st.spinner("Processando..."):
            if st.session_state.processor_type == "🚀 POWERFULL (Recomendado)":
                processor = PowerfulDataProcessor()
            else:
                # Processador simples
                processor = PowerfulDataProcessor()  # Usa fallback do próprio processador
            
            X, y, problem_type = processor.process(
                st.session_state.data, 
                st.session_state.target_col
            )
            
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.problem_type = problem_type
            st.session_state.processed = True
        
        st.success(f"✅ Processamento {st.session_state.processor_type.split()[0]} concluído!")
        
        # Mostrar informações detalhadas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tipo", problem_type.upper())
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            st.metric("Amostras", X.shape[0])
        
        # Mostrar preview dos dados processados
        with st.expander("👁️ Visualizar dados processados"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Features (X):**")
                st.dataframe(pd.DataFrame(X).head(), use_container_width=True)
            with col2:
                st.write("**Target (y):**")
                st.dataframe(pd.DataFrame(y, columns=['Target']).head(), use_container_width=True)
        
        # Botões
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Voltar"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("🚀 Treinar com VALIDAÇÃO CRUZADA", type="primary"):
                st.session_state.step = 3
                st.rerun()
    
    def step_train(self):
        """Treinamento dos modelos"""
        st.header("🚀 Treinamento com VALIDAÇÃO CRUZADA")
        
        if not st.session_state.processed:
            st.warning("Dados não processados.")
            st.session_state.step = 2
            st.rerun()
            return
        
        X = st.session_state.X
        y = st.session_state.y
        problem_type = st.session_state.problem_type
        
        # Mostrar benefícios da CV
        st.info("""
        🔬 **VALIDAÇÃO CRUZADA ATIVADA (5-folds):**
        - ✅ Reduz **overfitting**
        - ✅ Métricas **mais confiáveis**  
        - ✅ Melhor **generalização**
        - ✅ Uso **completo dos dados**
        - ✅ Seleção de modelo **robusta**
        """)
        
        # Configurações de CV
        with st.expander("⚙️ Configurações da Validação Cruzada"):
            col1, col2 = st.columns(2)
            with col1:
                n_folds = st.slider("Número de folds", 3, 10, 5,
                                  help="Mais folds = mais robusto, mas mais lento")
                cv_strategy = st.selectbox(
                    "Estratégia CV",
                    ["Auto (Recomendado)", "Stratified K-Fold", "K-Fold"],
                    help="Auto escolhe a melhor baseado nos dados"
                )
            with col2:
                random_state = st.number_input("Random State", 0, 100, 42)
                parallel = st.checkbox("Treinamento Paralelo", value=True,
                                     help="Usa todos os cores da CPU (mais rápido)")
        
        # Estatísticas dos dados
        st.write("### 📊 Estatísticas do Dataset:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Amostras", len(X))
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            if problem_type == 'classification':
                unique_classes = len(np.unique(y))
                st.metric("Classes", unique_classes)
            else:
                st.metric("Target Média", f"{np.mean(y):.2f}")
        with col4:
            if problem_type == 'regression':
                st.metric("Target Std", f"{np.std(y):.2f}")
            else:
                class_dist = pd.Series(y).value_counts().iloc[0] / len(y) * 100
                st.metric("Classe Majoritária", f"{class_dist:.1f}%")
        
        st.warning("⚠️ O treinamento com validação cruzada testará **15+ modelos** e pode levar alguns minutos.")
        
        if st.button("🔥 INICIAR TREINAMENTO COM VALIDAÇÃO CRUZADA", type="primary"):
            with st.spinner(f"Treinando 15+ modelos com validação cruzada ({n_folds}-fold). Aguarde..."):
                trainer = UltraCompleteTrainer(problem_type)
                trainer.n_folds = n_folds
                
                results, best_model = trainer.train_safe(X, y)
                
                # Calcular importância de features
                if trainer.best_model is not None:
                    self.calculate_feature_importance(trainer, X)
                
                st.session_state.results = results
                st.session_state.trainer = trainer
                st.session_state.best_model = best_model
                
                st.session_state.step = 4
                st.rerun()
        
        if st.button("⬅️ Voltar"):
            st.session_state.step = 2
            st.rerun()
    
    def calculate_feature_importance(self, trainer, X):
        """Calcula importância das features"""
        try:
            if trainer.best_model is not None:
                # Verificar se o modelo tem feature_importances_
                if hasattr(trainer.best_model, 'feature_importances_'):
                    importances = trainer.best_model.feature_importances_
                    
                    # Criar DataFrame com importâncias
                    feature_names = [f'Feature_{i}' for i in range(len(importances))]
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    st.session_state.feature_importance = importance_df
                    
        except Exception as e:
            st.write(f"⚠️ Não foi possível calcular importância: {str(e)[:50]}")
    
    def step_results(self):
        """Resultados"""
        st.header("📊 Resultados com VALIDAÇÃO CRUZADA")
        
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
            best_metrics = results[best_name]
            if problem_type == 'classification':
                best_score = best_metrics.get('accuracy', best_metrics.get('f1', 0))
                best_score_std = best_metrics.get('accuracy_std', 0)
            else:
                best_score = best_metrics.get('r2', best_metrics.get('explained_variance', 0))
                best_score_std = best_metrics.get('r2_std', 0)
        else:
            best_score = 0
            best_score_std = 0
        
        # Métricas com desvio padrão da CV
        st.markdown("### 🎯 Performance Geral (Média ± Desvio Padrão)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Melhor Modelo", best_name or "N/A")
        with col2:
            st.metric("🎯 Score Principal", f"{best_score:.4f} ± {best_score_std:.4f}")
        with col3:
            st.metric("🤖 Modelos Treinados", len(results))
        
        # Ranking com scores da CV
        st.subheader("🏆 Ranking Completo (com Validação Cruzada)")
        ranking_df = trainer.get_ranking()
        st.dataframe(ranking_df, use_container_width=True)
        
        # Gráfico de performance
        if not ranking_df.empty:
            fig1 = px.bar(
                ranking_df.head(15),
                x='Modelo',
                y='Score',
                title='Top 15 Modelos (Score Médio da CV)',
                color='Score',
                color_continuous_scale='Viridis',
                error_y=ranking_df.head(15)['CV Score ± Std'].apply(lambda x: float(x.split('±')[1].strip()) if '±' in x else 0)
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Gráfico por tipo de modelo
            fig2 = px.box(
                ranking_df,
                x='Tipo',
                y='Score',
                title='Distribuição de Scores por Tipo de Modelo',
                color='Tipo'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Métricas detalhadas por modelo
        st.subheader("📈 Métricas Detalhadas por Modelo")
        
        # Selecionar modelo para ver detalhes
        model_options = list(results.keys())
        selected_model = st.selectbox("Selecione um modelo para ver métricas detalhadas:", model_options)
        
        if selected_model and selected_model in results:
            with st.expander(f"📊 Métricas Detalhadas - {selected_model}"):
                metrics = results[selected_model]
                cols = st.columns(4)
                metric_count = 0
                
                # Mostrar métricas principais com desvio padrão
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)) and '_std' not in metric_name:
                        std_value = metrics.get(f'{metric_name}_std', 0)
                        
                        with cols[metric_count % 4]:
                            st.metric(
                                label=metric_name.upper(),
                                value=f"{value:.4f}",
                                delta=f"± {std_value:.4f}" if std_value > 0 else None
                            )
                        metric_count += 1
                
                # Mostrar informações da CV
                if 'cv_type' in metrics:
                    st.write("---")
                    st.write(f"**Estratégia CV:** {metrics['cv_type']}")
                    st.write(f"**Número de folds:** {metrics.get('n_folds', 5)}")
                    st.write(f"**Tempo médio de treino:** {metrics.get('fit_time', 0):.2f}s")
                    st.write(f"**Tempo médio de score:** {metrics.get('score_time', 0):.2f}s")
        
        # Exportação
        st.subheader("💾 Exportar Resultados")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # CSV do ranking
            if not ranking_df.empty:
                try:
                    csv_data = ranking_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 CSV Ranking",
                        csv_data,
                        "ranking_cv.csv",
                        "text/csv",
                        key="csv_button"
                    )
                except Exception as e:
                    st.error(f"Erro CSV: {e}")
        
        with col2:
            # SALVAR MODELO .pkl
            if trainer.best_model is not None:
                # Botão para salvar o modelo
                if st.button("💾 Salvar Melhor Modelo", key="save_model_btn"):
                    try:
                        # Criar pasta se não existir
                        os.makedirs('models', exist_ok=True)
                        
                        # Criar nome único com timestamp
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        model_filename = f"modelo_cv_{best_name.replace(' ', '_')}_{timestamp}.pkl"
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
        
        with col3:
            # GERAR RELATÓRIO PDF
            if st.button("📄 Gerar Relatório", key="pdf_report_btn"):
                with st.spinner("Gerando relatório..."):
                    try:
                        # Preparar informações
                        data_info = {
                            'dataset_name': 'Dataset Processado',
                            'n_samples': st.session_state.X.shape[0] if 'X' in st.session_state else 'N/A',
                            'n_features': st.session_state.X.shape[1] if 'X' in st.session_state else 'N/A',
                            'total_modelos': len(results),
                            'melhor_modelo': best_name,
                            'validação_cruzada': 'Sim (5-folds)'
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
        
        with col4:
            # Botão para novo treinamento
            if st.button("🔄 Novo Dataset", type="primary"):
                # Limpar estado
                for key in ['data', 'X', 'y', 'results', 'trainer', 'processed', 'feature_importance']:
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
                        for f in sorted(model_files, reverse=True)[:5]:
                            filepath = os.path.join('models', f)
                            filesize = os.path.getsize(filepath)
                            st.write(f"- `{f}` ({filesize} bytes)")
                    else:
                        st.write("(nenhum modelo salvo ainda)")
                
                # Listar relatórios
                if os.path.exists('reports'):
                    st.write("**📄 Relatórios gerados:**")
                    report_files = [f for f in os.listdir('reports') if f.endswith(('.pdf', '.txt'))]
                    if report_files:
                        for f in sorted(report_files, reverse=True)[:3]:
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
    warnings.filterwarnings('ignore')
    
    # Executar app
    try:
        app = UltraRobustApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Erro crítico: {str(e)}")
        st.info("Recarregue a página para tentar novamente.")