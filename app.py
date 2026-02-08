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
# ========== DETECTOR INTELIGENTE DE TARGET ==========
    """Detecta automaticamente a coluna target"""
    
    @staticmethod
    def detect(data, user_hint=None, auto_mode=True):
        """
        Detecta target automática ou manualmente
        Retorna: (target_col, X, y, problem_type, confidence)
        """
        
        if not auto_mode and user_hint and user_hint in data.columns:
            # Modo manual com dica válida
            return user_hint, data.drop(columns=[user_hint]), data[user_hint], None, 1.0
        
        elif auto_mode:
            # Modo automático inteligente
            return TargetDetector._detect_auto(data, user_hint)
        else:
            # Modo manual completo
            return TargetDetector._detect_manual(data)
    
    @staticmethod
    def _detect_auto(data, user_hint=None):
        """Detecção automática inteligente"""
        import streamlit as st
        
        st.info("🔍 Analisando dataset para detectar target automaticamente...")
        
        # Analisar todas as colunas
        scores = {}
        for col in data.columns:
            scores[col] = TargetDetector._score_column(data[col], col)
        
        # Ordenar
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top 3 candidatos
        top_candidates = [col for col, score in sorted_scores[:3] if score > 0.3]
        
        if not top_candidates:
            st.warning("⚠️ Não consegui detectar target automático.")
            # Usar última coluna
            target_col = data.columns[-1]
            confidence = 0.1
        else:
            # Mostrar opções
            st.write("🎯 **Candidatos detectados:**")
            for i, (col, score) in enumerate(sorted_scores[:5]):
                st.write(f"{i+1}. **{col}** (score: {score:.2f})")
            
            # Selecionar
            target_col = top_candidates[0]  # Primeiro como default
            confidence = scores[target_col]
        
        # Separar dados
        X = data.drop(columns=[target_col]).copy()
        y = data[target_col].copy()
        
        # Detectar tipo de problema
        problem_type = TargetDetector._detect_problem_type(y)
        
        st.success(f"✅ Target: **{target_col}** | Tipo: **{problem_type}**")
        
        return target_col, X, y, problem_type, confidence
    
    @staticmethod
    def _detect_manual(data):
        """Detecção manual"""
        import streamlit as st
        
        st.write("### ✏️ Seleção Manual")
        target_options = data.columns.tolist()
        target_col = st.selectbox(
            "Selecione a coluna target:",
            target_options,
            index=len(target_options)-1
        )
        
        # Separar dados
        X = data.drop(columns=[target_col]).copy()
        y = data[target_col].copy()
        
        # Detectar tipo
        problem_type = TargetDetector._detect_problem_type(y)
        
        return target_col, X, y, problem_type, 0.5
    
    @staticmethod
    def _score_column(column, col_name):
        """Pontua se uma coluna parece ser target"""
        score = 0
        
        try:
            n = len(column)
            n_unique = column.nunique()
            unique_ratio = n_unique / n
            
            # 1. Nome sugere target
            target_keywords = ['target', 'label', 'class', 'score', 'output', 
                              'result', 'y', 'price', 'value', 'rating']
            if any(kw in col_name.lower() for kw in target_keywords):
                score += 0.4
            
            # 2. Nome sugere NÃO target
            not_target_keywords = ['id', 'code', 'index', 'key', 'date', 'time']
            if any(kw in col_name.lower() for kw in not_target_keywords):
                score -= 0.3
            
            # 3. Poucos valores únicos (classificação)
            if n_unique <= 10:
                score += 0.3
            
            # 4. Muitos valores únicos (regressão)
            elif unique_ratio > 0.8:
                score += 0.2
            
            # 5. Distribuição não-uniforme
            if n_unique > 1:
                value_counts = column.value_counts(normalize=True)
                entropy = -sum(p * np.log(p) for p in value_counts if p > 0)
                max_entropy = np.log(n_unique)
                if max_entropy > 0:
                    if entropy / max_entropy < 0.7:
                        score += 0.2
            
            # 6. Muitos missing values penaliza
            missing_ratio = column.isna().sum() / n
            if missing_ratio > 0.3:
                score -= 0.2
            
            # Limitar entre 0 e 1
            score = max(0, min(1, score))
            
        except:
            score = 0
        
        return round(score, 3)
    
    @staticmethod
    def _detect_problem_type(y):
        """Detecta se é classificação ou regressão"""
        try:
            # Tentar converter para numérico
            y_numeric = pd.to_numeric(y, errors='coerce')
            not_na = y_numeric.notna().sum()
            
            # Se poucos são numéricos, é classificação
            if not_na / len(y) < 0.8:
                return 'classification'
            
            # Analisar valores únicos
            y_clean = y_numeric.dropna()
            unique_vals = len(y_clean.unique())
            
            # Regras
            if unique_vals <= 10:
                if all(y_clean.astype(int) == y_clean):
                    return 'classification'
                else:
                    return 'regression'
            elif unique_vals <= 30:
                # Verificar se parece categórico
                value_counts = y_clean.value_counts(normalize=True)
                if (value_counts > 0.3).any() and all(y_clean.astype(int) == y_clean):
                    return 'classification'
                else:
                    return 'regression'
            else:
                return 'regression'
                
        except:
            # Fallback
            try:
                if y.dtype == 'object' or len(y.unique()) <= 10:
                    return 'classification'
                else:
                    return 'regression'
            except:
                return 'regression'


# ========== PROCESSAMENTO DE DADOS POWERFULL PRO (ATUALIZADA) ==========
class PowerfulDataProcessor:
    """Processador de dados avançado com feature engineering e detecção automática"""
    
    def __init__(self):
        self.scaler = None
        self.encoders = {}
        self.imputer = None
        self.selected_features = []
        self.target_col = None
        self.problem_type = None
    
    def process(self, data=None, target_col=None, X=None, y=None, problem_type=None, auto_detect=True):
        """
        Processamento POWERFULL FLEXÍVEL
        
        Aceita múltiplos formatos:
        1. data + target_col (seu formato original)
        2. data apenas (detecta automaticamente)
        3. X e y já separados
        """
        try:
            # CASO A: X e y já separados (após detecção automática)
            if X is not None and y is not None:
                self.target_col = "target"  # Placeholder
                self.problem_type = problem_type or self.detect_problem_type_smart(y)
                
                # Processar features
                X_processed = self.powerful_preprocessing(X)
                X_engineered = self.advanced_feature_engineering(X_processed)
                X_final = self.smart_feature_selection(X_engineered, y, self.problem_type)
                
                # Processar target
                if self.problem_type == 'classification':
                    y_processed = self.process_target(y)
                else:
                    y_processed = self.process_target_regression(y)
                
                return X_final, y_processed, self.problem_type
            
            # CASO B: Data completo (seu formato original)
            elif data is not None:
                # Se target_col não fornecido, detectar automaticamente
                if target_col is None and auto_detect:
                    st.info("🔍 Detectando target automaticamente...")
                    detector = TargetDetector()
                    target_col, X, y, problem_type, confidence = detector.detect(data, auto_mode=True)
                    
                    # Salvar informações
                    self.target_col = target_col
                    self.problem_type = problem_type
                    
                    # Continuar com processamento normal
                    X_processed = self.powerful_preprocessing(X)
                    X_engineered = self.advanced_feature_engineering(X_processed)
                    X_final = self.smart_feature_selection(X_engineered, y, problem_type)
                    
                    if problem_type == 'classification':
                        y_processed = self.process_target(y)
                    else:
                        y_processed = self.process_target_regression(y)
                    
                    return X_final, y_processed, problem_type
                
                # Se target_col fornecido (seu método original)
                else:
                    # 1. Separar X e y
                    if target_col in data.columns:
                        X = data.drop(columns=[target_col]).copy()
                        y = data[target_col].copy()
                    else:
                        X = data.iloc[:, :-1].copy()
                        y = data.iloc[:, -1].copy()
                    
                    # 2. Detectar tipo de problema
                    problem_type = self.detect_problem_type_smart(y)
                    self.problem_type = problem_type
                    self.target_col = target_col
                    
                    # 3. Pré-processamento
                    X_processed = self.powerful_preprocessing(X)
                    
                    # 4. Feature Engineering
                    X_engineered = self.advanced_feature_engineering(X_processed)
                    
                    # 5. Seleção de Features
                    X_final = self.smart_feature_selection(X_engineered, y, problem_type)
                    
                    # 6. Processar target
                    if problem_type == 'classification':
                        y_processed = self.process_target(y)
                    else:
                        y_processed = self.process_target_regression(y)
                    
                    return X_final, y_processed, problem_type
            
            else:
                raise ValueError("❌ Dados insuficientes. Forneça 'data' ou 'X' e 'y'")
            
        except Exception as e:
            st.error(f"Erro no processamento POWERFULL: {str(e)}")
            return self.simple_fallback(data if data is not None else X, target_col)
    
    def detect_problem_type_smart(self, y):
        """Detecção INTELIGENTE de tipo de problema (MANTIDO IGUAL)"""
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
        """Pré-processamento avançado (MANTIDO IGUAL)"""
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
        """Feature engineering avançado (MANTIDO IGUAL)"""
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
        """Seleção inteligente de features (MANTIDO IGUAL)"""
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
        """Processar target para classificação (MANTIDO IGUAL)"""
        # Se for string, converter para numérico
        if y.dtype == 'object':
            y_encoded, _ = pd.factorize(y)
            return pd.Series(y_encoded)
        
        # Se for numérico mas com poucas classes, garantir que seja inteiro
        if len(y.unique()) <= 10:
            return y.astype(int)
        
        return y
    
    def process_target_regression(self, y):
        """Processar target para regressão (MANTIDO IGUAL)"""
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
        """Fallback simples (MANTIDO IGUAL)"""
        try:
            if data is None:
                # Fallback genérico
                n_samples = 100
                X = pd.DataFrame({
                    'feature_1': np.random.randn(n_samples),
                    'feature_2': np.random.randn(n_samples),
                })
                y = pd.Series(np.random.randint(0, 2, n_samples))
                return X, y, 'classification'
            
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

# ========== TREINAMENTO ULTRA COMPLETO COM VALIDAÇÃO CRUZADA + HOLDOUT ==========
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

# ========== DETECTOR INTELIGENTE DE TARGET (COMPLETO) ==========
class TargetDetector:
    """Detecta automaticamente a coluna target"""
    
    @staticmethod
    def detect_target(data, user_hint=None):
        """
        Detecta a coluna target automaticamente com inteligência
        Retorna: (target_col, X, y, confidence_score, problem_type)
        """
        import streamlit as st
        
        if user_hint and user_hint in data.columns:
            # Usuário forneceu dica válida
            X = data.drop(columns=[user_hint]).copy()
            y = data[user_hint].copy()
            problem_type = TargetDetector.detect_problem_type(y)
            return user_hint, X, y, 1.0, problem_type
        
        st.info("🔍 Analisando dataset para detectar target automaticamente...")
        
        # 1. Analisar cada coluna
        scores = {}
        
        for col in data.columns:
            score = TargetDetector.analyze_column(data[col], col)
            scores[col] = score
        
        # 2. Ordenar por score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 3. Mostrar análise
        st.write("📊 **Análise automática:**")
        analysis_df = pd.DataFrame(sorted_scores, columns=['Coluna', 'Score Target'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(analysis_df.head(10), use_container_width=True)
        
        with col2:
            # Visualizar distribuição do top candidate
            if len(sorted_scores) > 0:
                top_col = sorted_scores[0][0]
                try:
                    fig = px.histogram(data, x=top_col, title=f"Distribuição: {top_col}")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.write(f"*Não foi possível criar gráfico para {top_col}*")
        
        # 4. Selecionar target (usuário confirma)
        top_candidates = [col for col, score in sorted_scores[:3] if score > 0.3]
        
        if not top_candidates:
            st.warning("⚠️ Não consegui detectar target automaticamente.")
            # Usar última coluna como fallback
            target_col = data.columns[-1]
            confidence = 0.1
        else:
            # Let user choose from top candidates
            st.write("🎯 **Candidatos a target (escolha ou confirme):**")
            target_col = st.selectbox(
                "Selecione a coluna target:",
                options=top_candidates + ["⚠️ Nenhuma das acima"],
                index=0,
                key="auto_target_select"
            )
            
            if target_col == "⚠️ Nenhuma das acima":
                # Listar todas as colunas
                target_col = st.selectbox(
                    "Selecione manualmente:",
                    options=data.columns.tolist(),
                    index=len(data.columns)-1,
                    key="manual_fallback_select"
                )
                confidence = 0.5
            else:
                confidence = scores[target_col]
        
        # 5. Separar X e y
        X = data.drop(columns=[target_col]).copy()
        y = data[target_col].copy()
        
        # 6. Detectar tipo de problema
        problem_type = TargetDetector.detect_problem_type(y)
        
        st.success(f"✅ Target detectado: **{target_col}** (confiança: {confidence:.2f})")
        st.success(f"📊 Tipo de problema: **{problem_type.upper()}**")
        st.write(f"📐 Dimensões: X={X.shape}, y={y.shape}")
        
        return target_col, X, y, confidence, problem_type
    
    @staticmethod
    def analyze_column(column, col_name):
        """Analisa uma coluna e retorna score de ser target"""
        score = 0
        
        try:
            n_unique = column.nunique()
            n_total = len(column)
            unique_ratio = n_unique / n_total if n_total > 0 else 0
            
            # REGRAS PARA SER TARGET:
            
            # 1. Poucos valores únicos (possível classificação)
            if n_unique <= 10:
                score += 0.3
            
            # 2. Muitos valores únicos (possível regressão)
            elif unique_ratio > 0.9:
                score += 0.2
            
            # 3. Nome da coluna dá dica
            target_keywords = ['target', 'label', 'class', 'score', 'rating', 
                              'price', 'value', 'output', 'result', 'y']
            col_lower = col_name.lower()
            if any(keyword in col_lower for keyword in target_keywords):
                score += 0.4
            
            # 4. Distribuição não-uniforme (típico de target)
            if n_unique > 1:
                value_counts = column.value_counts(normalize=True)
                try:
                    entropy = -sum(p * np.log(p) for p in value_counts if p > 0)
                    max_entropy = np.log(n_unique)
                    if max_entropy > 0:
                        normalized_entropy = entropy / max_entropy
                        # Targets tendem a ter baixa entropia (classes desbalanceadas)
                        if normalized_entropy < 0.7:
                            score += 0.2
                except:
                    pass
            
            # 5. Verificar se parece ser numérico contínuo (regressão)
            if pd.api.types.is_numeric_dtype(column):
                # Altos valores podem ser preço, score, etc
                try:
                    if column.abs().max() > 1000:
                        score += 0.1
                except:
                    pass
            
            # 6. Penalizar colunas com muitos missing
            missing_ratio = column.isna().sum() / n_total if n_total > 0 else 0
            if missing_ratio > 0.3:
                score -= 0.3
            
            # 7. Penalizar colunas que parecem ser IDs
            if any(x in col_lower for x in ['id', 'code', 'num', 'index', 'key']):
                score -= 0.4
            
            # 8. Se todos valores são únicos, provavelmente é ID
            if n_unique == n_total and n_total > 100:
                score -= 0.5
            
            # 10. Verificar se parece ser data/hora
            date_keywords = ['date', 'time', 'day', 'month', 'year']
            if any(x in col_lower for x in date_keywords):
                score -= 0.3
            
            # Normalizar score entre 0 e 1
            score = max(0, min(1, score))
            
        except Exception as e:
            score = 0
        
        return round(score, 3)
    
    @staticmethod
    def detect_problem_type(y):
        """Detecta se é classificação ou regressão de forma robusta"""
        try:
            # Converter para numérico se possível
            y_numeric = pd.to_numeric(y, errors='coerce')
            not_na = y_numeric.notna().sum()
            
            # Se menos de 80% são numéricos, é classificação
            if not_na / len(y) < 0.8:
                return 'classification'
            
            # Agora trabalhar com os valores numéricos
            y_clean = y_numeric.dropna()
            if len(y_clean) == 0:
                return 'classification'  # Fallback
            
            unique_vals = len(y_clean.unique())
            
            # REGRAS AVANÇADAS:
            if unique_vals <= 5:
                # Poucos valores únicos - verificar se são inteiros
                try:
                    if all(y_clean.astype(int) == y_clean):
                        return 'classification'
                    else:
                        # Poucos valores mas não inteiros - regressão
                        return 'regression'
                except:
                    return 'classification'
            
            elif unique_vals <= 20:
                # Verificar distribuição
                value_counts = y_clean.value_counts(normalize=True)
                # Se algum valor tem > 25% dos dados, pode ser classificação
                if (value_counts > 0.25).any():
                    try:
                        # Verificar se valores parecem categóricos
                        if all(y_clean.astype(int) == y_clean):
                            return 'classification'
                        else:
                            return 'regression'
                    except:
                        return 'classification'
                else:
                    return 'regression'
            else:
                return 'regression'
                
        except Exception as e:
            # Fallback simples
            try:
                if hasattr(y, 'dtype'):
                    if y.dtype == 'object' or len(y.unique()) <= 10:
                        return 'classification'
                    else:
                        return 'regression'
                else:
                    # Para arrays numpy
                    unique_vals = len(np.unique(y))
                    if unique_vals <= 10:
                        return 'classification'
                    else:
                        return 'regression'
            except:
                return 'regression'  # Fallback final


# ========== APLICAÇÃO PRINCIPAL COM FIXES ==========
class UltraRobustApp:
    def __init__(self):
        # INICIALIZAÇÃO ROBUSTA
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            st.session_state.step = 1
            st.session_state.data = None
            st.session_state.processed = False
            st.session_state.processor_type = "POWERFULL"
            st.session_state.trainer_type = "ULTRA_COMPLETE"
            st.session_state.last_rerun = time.time()
    
    def safe_rerun(self, delay=0.1):
        """Rerun seguro com delay"""
        current_time = time.time()
        if current_time - st.session_state.last_rerun > 0.5:  # Evitar reruns muito rápidos
            time.sleep(delay)
            st.session_state.last_rerun = current_time
            try:
                st.rerun()
            except:
                st.rerun()
        else:
            # Aguardar um pouco e tentar novamente
            time.sleep(0.5)
            st.rerun()
    
    def run(self):
        """Executa a aplicação com tratamento de erros"""
        try:
            st.title("🤖 AutoML")
            st.markdown("""
            <div class='cv-badge'>✅ VALIDAÇÃO CRUZADA ATIVADA</div>
            Sistema profissional com **validação cruzada** e **30+ modelos**!
            """, unsafe_allow_html=True)
            
            # Progresso
            self.show_progress()
            
            # Conteúdo com try-catch
            try:
                if st.session_state.step == 1:
                    self.step_upload()
                elif st.session_state.step == 2:
                    self.step_process()
                elif st.session_state.step == 3:
                    self.step_train()
                elif st.session_state.step == 4:
                    self.step_results()
            except Exception as e:
                st.error(f"❌ Erro na etapa {st.session_state.step}: {str(e)}")
                if st.button("🔄 Reiniciar Aplicação", key="restart_app_error"):
                    self.reset_app()
                
        except Exception as e:
            st.error(f"❌ Erro crítico: {str(e)}")
            st.info("Recarregue a página para tentar novamente.")
    
    def show_progress(self):
        """Barra de progresso simples"""
        steps = ["📥 Upload", "🔧 Processar", "🚀 Treinar", "📊 Resultados"]
        current = st.session_state.step - 1
        
        # Usar HTML para evitar recriação de componentes
        html = """
        <div style="display: flex; justify-content: space-between; margin: 20px 0;">
        """
        
        for i, step in enumerate(steps):
            if i < current:
                html += f'<div style="padding: 10px; background: #4CAF50; color: white; border-radius: 5px; text-align: center; flex: 1; margin: 0 5px;">{step} ✅</div>'
            elif i == current:
                html += f'<div style="padding: 10px; background: #2196F3; color: white; border-radius: 5px; text-align: center; flex: 1; margin: 0 5px;">{step}</div>'
            else:
                html += f'<div style="padding: 10px; background: #f0f0f0; color: #666; border-radius: 5px; text-align: center; flex: 1; margin: 0 5px;">{step}</div>'
        
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
    
    def step_upload(self):
        """Upload do dataset SIMPLIFICADO para evitar erro do Streamlit"""
        st.header("📥 Upload do Dataset")
        
        # Usar container único para evitar recriação
        with st.container():
            uploaded_file = st.file_uploader(
                "Escolha um arquivo CSV", 
                type=['csv', 'txt', 'xlsx'],
                help="Suporta CSV, TXT e Excel",
                key="main_file_uploader"
            )
        
        if uploaded_file:
            try:
                # Ler arquivo
                if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                else:
                    data = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Dataset carregado: {data.shape[0]} linhas × {data.shape[1]} colunas")
                
                # Preview simples
                if st.checkbox("👁️ Visualizar dados", key="show_preview_upload"):
                    st.dataframe(data.head(), use_container_width=True)
                
                # SELEÇÃO DE TARGET SIMPLIFICADA
                st.subheader("🎯 Seleção do Target")
                
                # Opção direta - sem rádio para evitar recriação
                use_auto = st.checkbox("🤖 Usar detecção automática", value=True, key="use_auto_detect")
                
                if use_auto:
                    # Tenta detecção automática
                    try:
                        target_col, X, y, confidence, problem_type = TargetDetector.detect_target(data)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🎯 Target", target_col)
                        with col2:
                            st.metric("📊 Tipo", problem_type.upper())
                        
                        # Salvar no estado
                        st.session_state.target_col = target_col
                        st.session_state.X = X
                        st.session_state.y = y
                        st.session_state.problem_type = problem_type
                        st.session_state.auto_detected = True
                        st.session_state.data = data
                        
                        st.success("✅ Target detectado automaticamente!")
                        
                    except Exception as e:
                        st.error(f"❌ Detecção automática falhou: {str(e)}")
                        st.info("Por favor, selecione manualmente:")
                        use_auto = False
                
                if not use_auto or not st.session_state.get('auto_detected', False):
                    # Seleção manual simples
                    target_options = data.columns.tolist()
                    default_idx = len(target_options) - 1
                    
                    # Tentar encontrar um bom default
                    for i, col in enumerate(target_options):
                        col_lower = col.lower()
                        if any(kw in col_lower for kw in ['target', 'label', 'class', 'y', 'price', 'value']):
                            default_idx = i
                            break
                    
                    target_col = st.selectbox(
                        "Selecione a coluna target:",
                        target_options,
                        index=default_idx,
                        key="manual_target_selector_upload"
                    )
                    
                    # Separar dados
                    X = data.drop(columns=[target_col]).copy()
                    y = data[target_col].copy()
                    
                    # Detectar tipo
                    try:
                        problem_type = TargetDetector.detect_problem_type(y)
                    except:
                        # Fallback simples
                        if y.dtype == 'object' or len(y.unique()) <= 10:
                            problem_type = 'classification'
                        else:
                            problem_type = 'regression'
                    
                    # Salvar no estado
                    st.session_state.target_col = target_col
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.problem_type = problem_type
                    st.session_state.auto_detected = False
                    st.session_state.data = data
                    
                    st.success(f"✅ Target selecionado: {target_col}")
                    st.success(f"📊 Tipo: {problem_type.upper()}")
                
                # Botões SIMPLES e ÚNICOS
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Novo Upload", type="secondary", key="new_upload_simple_btn"):
                        # Limpar estado de forma segura
                        keys_to_keep = ['app_initialized', 'last_rerun']
                        keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]
                        for key in keys_to_remove:
                            del st.session_state[key]
                        time.sleep(0.5)
                        st.rerun()
                
                with col2:
                    if st.button("🔧 Continuar →", type="primary", key="continue_upload_btn"):
                        if 'target_col' not in st.session_state:
                            st.error("❌ Selecione um target primeiro!")
                        else:
                            # Verificar se temos dados suficientes
                            if len(st.session_state.X) < 10:
                                st.error("❌ Muito poucas amostras (mínimo 10)")
                            else:
                                st.session_state.step = 2
                                time.sleep(0.5)
                                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Erro ao processar arquivo: {str(e)}")
                
                # Tentar fallback
                try:
                    data = pd.read_csv(uploaded_file, encoding='latin-1')
                    st.success(f"✅ Carregado com encoding alternativo")
                    # Repetir o processo
                    st.rerun()
                except:
                    st.error("❌ Não foi possível ler o arquivo.")
    
    def step_process(self):
        """Processamento SIMPLIFICADO"""
        st.header("🔧 Processamento de Dados")
        
        if 'data' not in st.session_state or st.session_state.data is None:
            st.warning("⚠️ Nenhum dataset carregado.")
            if st.button("⬅️ Voltar para Upload", key="back_to_upload_process"):
                st.session_state.step = 1
                time.sleep(0.5)
                st.rerun()
            return
        
        # Mostrar informações atuais
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Amostras", st.session_state.data.shape[0])
        with col2:
            st.metric("Features", st.session_state.data.shape[1] - 1)
        with col3:
            st.metric("Target", st.session_state.target_col)
        
        # Processamento DIRETO - sem opções complexas
        if st.button("mento POWERFULL", type="primary", key="process_execute_btn"):
            with st.spinner("Processando dados..."):
                try:
                    processor = PowerfulDataProcessor()
                    
                    X, y, problem_type = processor.process(
                        st.session_state.data, 
                        st.session_state.target_col
                    )
                    
                    # Atualizar estado
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.problem_type = problem_type
                    st.session_state.processed = True
                    
                    st.success("✅ Processamento concluído!")
                    
                    # Mostrar resultados
                    with st.expander("📋 Resultados do Processamento"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Features (X):**")
                            st.write(f"- Dimensões: {X.shape}")
                            st.write(f"- Tipo: {type(X)}")
                        with col2:
                            st.write("**Target (y):**")
                            st.write(f"- Dimensões: {y.shape}")
                            st.write(f"- Tipo: {problem_type.upper()}")
                            if problem_type == 'classification':
                                st.write(f"- Classes: {len(np.unique(y))}")
                    
                    # Delay antes de mostrar botão próximo
                    time.sleep(1)
                    
                except Exception as e:
                    st.error(f"❌ Erro no processamento: {str(e)}")
                    # Fallback
                    try:
                        X = st.session_state.data.drop(columns=[st.session_state.target_col]).values
                        y = st.session_state.data[st.session_state.target_col].values
                        
                        st.session_state.X = X
                        st.session_state.y = y
                        st.session_state.processed = True
                        
                        st.success("✅ Processamento simples realizado")
                    except:
                        st.error("❌ Não foi possível processar os dados.")
        
        # Botão para próximo passo (só aparece após processamento)
        if st.session_state.get('processed', False):
            st.markdown("---")
            if st.button("🚀 Ir para Treinamento →", type="primary", key="go_to_train_btn"):
                st.session_state.step = 3
                time.sleep(0.5)
                st.rerun()
        
        # Botão para voltar
        if st.button("⬅️ Voltar", key="back_from_process_btn"):
            st.session_state.step = 1
            time.sleep(0.5)
            st.rerun()
    
    def step_train(self):
        """Treinamento com fix"""
        st.header("🚀 Treinamento com VALIDAÇÃO CRUZADA")
        
        if not st.session_state.get('processed', False):
            st.warning("Dados não processados.")
            if st.button("⬅️ Voltar", key="back_to_process_train"):
                st.session_state.step = 2
                time.sleep(0.1)
                st.rerun()
            return
        
        # Mostrar estatísticas SEM expander com key
        with st.expander("📊 Estatísticas do Dataset"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Amostras", len(st.session_state.X))
            with col2:
                st.metric("Features", st.session_state.X.shape[1])
            with col3:
                if st.session_state.problem_type == 'classification':
                    unique_classes = len(np.unique(st.session_state.y))
                    st.metric("Classes", unique_classes)
                else:
                    st.metric("Target Média", f"{np.mean(st.session_state.y):.2f}")
            with col4:
                if st.session_state.problem_type == 'regression':
                    st.metric("Target Std", f"{np.std(st.session_state.y):.2f}")
                else:
                    class_dist = pd.Series(st.session_state.y).value_counts().iloc[0] / len(st.session_state.y) * 100
                    st.metric("Classe Majoritária", f"{class_dist:.1f}%")
        
        # Configuração em container
        with st.container():
            st.info("🔬 **VALIDAÇÃO CRUZADA ATIVADA (5-folds)**")
            
            # Configurações da CV SEM expander com key
            with st.expander("⚙️ Configurações da Validação Cruzada"):
                col1, col2 = st.columns(2)
                with col1:
                    n_folds = st.slider("Número de folds", 3, 10, 5,
                                      help="Mais folds = mais robusto, mas mais lento",
                                      key="n_folds_slider_train")
                    cv_strategy = st.selectbox(
                        "Estratégia CV",
                        ["Auto (Recomendado)", "Stratified K-Fold", "K-Fold"],
                        help="Auto escolhe a melhor baseado nos dados",
                        key="cv_strategy_select_train"
                    )
                with col2:
                    random_state = st.number_input("Random State", 0, 100, 42, key="random_state_input_train")
                    parallel = st.checkbox("Treinamento Paralelo", value=True,
                                         help="Usa todos os cores da CPU (mais rápido)",
                                         key="parallel_checkbox_train")
            
            st.warning("⚠️ O treinamento testará **15+ modelos** e pode levar alguns minutos.")
            
            # Botão de treinamento PRINCIPAL - apenas um
            if st.button("🔥 INICIAR TREINAMENTO COMPLETO", type="primary", key="start_training_main_btn"):
                self._execute_training()
        
        # Botão de voltar SEPARADO
        if st.button("⬅️ Voltar para Processamento", key="back_to_process_train_2"):
            st.session_state.step = 2
            time.sleep(0.1)
            st.rerun()
    
    def _execute_training(self):
        """Executa treinamento em container separado"""
        with st.spinner("Treinando 15+ modelos..."):
            try:
                X = st.session_state.X
                y = st.session_state.y
                problem_type = st.session_state.problem_type
                
                trainer = UltraCompleteTrainer(problem_type)
                trainer.n_folds = 5
                
                results, best_model = trainer.train_safe(X, y)
                
                # Salvar resultados
                st.session_state.results = results
                st.session_state.trainer = trainer
                st.session_state.best_model = best_model
                
                st.success("✅ Treinamento concluído!")
                
                # Delay antes de avançar
                time.sleep(1)
                st.session_state.step = 4
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Erro no treinamento: {str(e)}")
    
    def step_results(self):
        """Resultados"""
        st.header("📊 Resultados")
        
        if 'results' not in st.session_state:
            st.warning("Nenhum resultado disponível.")
            if st.button("⬅️ Voltar", key="back_to_train_results"):
                st.session_state.step = 3
                time.sleep(0.1)
                st.rerun()
            return
        
        # Conteúdo dos resultados
        try:
            results = st.session_state.results
            trainer = st.session_state.trainer
            problem_type = st.session_state.problem_type
            
            # Melhor modelo
            best_name = trainer.best_model_name
            if best_name and best_name in results:
                best_metrics = results[best_name]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏆 Melhor Modelo", best_name)
                with col2:
                    if problem_type == 'classification':
                        score = best_metrics.get('accuracy', best_metrics.get('f1', 0))
                    else:
                        score = best_metrics.get('r2', best_metrics.get('explained_variance', 0))
                    st.metric("🎯 Score", f"{score:.4f}")
                with col3:
                    st.metric("🤖 Modelos Treinados", len(results))
            
            # Ranking
            with st.expander("🏆 Ranking Completo"):
                ranking_df = trainer.get_ranking()
                st.dataframe(ranking_df, use_container_width=True)
                
                # Gráfico do ranking
                if not ranking_df.empty:
                    fig = px.bar(
                        ranking_df.head(15),
                        x='Modelo',
                        y='Score',
                        title='Top 15 Modelos',
                        color='Score',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Métricas detalhadas
            if 'best_model' in st.session_state and st.session_state.best_model is not None:
                with st.expander("📈 Métricas Detalhadas"):
                    model_options = list(results.keys())
                    selected_model = st.selectbox(
                        "Selecione um modelo para ver métricas detalhadas:", 
                        model_options,
                        key="model_select_detailed_results"
                    )
                    
                    if selected_model in results:
                        metrics = results[selected_model]
                        
                        # Mostrar métricas principais
                        cols = st.columns(4)
                        metric_count = 0
                        
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)) and '_std' not in metric_name:
                                with cols[metric_count % 4]:
                                    st.metric(
                                        label=metric_name.upper(),
                                        value=f"{value:.4f}",
                                        delta=f"± {metrics.get(f'{metric_name}_std', 0):.4f}" if f'{metric_name}_std' in metrics else None
                                    )
                                metric_count += 1
                        
                        # Informações da validação cruzada
                        if 'cv_type' in metrics:
                            st.write("---")
                            st.write(f"**Estratégia CV:** {metrics['cv_type']}")
                            st.write(f"**Número de folds:** {metrics.get('n_folds', 5)}")
                            st.write(f"**Tempo médio de treino:** {metrics.get('fit_time', 0):.2f}s")
                            st.write(f"**Tempo médio de score:** {metrics.get('score_time', 0):.2f}s")
            
            # Exportação
            st.subheader("💾 Exportar Resultados")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV do ranking
                if st.button("📊 Exportar CSV", key="export_csv_results_btn"):
                    try:
                        ranking_df = trainer.get_ranking()
                        csv_data = ranking_df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            "⬇️ Baixar CSV",
                            csv_data,
                            f"ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key="download_csv_results_btn"
                        )
                    except Exception as e:
                        st.error(f"Erro CSV: {e}")
            
            with col2:
                # Salvar modelo
                if st.button("💾 Salvar Modelo", key="save_model_results_btn"):
                    if trainer.best_model is not None:
                        try:
                            os.makedirs('models', exist_ok=True)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            model_filename = f"modelo_{best_name.replace(' ', '_')}_{timestamp}.pkl"
                            model_path = f"models/{model_filename}"
                            
                            joblib.dump(trainer.best_model, model_path)
                            
                            if os.path.exists(model_path):
                                with open(model_path, 'rb') as f:
                                    model_bytes = f.read()
                                
                                st.download_button(
                                    "⬇️ Baixar Modelo",
                                    model_bytes,
                                    model_filename,
                                    "application/octet-stream",
                                    key=f"download_model_{timestamp}"
                                )
                                st.success(f"✅ Modelo salvo: {model_filename}")
                        except Exception as e:
                            st.error(f"❌ Erro ao salvar: {str(e)}")
            
            with col3:
                # Relatório PDF
                if st.button("📄 Gerar Relatório", key="generate_report_btn"):
                    with st.spinner("Gerando relatório..."):
                        try:
                            data_info = {
                                'dataset_name': 'Dataset Processado',
                                'n_samples': st.session_state.X.shape[0] if 'X' in st.session_state else 'N/A',
                                'n_features': st.session_state.X.shape[1] if 'X' in st.session_state else 'N/A',
                            }
                            
                            pdf_path = PDFReportGenerator.generate_report(
                                results, 
                                trainer, 
                                problem_type,
                                data_info
                            )
                            
                            if pdf_path and os.path.exists(pdf_path):
                                with open(pdf_path, 'rb') as f:
                                    file_bytes = f.read()
                                
                                st.download_button(
                                    "⬇️ Baixar Relatório",
                                    file_bytes,
                                    os.path.basename(pdf_path),
                                    "application/pdf",
                                    key="download_report_btn"
                                )
                            else:
                                st.warning("Relatório TXT gerado")
                        except Exception as e:
                            st.error(f"❌ Erro no relatório: {str(e)}")
            
            # Botão para novo treinamento
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("⬅️ Voltar", key="back_to_train_final"):
                    st.session_state.step = 3
                    time.sleep(0.1)
                    st.rerun()
            
            with col2:
                if st.button("🔄 Novo Dataset", type="primary", key="new_dataset_btn"):
                    # Limpar apenas dados de treinamento
                    training_keys = ['results', 'trainer', 'best_model', 'processed', 'X', 'y', 
                                   'problem_type', 'auto_detected', 'target_col', 'data']
                    for key in training_keys:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.session_state.step = 1
                    time.sleep(0.2)
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Erro nos resultados: {str(e)}")
            if st.button("🔄 Reiniciar Aplicação", key="restart_app_results"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    def _clear_state(self):
        """Limpa estado de forma segura"""
        keys_to_preserve = ['app_initialized', 'last_rerun']
        keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_preserve]
        
        for key in keys_to_remove:
            del st.session_state[key]
    
    def _clear_training_state(self):
        """Limpa apenas estado de treinamento"""
        training_keys = ['results', 'trainer', 'best_model', 'processed', 'X', 'y']
        for key in training_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    def reset_app(self):
        """Reinicia aplicação completamente"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ========== EXECUTAR COM TRY-CATCH GLOBAL ==========
if __name__ == "__main__":
    # Configurar para evitar warnings
    warnings.filterwarnings('ignore')
    
    # Executar app com proteção
    try:
        app = UltraRobustApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Erro crítico na aplicação: {str(e)}")
        st.button("🔄 Reiniciar Aplicação", key="restart_app_final", on_click=lambda: st.rerun())