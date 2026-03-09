# Importa a biblioteca Streamlit para criar a interface web
import streamlit as st
# Importa a biblioteca Pandas para manipulação e análise de dados
import pandas as pd
# Importa a biblioteca NumPy para operações numéricas, especialmente com arrays
import numpy as np
# Importa a biblioteca Plotly Express para criar visualizações interativas de forma simples
import plotly.express as px
# Importa a biblioteca Plotly Graph Objects para criar gráficos mais complexos e customizados
import plotly.graph_objects as go
# Importa a biblioteca Time para funcionalidades relacionadas ao tempo, como pausas
import time
# Importa a biblioteca Base64 para codificação e decodificação de dados
import base64
# Importa a biblioteca Joblib para salvar e carregar modelos de Machine Learning de forma eficiente
import joblib
# Importa a classe datetime do módulo datetime para trabalhar com datas e horas
from datetime import datetime
# Importa o módulo OS para interagir com o sistema operacional, como criar diretórios
import os
# Importa a biblioteca warnings para controlar avisos de depreciação ou outros
import warnings
# Ignora todos os avisos para manter a saída limpa
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
            try:
                from fpdf import FPDF

                # Inicializa um novo documento PDF
                pdf = FPDF()
                # Adiciona uma nova página ao documento
                pdf.add_page()

                # Define a fonte e adiciona o título principal do relatório
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "RELATORIO AUTOML PRO", ln=True, align='C')
                pdf.ln(5)

                # Define a fonte para informações menores e adiciona a data e hora de geração do relatório
                pdf.set_font("Arial", '', 10)
                pdf.cell(0, 10, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
                pdf.ln(10)

                # Adiciona um subtítulo para as informações do projeto
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "INFORMACOES DO PROJETO", ln=True)
                pdf.set_font("Arial", '', 10)

                # Se houver informações do dataset, adiciona-as ao PDF
                if data_info:
                    pdf.cell(0, 10, f"Dataset: {data_info.get('dataset_name', 'N/A')}", ln=True)
                    pdf.cell(0, 10, f"Amostras: {data_info.get('n_samples', 'N/A')}", ln=True)
                    pdf.cell(0, 10, f"Features: {data_info.get('n_features', 'N/A')}", ln=True)

                # Adiciona o tipo de problema e o número total de modelos treinados
                pdf.cell(0, 10, f"Tipo de problema: {problem_type.upper()}", ln=True)
                pdf.cell(0, 10, f"Total de modelos treinados: {len(results)}", ln=True)
                pdf.ln(10)

                # Adiciona um subtítulo para o melhor modelo
                # Obtém o nome do melhor modelo a partir do objeto 'trainer'
                best_name = trainer.best_model_name
                # Verifica se o nome do melhor modelo existe e se está presente nos resultados
                if best_name and best_name in results:
                    # Define a fonte para o subtítulo "MELHOR MODELO" (Arial, negrito, tamanho 12)
                    pdf.set_font("Arial", 'B', 12)
                    # Adiciona o subtítulo "MELHOR MODELO" ao PDF, centralizado e com quebra de linha
                    pdf.cell(0, 10, "MELHOR MODELO", ln=True)
                    # Define a fonte para as informações do modelo (Arial, normal, tamanho 10)
                    pdf.set_font("Arial", '', 10)


                    # Obtém as métricas do melhor modelo a partir do dicionário de resultados
                    best_metrics = results[best_name]
                    # Adiciona o nome do melhor modelo ao PDF
                    pdf.cell(0, 10, f"Modelo: {best_name}", ln=True)

                    # Verifica o tipo de problema (classificação ou regressão) para exibir a métrica principal correta
                    if problem_type == 'classification':
                        # Para classificação, tenta obter a acurácia ou um score genérico
                        score = best_metrics.get('accuracy', best_metrics.get('score', 0))
                        # Adiciona a acurácia formatada ao PDF
                        pdf.cell(0, 10, f"Acuracia: {score:.4f}", ln=True)
                    else:
                        # Para regressão, tenta obter o R2 Score ou um score genérico
                        score = best_metrics.get('r2', best_metrics.get('score', 0))
                        # Adiciona o R2 Score formatado ao PDF
                        pdf.cell(0, 10, f"R2 Score: {score:.4f}", ln=True)

                    # Adiciona uma quebra de linha com espaçamento ao PDF para separação visual
                    pdf.ln(10)

                # Define a fonte para o subtítulo "RANKING DOS MODELOS" (Arial, negrito, tamanho 12)
                pdf.set_font("Arial", 'B', 12)
                # Adiciona o subtítulo "RANKING DOS MODELOS" ao PDF, com quebra de linha
                pdf.cell(0, 10, "RANKING DOS MODELOS", ln=True)
                # Define a fonte para as informações do ranking (Arial, normal, tamanho 10)
                pdf.set_font("Arial", '', 10)

                # Obtém o DataFrame de ranking de modelos a partir do objeto 'trainer'
                ranking_df = trainer.get_ranking()

                # Define a cor de preenchimento para o cabeçalho da tabela (cinza claro)
                pdf.set_fill_color(240, 240, 240)
                # Adiciona a célula "Posicao" ao cabeçalho da tabela, com borda e preenchimento
                pdf.cell(30, 10, "Posicao", border=1, fill=True)
                # Adiciona a célula "Modelo" ao cabeçalho da tabela, com borda e preenchimento
                pdf.cell(80, 10, "Modelo", border=1, fill=True)
                # Adiciona a célula "Score" ao cabeçalho da tabela, com borda, preenchimento e quebra de linha
                pdf.cell(45, 10, "Score", border=1, fill=True, ln=True)

                # Itera sobre cada linha do DataFrame de ranking
                for _, row in ranking_df.iterrows():
                    # Converte a posição do ranking para string
                    pos = str(row['Posição'])
                    # Converte o nome do modelo para string e limita a 35 caracteres
                    model = str(row['Modelo'])
                    # Formata o score do modelo para 4 casas decimais e converte para string
                    score = f"{float(row['Score']):.4f}"

                    # Adiciona a célula da posição ao PDF, com borda
                    pdf.cell(30, 10, pos, border=1)
                    # Adiciona a célula do nome do modelo (truncado) ao PDF, com borda
                    pdf.cell(80, 10, model[:35], border=1)
                    # Adiciona a célula do score ao PDF, com borda e quebra de linha
                    pdf.cell(45, 10, score, border=1, ln=True)

                pdf.ln(10)  # Adiciona uma quebra de linha com espaçamento ao PDF para separação visual.

                if len(results) > 0:  # Verifica se há resultados de modelos para exibir.
                    pdf.set_font("Arial", 'B', 12)  # Define a fonte para o subtítulo "METRICAS DETALHADAS".
                    pdf.cell(0, 10, "METRICAS DETALHADAS", ln=True)  # Adiciona o subtítulo "METRICAS DETALHADAS" ao PDF, com quebra de linha.
                    pdf.set_font("Arial", '', 10)  # Define a fonte para as informações das métricas (Arial, normal, tamanho 10).

                    for model_name, metrics in results.items():  # Itera sobre cada modelo e suas métricas no dicionário 'results'.
                        pdf.set_font("Arial", 'B', 10)  # Define a fonte para o nome do modelo (Arial, negrito, tamanho 10).
                        pdf.cell(0, 10, f"Modelo: {model_name}", ln=True)  # Adiciona o nome do modelo ao PDF.
                        pdf.set_font("Arial", '', 9)  # Define a fonte para as métricas individuais (Arial, normal, tamanho 9).

                        for metric_name, value in metrics.items():  # Itera sobre cada métrica e seu valor para o modelo atual.
                            # Verifica se o valor da métrica é um tipo numérico (int, float, numpy float ou int).
                            if isinstance(value, (int, float, np.floating, np.integer)):
                                # Adiciona a métrica e seu valor formatado ao PDF, com quebra de linha.
                                pdf.cell(0, 8, f"  {metric_name}: {float(value):.4f}", ln=True)
                        pdf.ln(5)  # Adiciona uma quebra de linha com espaçamento após as métricas de cada modelo.

                    pdf.ln(10)  # Adiciona uma quebra de linha com espaçamento após a seção de métricas detalhadas.

                pdf.set_font("Arial", 'B', 12)  # Define a fonte para o subtítulo "RECOMENDACOES".
                pdf.cell(0, 10, "RECOMENDACOES", ln=True)  # Adiciona o subtítulo "RECOMENDACOES" ao PDF, com quebra de linha.
                pdf.set_font("Arial", '', 10)  # Define a fonte para as recomendações (Arial, normal, tamanho 10).

                recommendations = [
                    "1. Implemente o melhor modelo em producao",
                    "2. Monitore performance periodicamente",
                    "3. Re-treine com novos dados regularmente",
                    "4. Considere tecnicas de ensemble",
                    "5. Valide com testes A/B antes de deploy"
                ]

                for rec in recommendations:  # Itera sobre cada recomendação na lista 'recommendations'.
                    pdf.cell(0, 8, rec, ln=True)  # Adiciona a recomendação como uma célula de texto ao PDF, com quebra de linha.

                os.makedirs('reports', exist_ok=True)  # Cria o diretório 'reports' se ele não existir.
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Gera um timestamp para usar no nome do arquivo.
                filename = f'reports/relatorio_automl_{timestamp}.pdf'  # Define o nome do arquivo PDF.

                try:
                    pdf.output(filename)  # Salva o documento PDF no arquivo especificado.
                    if os.path.exists(filename):  # Verifica se o arquivo PDF foi criado com sucesso.
                        return filename  # Retorna o nome do arquivo se ele existir.
                    else:  # Se o arquivo não foi criado.
                        st.error("PDF não foi criado")  # Exibe uma mensagem de erro no Streamlit.
                        return None  # Retorna None indicando falha.
                except Exception as e:  # Captura exceções que ocorrem ao salvar o PDF.
                    st.error(f"Erro ao salvar PDF: {str(e)}")  # Exibe a mensagem de erro no Streamlit.
                    return PDFReportGenerator.generate_txt_report(results, trainer, problem_type, data_info)  # Tenta gerar um relatório TXT como fallback.

            except ImportError:  # Captura o erro se a biblioteca fpdf não estiver instalada.
                st.warning("fpdf2 não encontrado. Gerando relatório TXT...")  # Alerta que fpdf não foi encontrado e que um TXT será gerado.
                return PDFReportGenerator.generate_txt_report(results, trainer, problem_type, data_info)  # Gera um relatório TXT como fallback.
            except Exception as e:  # Captura outras exceções relacionadas ao uso do fpdf.
                st.error(f"Erro no fpdf: {str(e)}")  # Exibe a mensagem de erro no Streamlit.
                return PDFReportGenerator.generate_txt_report(results, trainer, problem_type, data_info)  # Gera um relatório TXT como fallback.

        except Exception as e:  # Captura exceções gerais que ocorrem na função generate_report.
            st.error(f"Erro ao gerar relatório: {str(e)}")  # Exibe a mensagem de erro no Streamlit.
            return PDFReportGenerator.generate_txt_report(results, trainer, problem_type, data_info)  # Gera um relatório TXT como fallback.

    @staticmethod
    def generate_txt_report(results, trainer, problem_type, data_info=None):
        """Gera relatório em texto (fallback)"""
        try:
            # Cria o diretório 'reports' se ele não existir, para salvar o relatório.
            os.makedirs('reports', exist_ok=True)

            # Gera um timestamp no formato 'YYYYMMDD_HHMMSS' para o nome do arquivo.
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Define o nome completo do arquivo TXT, incluindo o caminho do diretório e o timestamp.
            filename = f'reports/relatorio_automl_{timestamp}.txt'

            # Abre o arquivo TXT no modo de escrita ('w') com codificação UTF-8.
            with open(filename, 'w', encoding='utf-8') as f:
                # Escreve uma linha de separação no arquivo.
                f.write("=" * 60 + "\n")
                # Escreve o título principal do relatório no arquivo.
                f.write("RELATORIO AUTOML PRO - TODOS OS MODELOS\n")
                # Escreve outra linha de separação.
                f.write("=" * 60 + "\n\n")

                # Escreve a data e hora de geração do relatório.
                f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
                # Escreve o tipo de problema (classificação ou regressão).
                f.write(f"Tipo de problema: {problem_type.upper()}\n")
                # Escreve o número total de modelos treinados.
                f.write(f"Total de modelos: {len(results)}\n")

                # Verifica se há informações do dataset para incluir.
                if data_info:
                    # Escreve o número de amostras do dataset.
                    f.write(f"Amostras: {data_info.get('n_samples', 'N/A')}\n")
                    # Escreve o número de features do dataset.
                    f.write(f"Features: {data_info.get('n_features', 'N/A')}\n")

                # Escreve uma quebra de linha e uma linha de separação para a seção "MELHOR MODELO".
                f.write("\n" + "=" * 60 + "\n")
                # Escreve o título da seção "MELHOR MODELO".
                f.write("MELHOR MODELO\n")
                # Escreve outra linha de separação.
                f.write("=" * 60 + "\n\n")

                # Obtém o nome do melhor modelo do objeto 'trainer'.
                best_name = trainer.best_model_name
                # Verifica se um melhor modelo foi identificado e está nos resultados.
                if best_name and best_name in results:
                    # Escreve o nome do melhor modelo.
                    f.write(f"Modelo: {best_name}\n")
                    # Obtém as métricas do melhor modelo.
                    best_metrics = results[best_name]

                    # Itera sobre cada métrica e seu valor para o melhor modelo.
                    for metric, value in best_metrics.items():
                        # Verifica se o valor da métrica é numérico.
                        if isinstance(value, (int, float, np.floating, np.integer)):
                            # Escreve o nome da métrica e seu valor formatado com 4 casas decimais.
                            f.write(f"{metric}: {float(value):.4f}\n")

                # Adiciona uma quebra de linha e uma linha de separação para a seção "RANKING COMPLETO".
                f.write("\n" + "=" * 60 + "\n")
                # Escreve o título da seção "RANKING COMPLETO".
                f.write("RANKING COMPLETO\n")
                # Escreve outra linha de separação.
                f.write("=" * 60 + "\n\n")

                # Obtém o DataFrame de ranking de modelos do objeto 'trainer'.
                ranking_df = trainer.get_ranking()
                # Itera sobre cada linha do DataFrame de ranking.
                for _, row in ranking_df.iterrows():
                    # Escreve a posição, nome do modelo e score formatado no arquivo.
                    f.write(f"{row['Posição']}. {row['Modelo']} - Score: {float(row['Score']):.4f}\n")

                # Adiciona uma quebra de linha e uma linha de separação para a seção "METRICAS POR MODELO".
                f.write("\n" + "=" * 60 + "\n")
                # Escreve o título da seção "METRICAS POR MODELO".
                f.write("METRICAS POR MODELO\n")
                # Escreve outra linha de separação.
                f.write("=" * 60 + "\n\n")

                # Itera sobre cada modelo e suas métricas no dicionário 'results'.
                for model_name, metrics in results.items():
                    # Escreve o nome do modelo no arquivo.
                    f.write(f"{model_name}:\n")
                    # Itera sobre cada métrica e seu valor para o modelo atual.
                    for metric, value in metrics.items():
                        # Verifica se o valor da métrica é um tipo numérico.
                        if isinstance(value, (int, float, np.floating, np.integer)):
                            # Escreve a métrica e seu valor formatado com 4 casas decimais.
                            f.write(f"  {metric}: {float(value):.4f}\n")
                    # Adiciona uma quebra de linha após as métricas de cada modelo para separação.
                    f.write("\n")

                # Escreve uma linha de separação para a seção "RECOMENDACOES".
                f.write("=" * 60 + "\n")
                # Escreve o título da seção "RECOMENDACOES".
                f.write("RECOMENDACOES\n")
                # Escreve outra linha de separação.
                f.write("=" * 60 + "\n\n")

                # Lista de recomendações a serem incluídas no relatório.
                recs = [
                    "• Use o melhor modelo em producao",
                    "• Monitore performance",
                    "• Re-treine regularmente",
                    "• Valide com novos dados"
                ]

                for rec in recs:  # Itera sobre cada recomendação na lista 'recs'.
                    f.write(f"{rec}\n")  # Escreve a recomendação no arquivo TXT, seguida de uma quebra de linha.

            return filename  # Retorna o nome do arquivo TXT gerado.

        except Exception as e:  # Captura qualquer exceção que ocorra durante a geração do relatório TXT.
            st.error(f"Erro ao gerar TXT: {str(e)}")  # Exibe uma mensagem de erro no Streamlit com os detalhes da exceção.
            return None  # Retorna None indicando que o relatório TXT não pôde ser gerado.

# ========== PROCESSAMENTO DE DADOS ==========
class PowerfulDataProcessor:
    """Processador de dados avançado com feature engineering e detecção automática"""

    def __init__(self):
        # Inicializa o scaler para normalização de dados numéricos (e.g., StandardScaler)
        self.scaler = None
        # Dicionário para armazenar encoders para colunas categóricas, se necessário
        self.encoders = {}
        # Inicializa o imputer para tratamento de valores ausentes, se necessário
        self.imputer = None
        # Lista para armazenar os nomes ou índices das features selecionadas
        self.selected_features = []
        # Nome da coluna target, será definido durante o processamento
        self.target_col = None
        # Tipo de problema (classificação ou regressão), será detectado ou definido
        self.problem_type = None

    def process(self, data=None, target_col=None, X=None, y=None, problem_type=None, auto_detect=True):
        """
        Processamento

        Aceita múltiplos formatos:
        1. data + target_col
        2. data apenas
        3. X e y já separados
        """
        try:
            # Verifica se X e y já foram fornecidos diretamente
            if X is not None and y is not None:
                # Define o nome da coluna target como "target" para fins internos
                self.target_col = "target"
                # Detecta o tipo de problema (classificação ou regressão) se não for fornecido
                self.problem_type = problem_type or self.detect_problem_type_smart(y)

                # Realiza o pré-processamento poderoso nos dados X
                X_processed = self.powerful_preprocessing(X)
                # Aplica engenharia de features avançada aos dados processados
                X_engineered = self.advanced_feature_engineering(X_processed)
                # Executa a seleção inteligente de features
                X_final = self.smart_feature_selection(X_engineered, y, self.problem_type)

                # Se for um problema de classificação, processa a coluna target
                if self.problem_type == 'classification':
                    y_processed = self.process_target(y)
                # Caso contrário (regressão), processa a coluna target para regressão
                else:
                    y_processed = self.process_target_regression(y)

                # Retorna os dados X e y processados e o tipo de problema
                return X_final, y_processed, self.problem_type

            # Verifica se um DataFrame 'data' foi fornecido
            elif data is not None:
                # Se a coluna target não foi especificada e a detecção automática está ativada
                if target_col is None and auto_detect:
                    # Exibe uma mensagem informando que a detecção automática está em andamento
                    st.info("🔍 Detectando target automaticamente...")
                    # Chama o detector de target para identificar a coluna target e o tipo de problema
                    target_col, X, y, confidence, problem_type = TargetDetector.detect_target(data)

                    # Armazena o nome da coluna target detectada
                    self.target_col = target_col
                    # Armazena o tipo de problema detectado
                    self.problem_type = problem_type

                    # Realiza o pré-processamento poderoso nos dados X
                    X_processed = self.powerful_preprocessing(X)
                    # Aplica engenharia de features avançada aos dados processados
                    X_engineered = self.advanced_feature_engineering(X_processed)
                    # Executa a seleção inteligente de features
                    X_final = self.smart_feature_selection(X_engineered, y, problem_type)

                    # Se for um problema de classificação, processa a coluna target
                    if problem_type == 'classification':
                        y_processed = self.process_target(y)
                    # Caso contrário (regressão), processa a coluna target para regressão
                    else:
                        y_processed = self.process_target_regression(y)

                    # Retorna os dados X e y processados e o tipo de problema
                    return X_final, y_processed, problem_type

                # Se a coluna target foi especificada ou a detecção automática não foi usada
                else:
                    # Verifica se a coluna target especificada está no DataFrame
                    if target_col in data.columns:
                        # Separa X (features) removendo a coluna target
                        X = data.drop(columns=[target_col]).copy()
                        # Separa y (target) pegando a coluna target
                        y = data[target_col].copy()
                    # Se a coluna target não estiver explicitamente no DataFrame (fallback)
                    else:
                        # Assume que a última coluna é a target
                        X = data.iloc[:, :-1].copy()
                        # Assume que a última coluna é a target
                        y = data.iloc[:, -1].copy()

                    # Detecta o tipo de problema usando a função inteligente
                    problem_type = self.detect_problem_type_smart(y)
                    # Armazena o tipo de problema detectado
                    self.problem_type = problem_type
                    # Armazena o nome da coluna target
                    self.target_col = target_col

                    # Realiza o pré-processamento poderoso nos dados X
                    X_processed = self.powerful_preprocessing(X)
                    # Aplica engenharia de features avançada aos dados processados
                    X_engineered = self.advanced_feature_engineering(X_processed)
                    # Executa a seleção inteligente de features
                    X_final = self.smart_feature_selection(X_engineered, y, problem_type)

                    # Se for um problema de classificação, processa a coluna target
                    if problem_type == 'classification':
                        y_processed = self.process_target(y)
                    # Caso contrário (regressão), processa a coluna target para regressão
                    else:
                        y_processed = self.process_target_regression(y)

                    # Retorna os dados X e y processados e o tipo de problema
                    return X_final, y_processed, problem_type

            # Se nem 'data' nem 'X' e 'y' foram fornecidos, levanta um erro
            else:
                raise ValueError("❌ Dados insuficientes. Forneça 'data' ou 'X' e 'y'")

        # Captura qualquer exceção que ocorra durante o processamento
        except Exception as e:
            # Exibe uma mensagem de erro no Streamlit
            st.error(f"Erro no processamento: {str(e)}")
            # Retorna um fallback simples para evitar que o programa pare
            return self.simple_fallback(data if data is not None else X, target_col)

    def detect_problem_type_smart(self, y):
        """Detecção INTELIGENTE de tipo de problema"""
        try:
            # Tenta converter a série 'y' para numérica, substituindo valores não numéricos por NaN
            y_numeric = pd.to_numeric(y, errors='coerce')
            # Conta o número de valores não nulos na série numérica
            not_na = y_numeric.notna().sum()

            # Se a proporção de valores não nulos for menor que 80%, assume classificação (dados muito bagunçados)
            if not_na / len(y) < 0.8:
                return 'classification'

            # Remove os valores nulos da série numérica para análise
            y_clean = y_numeric.dropna()
            # Obtém o número de valores únicos na série limpa
            unique_vals = len(y_clean.unique())

            # Se o número de valores únicos for menor ou igual a 10
            if unique_vals <= 10:
                # Verifica se todos os valores podem ser convertidos para inteiros sem perda de informação
                if all(y_clean.astype(int) == y_clean):
                    # Se sim, e com poucos valores únicos, é provável que seja classificação
                    return 'classification'
                else:
                    # Se não, e mesmo com poucos valores únicos (float), pode ser regressão
                    return 'regression'
            # Se o número de valores únicos estiver entre 11 e 30
            elif unique_vals <= 30:
                # Calcula a contagem de frequência normalizada de cada valor
                value_counts = y_clean.value_counts(normalize=True)
                # Se algum valor único representa mais de 30% dos dados, sugere classificação
                if (value_counts > 0.3).any():
                    return 'classification'
                else:
                    # Caso contrário, sugere regressão
                    return 'regression'
            # Se o número de valores únicos for maior que 30, é provável que seja regressão
            else:
                return 'regression'

        # Captura qualquer exceção que ocorra durante o bloco try
        except Exception:
            try:
                # Tenta obter o número de valores únicos diretamente da série original 'y'
                unique_vals = len(y.unique())
                # Se o tipo de dado for 'object' (string) ou tiver poucos valores únicos (<= 10)
                if y.dtype == 'object' or unique_vals <= 10:
                    # Retorna classificação
                    return 'classification'
                else:
                    # Caso contrário, retorna regressão
                    return 'regression'
            # Captura qualquer exceção do segundo bloco try
            except Exception:
                # Como fallback final, retorna regressão se tudo falhar
                return 'regression'

    def powerful_preprocessing(self, X):
        """Pré-processamento avançado"""
        # Cria uma cópia do DataFrame X para evitar modificar o original
        X_clean = X.copy()

        # Itera sobre cada coluna no DataFrame copiado
        for col in X_clean.columns:
            # Verifica se a coluna possui valores ausentes (NaN)
            if X_clean[col].isna().any():
                # Se a coluna for de tipo numérico
                if pd.api.types.is_numeric_dtype(X_clean[col]):
                    # Calcula a assimetria (skewness) da coluna numérica
                    if X_clean[col].skew() > 1:
                        # Se for muito assimétrica (skew > 1), preenche NaN com a mediana
                        X_clean[col] = X_clean[col].fillna(X_clean[col].median())
                    else:
                        # Caso contrário, preenche NaN com a média
                        X_clean[col] = X_clean[col].fillna(X_clean[col].mean())
                else:
                    # Se a coluna não for numérica (categórica), calcula a moda
                    mode = X_clean[col].mode()
                    # Define o valor de preenchimento como a moda (se existir), ou "missing"
                    fill_value = mode.iloc[0] if len(mode) > 0 else "missing"
                    # Preenche os valores ausentes com o valor definido
                    X_clean[col] = X_clean[col].fillna(fill_value)

        # Seleciona as colunas numéricas do DataFrame
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        # Seleciona as colunas categóricas do DataFrame
        categorical_cols = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()

        # Se houver colunas numéricas
        if numeric_cols:
            # Cria uma cópia do subconjunto de colunas numéricas
            X_numeric = X_clean[numeric_cols].copy()

            # Itera sobre cada coluna numérica para tratamento de outliers (winsorization)
            for col in X_numeric.columns:
                # Calcula o primeiro quartil (Q1)
                Q1 = X_numeric[col].quantile(0.25)
                # Calcula o terceiro quartil (Q3)
                Q3 = X_numeric[col].quantile(0.75)
                # Calcula o Intervalo Interquartil (IQR)
                IQR = Q3 - Q1
                # Calcula o limite inferior para outliers
                lower_bound = Q1 - 1.5 * IQR
                # Calcula o limite superior para outliers
                upper_bound = Q3 + 1.5 * IQR
                # Limita os valores da coluna dentro dos limites inferior e superior (winsorization)
                X_numeric[col] = np.clip(X_numeric[col], lower_bound, upper_bound)

            # Importa o StandardScaler para normalização
            from sklearn.preprocessing import StandardScaler
            # Inicializa o StandardScaler
            self.scaler = StandardScaler()
            # Aplica a padronização (transformação z-score) aos dados numéricos
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)
            # Atribui os dados padronizados de volta às colunas numéricas no DataFrame principal
            X_clean[numeric_cols] = X_numeric_scaled

        # Se houver colunas categóricas
        if categorical_cols:
            # Itera sobre cada coluna categórica
            for col in categorical_cols:
                # Obtém o número de valores únicos na coluna (convertendo para string para robustez)
                unique_vals = len(X_clean[col].astype(str).unique())
                # Se o número de valores únicos for menor ou igual a 10 (baixa cardinalidade)
                if unique_vals <= 10:
                    # Aplica One-Hot Encoding (cria colunas dummy)
                    dummies = pd.get_dummies(X_clean[col], prefix=col, drop_first=True)
                    # Concatena as novas colunas dummy e remove a coluna categórica original
                    X_clean = pd.concat([X_clean.drop(columns=[col]), dummies], axis=1)
                else:
                    # Se tiver alta cardinalidade, aplica Frequency Encoding
                    # Calcula a frequência normalizada de cada valor
                    freq = X_clean[col].astype(str).value_counts(normalize=True)
                    # Substitui os valores da coluna pela sua frequência
                    X_clean[col] = X_clean[col].astype(str).map(freq)

        # Retorna o DataFrame processado
        return X_clean

    def advanced_feature_engineering(self, X):
        """Feature engineering avançado"""
        # Cria uma cópia do DataFrame X para evitar modificar o original
        X_engineered = X.copy()

        # Seleciona as colunas numéricas do DataFrame
        numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()

        # Verifica se há pelo menos duas colunas numéricas para criar interações
        if len(numeric_cols) >= 2:
            # Itera sobre as primeiras 3 colunas numéricas para criar features de interação
            for i in range(min(3, len(numeric_cols))):
                # Itera sobre as próximas 3 colunas numéricas (a partir de 'i+1')
                for j in range(i + 1, min(i + 3, len(numeric_cols))):
                    # Obtém o nome da primeira coluna
                    col1 = numeric_cols[i]
                    # Obtém o nome da segunda coluna
                    col2 = numeric_cols[j]
                    # Adiciona uma nova feature que é o produto das duas colunas
                    X_engineered[f'{col1}_x_{col2}'] = X_engineered[col1] * X_engineered[col2]
                    # Adiciona uma nova feature que é a divisão da primeira pela segunda (com pequena constante para evitar divisão por zero)
                    X_engineered[f'{col1}_div_{col2}'] = X_engineered[col1] / (X_engineered[col2] + 1e-10)

        # Verifica se há colunas numéricas para criar features estatísticas
        if len(numeric_cols) > 0:
            # Adiciona uma feature com a média de todas as colunas numéricas para cada linha
            X_engineered['mean_features'] = X_engineered[numeric_cols].mean(axis=1)
            # Adiciona uma feature com o desvio padrão de todas as colunas numéricas para cada linha
            X_engineered['std_features'] = X_engineered[numeric_cols].std(axis=1)
            # Adiciona uma feature com o valor máximo de todas as colunas numéricas para cada linha
            X_engineered['max_features'] = X_engineered[numeric_cols].max(axis=1)
            # Adiciona uma feature com o valor mínimo de todas as colunas numéricas para cada linha
            X_engineered['min_features'] = X_engineered[numeric_cols].min(axis=1)

        # Verifica se há colunas numéricas para criar features polinomiais/transformadas
        if len(numeric_cols) > 0:
            # Itera sobre as primeiras 3 colunas numéricas
            for col in numeric_cols[:3]:
                # Adiciona uma nova feature que é o quadrado da coluna
                X_engineered[f'{col}_squared'] = X_engineered[col] ** 2
                # Adiciona uma nova feature que é a raiz quadrada do valor absoluto da coluna (com pequena constante para evitar raiz de zero)
                X_engineered[f'{col}_sqrt'] = np.sqrt(np.abs(X_engineered[col]) + 1e-10)

        # Retorna o DataFrame com as novas features engenheiradas
        return X_engineered

    def smart_feature_selection(self, X, y, problem_type):
        """Seleção inteligente de features"""
        try:
            # Se o número de features for menor ou igual a 20, não realiza seleção e retorna o X original
            if X.shape[1] <= 20:
                return X

            # Importa a classe VarianceThreshold para remover features com baixa variância
            from sklearn.feature_selection import VarianceThreshold
            # Inicializa o seletor com um limite de variância de 0.01 (remove features com variância muito baixa)
            selector = VarianceThreshold(threshold=0.01)
            # Aplica a transformação para remover as features de baixa variância
            X_selected = selector.fit_transform(X)

            # Verifica se o número de features ainda é alto após a filtragem por variância
            if X_selected.shape[1] > 50:
                # Se for um problema de classificação, usa RandomForestClassifier
                if problem_type == 'classification':
                    from sklearn.ensemble import RandomForestClassifier
                    # Inicializa o modelo RandomForestClassifier com 50 estimadores e random_state fixo
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                # Se for um problema de regressão, usa RandomForestRegressor
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    # Inicializa o modelo RandomForestRegressor com 50 estimadores e random_state fixo
                    model = RandomForestRegressor(n_estimators=50, random_state=42)

                # Treina o modelo nas features selecionadas e no target
                model.fit(X_selected, y)
                # Obtém a importância de cada feature do modelo treinado
                importances = model.feature_importances_

                # Obtém os índices das 30 features mais importantes
                top_indices = np.argsort(importances)[-30:]
                # Seleciona apenas as 30 features mais importantes do dataset
                X_final = X_selected[:, top_indices]
                # Armazena os índices das features selecionadas (para uso futuro, se necessário)
                self.selected_features = top_indices
            # Se o número de features após a filtragem por variância for <= 50, usa todas elas
            else:
                X_final = X_selected

            # Retorna o DataFrame com as features selecionadas
            return X_final

        # Captura qualquer exceção que ocorra durante o processo de seleção de features
        except Exception as e:
            # Exibe uma mensagem de aviso no Streamlit se a seleção de features falhar
            st.write(f"⚠️ Feature selection falhou: {str(e)[:50]}")
            # Em caso de falha, retorna o DataFrame X original (sem seleção)
            return X

    def process_target(self, y):
        """Processar target para classificação"""
        # Verifica se o tipo de dado da série 'y' é 'object' (geralmente strings ou misto)
        if y.dtype == 'object':
            # Se for 'object', usa pd.factorize para converter categorias em números inteiros
            y_encoded, _ = pd.factorize(y)
            # Retorna a série codificada como um Pandas Series
            return pd.Series(y_encoded)

        # Se o número de valores únicos na série 'y' for menor ou igual a 10 (indicando classes discretas)
        if len(pd.Series(y).unique()) <= 10:
            # Converte a série para o tipo inteiro e retorna como um Pandas Series
            return pd.Series(y).astype(int)

        # Caso contrário, retorna a série 'y' original como um Pandas Series (já numérico e com muitos valores únicos)
        return pd.Series(y)

    def process_target_regression(self, y):
        """Processar target para regressão"""
        try:
            # Tenta converter a série 'y' para numérica, transformando erros em NaN
            y_numeric = pd.to_numeric(y, errors='coerce')

            # Se o número de amostras for maior que 100
            if len(y_numeric) > 100:
                # Calcula o primeiro quartil (Q1)
                Q1 = y_numeric.quantile(0.25)
                # Calcula o terceiro quartil (Q3)
                Q3 = y_numeric.quantile(0.75)
                # Calcula o Intervalo Interquartil (IQR)
                IQR = Q3 - Q1
                # Define o limite inferior para detecção de outliers (3 * IQR para um critério mais flexível)
                lower_bound = Q1 - 3 * IQR
                # Define o limite superior para detecção de outliers
                upper_bound = Q3 + 3 * IQR
                # Aplica winsorization, limitando os valores dentro dos limites calculados
                y_numeric = np.clip(y_numeric, lower_bound, upper_bound)

            # Retorna a série numérica processada, preenchendo quaisquer NaNs remanescentes com a mediana
            return pd.Series(y_numeric).fillna(pd.Series(y_numeric).median())
        except Exception:
            # Em caso de erro, retorna a série 'y' original como um Pandas Series
            return pd.Series(y)

    def simple_fallback(self, data, target_col):
        """Fallback simples"""
        try:
            # Se nenhum dado foi fornecido (data é None)
            if data is None:
                # Define um número padrão de amostras
                n_samples = 100
                # Cria um DataFrame X com duas features numéricas aleatórias
                X = pd.DataFrame({
                    'feature_1': np.random.randn(n_samples),
                    'feature_2': np.random.randn(n_samples),
                })
                # Cria uma série y para classificação binária aleatória
                y = pd.Series(np.random.randint(0, 2, n_samples))
                # Retorna os dados gerados e o tipo de problema 'classification'
                return X, y, 'classification'

            # Verifica se a coluna target especificada está no DataFrame
            if target_col in data.columns:
                # Separa X (features) removendo a coluna target
                X = data.drop(columns=[target_col]).copy()
                # Separa y (target) pegando a coluna target
                y = data[target_col].copy()
            else:
                # Se a coluna target não estiver explicitamente no DataFrame, assume que a última coluna é a target
                X = data.iloc[:, :-1].copy()
                # Assume que a última coluna é a target
                y = data.iloc[:, -1].copy()

            # Cria uma cópia de X para pré-processamento numérico
            X_num = X.copy()
            # Itera sobre cada coluna no DataFrame X_num
            for col in X_num.columns:
                try:
                    # Tenta converter a coluna para numérica, transformando erros em NaN
                    X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
                except Exception:
                    # Se não for numérica, usa pd.factorize para codificar categorias em números
                    X_num[col] = pd.factorize(X_num[col])[0]

            # Preenche quaisquer valores ausentes (NaN) em X_num com 0
            X_num = X_num.fillna(0)

            try:
                # Obtém o número de valores únicos na série y
                unique_y = len(pd.Series(y).unique())
                # Se o tipo de dado de y for 'object' ou tiver poucos valores únicos (<= 10), assume classificação
                if getattr(y, 'dtype', None) == 'object' or unique_y <= 10:
                    problem_type = 'classification'
                else:
                    # Caso contrário, assume regressão
                    problem_type = 'regression'
            except Exception:
                # Em caso de erro na detecção do tipo de problema, assume regressão como fallback
                problem_type = 'regression'

            # Retorna os dados X e y processados de forma simples e o tipo de problema
            return X_num, y, problem_type
        except Exception:
            # Em caso de qualquer erro crítico no fallback simples, gera dados aleatórios como último recurso
            n_samples = 100
            # Cria um DataFrame X com duas features numéricas aleatórias
            X = pd.DataFrame({
                'feature_1': np.random.randn(n_samples),
                'feature_2': np.random.randn(n_samples),
            })
            # Cria uma série y para classificação binária aleatória
            y = pd.Series(np.random.randint(0, 2, n_samples))
            # Retorna os dados gerados e o tipo de problema 'classification'
            return X, y, 'classification'

# ========== TREINAMENTO COM VALIDAÇÃO CRUZADA ==========
class UltraCompleteTrainer:
    # Inicializa a classe UltraCompleteTrainer.
    def __init__(self, problem_type):
        # Armazena o tipo de problema (classificação ou regressão).
        self.problem_type = problem_type
        # Dicionário para armazenar os objetos dos modelos treinados.
        self.models = {}
        # Dicionário para armazenar as métricas de desempenho de cada modelo.
        self.results = {}
        # Dicionário para armazenar os scores detalhados de cada fold da validação cruzada.
        self.cv_scores = {}
        # Variável para armazenar o objeto do melhor modelo treinado.
        self.best_model = None
        # Variável para armazenar o nome do melhor modelo.
        self.best_model_name = ""
        # Flag para indicar se a validação cruzada será utilizada (sempre True para esta classe).
        self.use_cross_validation = True
        # Número de folds a ser usado na validação cruzada, padrão é 5.
        self.n_folds = 5

    # Método principal para iniciar o treinamento de forma segura com validação cruzada.
    def train_safe(self, X, y):
        """Treinamento com VALIDAÇÃO CRUZADA AUTOMÁTICA"""
        # Exibe uma mensagem informativa no Streamlit.
        st.info("🔬 Iniciando treinamento com VALIDAÇÃO CRUZADA...")

        try:
            # Verifica se o dataset é muito pequeno para validação cruzada robusta.
            if len(X) < 20:
                # Exibe um aviso se o dataset for pequeno.
                st.warning("⚠️ Dataset pequeno. Usando validação simples.")
                # Chama um método de fallback para treinamento simples.
                return self.train_simple_fallback(X, y)

            # Obtém todos os modelos disponíveis para o tipo de problema.
            models = self.get_all_models()

            # Contador para o número de modelos treinados.
            trained_count = 0
            # Total de modelos a serem treinados.
            total_models = len(models)

            # Cria uma barra de progresso no Streamlit.
            progress_bar = st.progress(0)

            # Itera sobre cada modelo no dicionário de modelos.
            for name, model in models.items():
                try:
                    # Exibe um spinner enquanto o modelo está sendo treinado.
                    with st.spinner(f"🔄 {name} (CV {self.n_folds}-fold)..."):
                        # Treina o modelo usando validação cruzada e obtém métricas e scores por fold.
                        cv_metrics, cv_scores = self.train_with_cross_validation(model, X, y)

                        # Armazena o objeto do modelo.
                        self.models[name] = model
                        # Armazena as métricas do modelo.
                        self.results[name] = cv_metrics
                        # Armazena os scores por fold da validação cruzada.
                        self.cv_scores[name] = cv_scores
                        # Incrementa o contador de modelos treinados.
                        trained_count += 1

                        # Calcula o progresso e atualiza a barra.
                        progress = trained_count / total_models
                        progress_bar.progress(progress)

                        # Determina a métrica principal para exibir (acurácia para classificação, R2 para regressão).
                        if self.problem_type == 'classification':
                            score = cv_metrics.get('accuracy', 0)
                        else:
                            score = cv_metrics.get('r2', 0)

                        # Exibe o score médio e o desvio padrão do modelo.
                        st.write(f"✅ **{name}**: {score:.4f} ± {cv_metrics.get('std', 0.0):.4f}")

                # Captura exceções que ocorrem durante o treinamento de um modelo específico.
                except Exception as e:
                    # Exibe um aviso se o treinamento de um modelo falhar.
                    st.write(f"⚠️ {name}: {str(e)[:50]}...")
                    # Continua para o próximo modelo.
                    continue

            # Verifica se há resultados após o treinamento.
            if self.results:
                # Determina o melhor modelo com base nas métricas.
                self.determine_best_model_complete()
                # Exibe uma mensagem de sucesso com o número de modelos treinados.
                st.success(f"✅ {trained_count} modelos treinados com VALIDAÇÃO CRUZADA!")

                # Se um melhor modelo foi identificado.
                if self.best_model_name:
                    # Treina o melhor modelo novamente com todos os dados.
                    self.train_final_model(X, y)
                    # Mostra os resultados detalhados da validação cruzada para o melhor modelo.
                    self.show_cv_results()
                    # Exibe o nome do melhor modelo.
                    st.success(f"🏆 **MELHOR MODELO**: {self.best_model_name}")

            # Retorna os resultados de todos os modelos e o nome do melhor modelo.
            return self.results, self.best_model_name

        # Captura exceções gerais que ocorrem no método train_safe.
        except Exception as e:
            # Exibe uma mensagem de erro.
            st.error(f"❌ Erro no treinamento: {str(e)}")
            # Em caso de erro, retorna um treinamento de fallback simples.
            return self.train_simple_fallback(X, y)

    # Método para treinar um modelo usando validação cruzada.
    def train_with_cross_validation(self, model, X, y):
        """Treina com validação cruzada e retorna métricas"""
        # Importa as funções necessárias para validação cruzada.
        from sklearn.model_selection import cross_validate, StratifiedKFold, KFold

        # Determina a estratégia de validação cruzada: Stratified K-Fold para classificação com mais de uma classe, senão K-Fold.
        if self.problem_type == 'classification' and len(np.unique(y)) > 1:
            # Cria um objeto StratifiedKFold para manter a proporção das classes em cada fold.
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            # Define o tipo de CV para registro.
            cv_type = "Stratified K-Fold"
        else:
            # Cria um objeto KFold para divisão simples.
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            # Define o tipo de CV para registro.
            cv_type = "K-Fold"

        # Define as métricas de scoring para classificação.
        if self.problem_type == 'classification':
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision_weighted',
                'recall': 'recall_weighted',
                'f1': 'f1_weighted'
            }
        # Define as métricas de scoring para regressão.
        else:
            scoring = {
                'r2': 'r2',
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'neg_mean_absolute_error': 'neg_mean_absolute_error'
            }

        try:
            # Executa a validação cruzada usando cross_validate.
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1, # Usa todos os núcleos da CPU para paralelização.
                verbose=0
            )

            # Dicionário para armazenar as métricas médias.
            metrics = {}
            # Dicionário para armazenar os scores de cada fold.
            scores_dict = {}

            # Itera sobre as chaves de scoring para calcular a média e o desvio padrão.
            for metric_name in scoring.keys():  # Itera sobre cada nome de métrica definido no dicionário 'scoring'.
                score_key = f'test_{metric_name}'  # Constrói a chave esperada para os resultados de teste (e.g., 'test_accuracy').
                if score_key in cv_results:  # Verifica se essa chave de score existe nos resultados da validação cruzada.
                    scores = cv_results[score_key]  # Obtém a lista de scores para a métrica atual através de todos os folds.
                    metrics[metric_name] = float(np.mean(scores))  # Calcula a média dos scores e armazena no dicionário 'metrics'.
                    metrics[f'{metric_name}_std'] = float(np.std(scores))  # Calcula o desvio padrão dos scores e armazena no dicionário 'metrics'.
                    scores_dict[metric_name] = scores.tolist()  # Converte a lista de scores para uma lista Python e armazena em 'scores_dict'.

            if self.problem_type == 'regression' and 'neg_mean_squared_error' in metrics:  # Verifica se o problema é regressão e se a métrica de erro quadrático médio negativo está presente.
                # Calcula o RMSE (Root Mean Squared Error) a partir do MSE negativo, garantindo que o valor seja não negativo antes da raiz.
                metrics['rmse'] = float(np.sqrt(max(0, -metrics['neg_mean_squared_error'])))
                # Armazena o desvio padrão do RMSE, usando o desvio padrão do neg_mean_squared_error como aproximação.
                metrics['rmse_std'] = float(metrics.get('neg_mean_squared_error_std', 0.0))

            # Verifica se o problema é de regressão e se a métrica 'neg_mean_absolute_error' está disponível.
            if self.problem_type == 'regression' and 'neg_mean_absolute_error' in metrics:
                # Calcula o MAE (Mean Absolute Error) invertendo o sinal da métrica neg_mean_absolute_error (que é negativa).
                metrics['mae'] = float(-metrics['neg_mean_absolute_error'])
                # Armazena o desvio padrão do MAE, obtendo-o do dicionário 'metrics' ou usando 0.0 como fallback.
                metrics['mae_std'] = float(metrics.get('neg_mean_absolute_error_std', 0.0))


            # Adiciona os tempos de treinamento e pontuação.
            metrics['fit_time'] = float(np.mean(cv_results['fit_time']))
            metrics['score_time'] = float(np.mean(cv_results['score_time']))
            # Adiciona o tipo de CV usado.
            metrics['cv_type'] = cv_type
            # Adiciona o número de folds.
            metrics['n_folds'] = self.n_folds

            # Adiciona o desvio padrão da métrica principal para exibição simplificada.
            if self.problem_type == 'classification':
                metrics['std'] = metrics.get('accuracy_std', 0.0)
            else:
                metrics['std'] = metrics.get('r2_std', 0.0)

            # Retorna as métricas e os scores por fold.
            return metrics, scores_dict

        # Captura exceções que ocorrem durante a validação cruzada.
        except Exception as e:
            # Exibe um aviso se o CV falhar para o modelo.
            st.write(f"⚠️ CV falhou para este modelo: {str(e)[:50]}")
            # Retorna métricas de um treinamento simples como fallback e um dicionário vazio de scores.
            return self.train_simple_model(model, X, y), {}

    # Método para realizar um treinamento simples (sem validação cruzada).
    def train_simple_model(self, model, X, y):
        """Fallback: treino simples sem CV"""
        # Importa a função para dividir os dados em treino e teste.
        from sklearn.model_selection import train_test_split

        # Divide os dados em conjuntos de treino e teste.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            # Usa estratificação para classificação para manter as proporções de classe.
            stratify=y if self.problem_type == 'classification' else None
        )

        # Treina o modelo com os dados de treino.
        model.fit(X_train, y_train)
        # Faz previsões nos dados de teste.
        y_pred = model.predict(X_test)

        # Calcula e retorna as métricas completas com base nas previsões.
        return self.calculate_complete_metrics(y_test, y_pred)

    # Método para treinar o melhor modelo identificado com todos os dados disponíveis.
    def train_final_model(self, X, y):
        """Treina o melhor modelo com todos os dados"""
        # Verifica se um melhor modelo foi identificado e existe no dicionário de modelos.
        if self.best_model_name and self.best_model_name in self.models:
            # Importa a função clone para criar uma cópia "limpa" do modelo.
            from sklearn.base import clone
            # Clona o melhor modelo para treiná-lo com todos os dados.
            final_model = clone(self.models[self.best_model_name])
            # Treina o modelo final com todo o conjunto de dados.
            final_model.fit(X, y)
            # Armazena o modelo final treinado.
            self.best_model = final_model

    # Método para exibir os resultados da validação cruzada para o melhor modelo.
    def show_cv_results(self):
        """Mostra resultados da validação cruzada"""
        # Verifica se o nome do melhor modelo está definido e se existem scores de CV para ele.
        if self.best_model_name and self.best_model_name in self.cv_scores:
            # Obtém os scores de CV do melhor modelo.
            cv_scores = self.cv_scores[self.best_model_name]

            # Cria um expansor no Streamlit para mostrar os resultados detalhados.
            with st.expander(f"📊 Resultados CV - {self.best_model_name}"):
                # Itera sobre cada métrica e seus scores por fold.
                for metric, scores in cv_scores.items():
                    # Verifica se há scores para a métrica.
                    if len(scores) > 0:
                        # Exibe o título da métrica.
                        st.write(f"**{metric} por fold:**")
                        # Exibe o score para cada fold.
                        for i, score in enumerate(scores):
                            st.write(f"  Fold {i + 1}: {score:.4f}")
                        # Exibe a média e o desvio padrão dos scores.
                        st.write(f"  **Média:** {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                        # Adiciona uma linha de separação.
                        st.write("---")

    # Método de fallback completo para treinamento sem validação cruzada.
    def train_simple_fallback(self, X, y):
        """Fallback completo sem CV"""
        # Exibe uma mensagem informativa.
        st.info("Usando treinamento simples (sem CV)...")

        # Importa a função para dividir os dados em treino e teste.
        from sklearn.model_selection import train_test_split

        # Divide os dados em conjuntos de treino e teste sem estratificação.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Escolhe um modelo Random Forest com base no tipo de problema.
        if self.problem_type == 'classification':
            # Importa RandomForestClassifier.
            from sklearn.ensemble import RandomForestClassifier
            # Inicializa o modelo de classificação.
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            # Importa RandomForestRegressor.
            from sklearn.ensemble import RandomForestRegressor
            # Inicializa o modelo de regressão.
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Treina o modelo com os dados de treino.
        model.fit(X_train, y_train)
        # Faz previsões nos dados de teste.
        y_pred = model.predict(X_test)
        # Calcula as métricas completas.
        metrics = self.calculate_complete_metrics(y_test, y_pred)

        # Define o nome do modelo de fallback.
        model_name = "Random Forest"
        # Armazena o modelo de fallback.
        self.models[model_name] = model
        # Armazena as métricas do modelo de fallback.
        self.results[model_name] = metrics
        # Define o modelo de fallback como o melhor modelo.
        self.best_model_name = model_name
        # Armazena o objeto do melhor modelo.
        self.best_model = model

        # Retorna os resultados e o nome do melhor modelo.
        return self.results, self.best_model_name

    # Método para obter todos os modelos disponíveis com base no tipo de problema.
    def get_all_models(self):
        """Retorna TODOS os modelos disponíveis"""
        # Retorna modelos de classificação se o problema for 'classification'.
        if self.problem_type == 'classification':
            return self.get_all_classification_models()
        # Retorna modelos de regressão se o problema for outro (regressão).
        else:
            return self.get_all_regression_models()

    # Método para obter todos os modelos de classificação disponíveis.
    def get_all_classification_models(self):
        """Retorna TODOS os modelos de classificação"""
        # Dicionário para armazenar os modelos de classificação.
        models = {}

        try:
            # Importa modelos de ensemble para classificação.
            from sklearn.ensemble import (
                RandomForestClassifier, GradientBoostingClassifier,
                AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
            )

            # Adiciona Random Forest Classifier.
            models['Random Forest'] = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
            # Adiciona Gradient Boosting Classifier.
            models['Gradient Boosting'] = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42
            )
            # Adiciona AdaBoost Classifier.
            models['AdaBoost'] = AdaBoostClassifier(
                n_estimators=100, random_state=42
            )
            # Adiciona Extra Trees Classifier.
            models['Extra Trees'] = ExtraTreesClassifier(
                n_estimators=100, random_state=42
            )
            # Adiciona Bagging Classifier.
            models['Bagging'] = BaggingClassifier(
                n_estimators=50, random_state=42
            )

            # Importa modelos lineares para classificação.
            from sklearn.linear_model import (
                LogisticRegression, RidgeClassifier, SGDClassifier
            )

            # Adiciona Logistic Regression.
            models['Logistic Regression'] = LogisticRegression(
                max_iter=1000, random_state=42, C=1.0
            )
            # Adiciona Ridge Classifier.
            models['Ridge Classifier'] = RidgeClassifier(
                alpha=1.0, random_state=42
            )
            # Adiciona SGD Classifier.
            models['SGD Classifier'] = SGDClassifier(
                max_iter=1000, random_state=42
            )

            # Importa modelos SVM e KNN para classificação.
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier

            # Adiciona SVM com kernel RBF.
            models['SVM RBF'] = SVC(
                kernel='rbf', probability=True, random_state=42
            )
            # Adiciona K-Nearest Neighbors Classifier.
            models['KNN'] = KNeighborsClassifier(
                n_neighbors=5
            )

            # Importa Decision Tree e Naive Bayes para classificação.
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.naive_bayes import GaussianNB

            # Adiciona Decision Tree Classifier.
            models['Decision Tree'] = DecisionTreeClassifier(
                max_depth=10, random_state=42
            )
            # Adiciona Gaussian Naive Bayes.
            models['Gaussian NB'] = GaussianNB()

            try:
                # Tenta importar e adicionar XGBoost Classifier.
                from xgboost import XGBClassifier
                # Adiciona XGBoost Classifier com parâmetros específicos para evitar avisos.
                models['XGBoost'] = XGBClassifier(
                    n_estimators=100, random_state=42, use_label_encoder=False,
                    eval_metric='logloss'
                )
            except Exception:
                # Ignora se XGBoost não estiver disponível.
                pass

            try:
                # Tenta importar e adicionar LightGBM Classifier.
                from lightgbm import LGBMClassifier
                models['LightGBM'] = LGBMClassifier(
                    n_estimators=100, random_state=42
                )
            except Exception:
                # Ignora se LightGBM não estiver disponível.
                pass

            # Importa Linear Discriminant Analysis e MLP Classifier.
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.neural_network import MLPClassifier

            # Adiciona Linear Discriminant Analysis.
            models['LDA'] = LinearDiscriminantAnalysis()
            # Adiciona Multi-layer Perceptron Classifier.
            models['MLP'] = MLPClassifier(
                hidden_layer_sizes=(100,), max_iter=1000, random_state=42
            )

        # Captura exceções que ocorrem ao carregar modelos.
        except Exception as e:
            # Exibe um aviso se alguns modelos não puderem ser carregados.
            st.write(f"⚠️ Erro ao carregar alguns modelos: {str(e)[:50]}")

        # Retorna o dicionário de modelos de classificação.
        return models

    # Método para obter todos os modelos de regressão disponíveis.
    def get_all_regression_models(self):
        """Retorna TODOS os modelos de regressão"""
        # Dicionário para armazenar os modelos de regressão.
        models = {}

        try:
            # Importa modelos de ensemble para regressão.
            from sklearn.ensemble import (
                RandomForestRegressor, GradientBoostingRegressor,
                AdaBoostRegressor, ExtraTreesRegressor
            )

            # Adiciona Random Forest Regressor.
            models['Random Forest'] = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            # Adiciona Gradient Boosting Regressor.
            models['Gradient Boosting'] = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, random_state=42
            )
            # Adiciona AdaBoost Regressor.
            models['AdaBoost'] = AdaBoostRegressor(
                n_estimators=100, random_state=42
            )
            # Adiciona Extra Trees Regressor.
            models['Extra Trees'] = ExtraTreesRegressor(
                n_estimators=100, random_state=42
            )

            # Importa modelos lineares para regressão.
            from sklearn.linear_model import (
                LinearRegression, Ridge, Lasso, ElasticNet,
                BayesianRidge
            )

            # Adiciona Linear Regression.
            models['Linear Regression'] = LinearRegression()
            # Adiciona Ridge Regressor.
            models['Ridge'] = Ridge(alpha=1.0, random_state=42)
            # Adiciona Lasso Regressor.
            models['Lasso'] = Lasso(alpha=0.1, random_state=42)
            # Adiciona ElasticNet Regressor.
            models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            # Adiciona Bayesian Ridge Regressor.
            models['Bayesian Ridge'] = BayesianRidge()

            # Importa modelos SVM e KNN para regressão.
            from sklearn.svm import SVR
            from sklearn.neighbors import KNeighborsRegressor

            # Adiciona Support Vector Regressor com kernel RBF.
            models['SVR RBF'] = SVR(kernel='rbf')
            # Adiciona K-Nearest Neighbors Regressor.
            models['KNN Regressor'] = KNeighborsRegressor(n_neighbors=5)

            # Importa Decision Tree Regressor.
            from sklearn.tree import DecisionTreeRegressor

            # Adiciona Decision Tree Regressor.
            models['Decision Tree'] = DecisionTreeRegressor(
                max_depth=10, random_state=42
            )

            try:
                # Tenta importar e adicionar XGBoost Regressor.
                from xgboost import XGBRegressor
                models['XGBoost'] = XGBRegressor(
                    n_estimators=100, random_state=42
                )
            except Exception:
                # Ignora se XGBoost não estiver disponível.
                pass

            try:
                # Tenta importar e adicionar LightGBM Regressor.
                from lightgbm import LGBMRegressor
                models['LightGBM'] = LGBMRegressor(
                    n_estimators=100, random_state=42
                )
            except Exception:
                # Ignora se LightGBM não estiver disponível.
                pass

            # Importa MLP Regressor.
            from sklearn.neural_network import MLPRegressor

            # Adiciona Multi-layer Perceptron Regressor.
            models['MLP Regressor'] = MLPRegressor(
                hidden_layer_sizes=(100,), max_iter=1000, random_state=42
            )

        # Captura exceções que ocorrem ao carregar modelos.
        except Exception as e:
            # Exibe um aviso se alguns modelos não puderem ser carregados.
            st.write(f"⚠️ Erro ao carregar alguns modelos: {str(e)[:50]}")

        # Retorna o dicionário de modelos de regressão.
        return models

    # Método para calcular um conjunto completo de métricas.
    def calculate_complete_metrics(self, y_true, y_pred):
        """Cálculo COMPLETO de métricas"""
        try:
            # Cálculos de métricas para classificação.
            if self.problem_type == 'classification':

                from sklearn.metrics import ( # Importa o módulo metrics da biblioteca sklearn para cálculo de métricas de avaliação.
                    accuracy_score, precision_score, recall_score, # Importa as funções específicas para acurácia, precisão e recall.
                    f1_score # Importa a função F1-score para classificação.
                )

                metrics = { # Inicializa um dicionário para armazenar as métricas calculadas.
                    'accuracy': float(accuracy_score(y_true, y_pred)), # Calcula a acurácia (precisão geral) e armazena como float.
                    'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), # Calcula a precisão ponderada para classificação multiclasse e armazena como float.
                    'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)), # Calcula o recall ponderado para classificação multiclasse e armazena como float.
                    'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)) # Calcula o F1-score ponderado para classificação multiclasse e armazena como float.
                }

                # Retorna as métricas de classificação.
                return metrics

            # Cálculos de métricas para regressão.
            else:
                # Importa as métricas de regressão necessárias.
                from sklearn.metrics import (
                    r2_score, mean_squared_error, mean_absolute_error,
                    explained_variance_score
                )

                # Dicionário para armazenar as métricas de regressão.
                metrics = {
                    'r2': float(r2_score(y_true, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    'mae': float(mean_absolute_error(y_true, y_pred)),
                    'explained_variance': float(explained_variance_score(y_true, y_pred))
                }

                # Retorna as métricas de regressão.
                return metrics

        # Captura exceções que ocorrem durante o cálculo de métricas.
        except Exception as e:
            # Exibe um aviso se houver um erro no cálculo das métricas.
            st.write(f"⚠️ Erro em métricas: {str(e)[:50]}")
            # Em caso de erro, retorna uma métrica básica para classificação.
            if self.problem_type == 'classification':
                from sklearn.metrics import accuracy_score
                return {'accuracy': float(accuracy_score(y_true, y_pred))}
            # Em caso de erro, retorna uma métrica básica para regressão.
            else:
                from sklearn.metrics import r2_score
                return {'r2': float(r2_score(y_true, y_pred))}

    # Método para determinar o melhor modelo considerando múltiplas métricas.
    def determine_best_model_complete(self):
        """Determina melhor modelo considerando múltiplas métricas"""
        # Se não houver resultados, retorna.
        if not self.results:
            return

        # Define pesos das métricas e a métrica principal para classificação.
        if self.problem_type == 'classification':
            metric_weights = {'accuracy': 0.4, 'f1': 0.3, 'precision': 0.2, 'recall': 0.1}
            main_metric = 'accuracy'
        # Define pesos das métricas e a métrica principal para regressão.
        else:
            metric_weights = {'r2': 0.5, 'rmse': -0.3, 'mae': -0.2} # RMSE e MAE são negativos porque menores são melhores
            main_metric = 'r2'

        # Inicializa a melhor pontuação e o nome do melhor modelo.
        best_score = -float('inf')
        best_name = ""

        # Itera sobre os resultados de cada modelo.
        for name, metrics in self.results.items():
            # Inicializa a pontuação ponderada.
            weighted_score = 0

            # Calcula a pontuação ponderada com base nas métricas e seus pesos.
            for metric, weight in metric_weights.items():
                if metric in metrics:
                    value = metrics[metric]
                    # Normaliza RMSE e MAE (onde valores menores são melhores).
                    if metric in ['rmse', 'mae']:
                        # Coleta todos os valores da métrica para normalização.
                        metric_values = [m.get(metric, 0) for m in self.results.values() if metric in m]
                        # Encontra o valor máximo para normalização.
                        max_val = max(metric_values) if metric_values else 0
                        if max_val > 0:
                            # Normaliza para que 1 seja o melhor (menor valor).
                            normalized = 1 - (value / max_val)
                            weighted_score += normalized * abs(weight)
                    # Para outras métricas (onde valores maiores são melhores).
                    else:
                        weighted_score += value * weight

            # Combina a métrica principal com a pontuação ponderada.
            if main_metric in metrics:
                main_score = metrics[main_metric]
                final_score = 0.7 * main_score + 0.3 * weighted_score # Ponderação entre métrica principal e score ponderado

                if final_score > best_score:
                    best_score = final_score
                    best_name = name

        # Armazena o nome e o objeto do melhor modelo.
        self.best_model_name = best_name
        self.best_model = self.models.get(best_name)

        # Adiciona o score ponderado ao resultado do melhor modelo.
        if best_name in self.results:
            self.results[best_name]['weighted_score'] = float(best_score)

    # Método para gerar um DataFrame de ranking dos modelos.
    def get_ranking(self):
        """Ranking com todas as métricas"""
        # Se não houver resultados, retorna um DataFrame vazio.
        if not self.results:
            return pd.DataFrame(columns=['Modelo', 'Score', 'Tipo', 'CV Score ± Std'])

        # Lista para armazenar os dados do ranking.
        ranking = []
        # Itera sobre os resultados de cada modelo.
        for name, metrics in self.results.items():
            # Obtém a métrica principal e seu desvio padrão para classificação.
            if self.problem_type == 'classification':
                score = metrics.get('accuracy', metrics.get('f1', metrics.get('score', 0)))
                score_std = metrics.get('accuracy_std', 0)
            # Obtém a métrica principal e seu desvio padrão para regressão.
            else:
                score = metrics.get('r2', metrics.get('explained_variance', metrics.get('score', 0)))
                score_std = metrics.get('r2_std', 0)

            # Determina o tipo do modelo.
            model_type = self.get_model_type(name)
            # Formata o score e seu desvio padrão para exibição.
            cv_score = f"{float(score):.4f} ± {float(score_std):.4f}"

            # Adiciona os dados do modelo à lista de ranking.
            ranking.append({
                'Modelo': name,
                'Score': float(score),
                'CV Score ± Std': cv_score,
                'Tipo': model_type
            })

        # Cria um DataFrame a partir da lista de ranking.
        df = pd.DataFrame(ranking)
        # Classifica o DataFrame pelo score em ordem decrescente.
        df = df.sort_values('Score', ascending=False).reset_index(drop=True)
        # Insere uma coluna de 'Posição'.
        df.insert(0, 'Posição', range(1, len(df) + 1))

        # Retorna o DataFrame de ranking.
        return df

    # Método para determinar o tipo de um modelo com base em seu nome.
    def get_model_type(self, model_name):
        """Determina o tipo do modelo baseado no nome"""
        # Converte o nome do modelo para minúsculas para comparação.
        model_name_lower = model_name.lower()

        # Verifica e retorna o tipo de boosting.
        if any(x in model_name_lower for x in ['xgboost', 'lightgbm']):
            return 'Boosting'
        # Verifica e retorna o tipo de ensemble.
        elif any(x in model_name_lower for x in ['random forest', 'extra trees', 'bagging']):
            return 'Ensemble'
        # Verifica e retorna o tipo de SVM.
        elif any(x in model_name_lower for x in ['svm', 'svc', 'svr']):
            return 'SVM'
        # Verifica e retorna o tipo linear.
        elif any(x in model_name_lower for x in ['linear', 'logistic', 'ridge', 'lasso', 'elastic']):
            return 'Linear'
        # Verifica e retorna o tipo KNN.
        elif any(x in model_name_lower for x in ['knn', 'neighbors']):
            return 'KNN'
        # Verifica e retorna o tipo de árvore.
        elif any(x in model_name_lower for x in ['tree', 'decision']):
            return 'Árvore'
        # Verifica e retorna o tipo Bayes.
        elif any(x in model_name_lower for x in ['naive', 'bayes']):
            return 'Bayes'
        # Verifica e retorna o tipo de rede neural.
        elif any(x in model_name_lower for x in ['mlp', 'neural']):
            return 'Neural'
        # Verifica e retorna o tipo de boosting (alternativo, já que alguns já foram pegos).
        elif any(x in model_name_lower for x in ['adaboost', 'gradient']):
            return 'Boosting'
        # Retorna 'Outro' se nenhum tipo for correspondido.
        else:
            return 'Outro'

# ========== DETECTOR INTELIGENTE DE TARGET ==========
class TargetDetector:
    """Detecta automaticamente a coluna target"""

    @staticmethod
    def detect_target(data, user_hint=None):
        """
        Detecta a coluna target automaticamente com inteligência
        Retorna: (target_col, X, y, confidence_score, problem_type)
        """
        # Verifica se uma sugestão de coluna target foi fornecida e se ela existe no DataFrame.
        if user_hint and user_hint in data.columns:
            # Se a sugestão existe, cria um DataFrame 'X' (features) removendo a coluna target sugerida.
            X = data.drop(columns=[user_hint]).copy()
            # Cria uma Série 'y' (target) com os valores da coluna target sugerida.
            y = data[user_hint].copy()
            # Detecta o tipo de problema (classificação ou regressão) com base na série 'y'.
            problem_type = TargetDetector.detect_problem_type(y)
            # Retorna a coluna target, os dados X e y, uma confiança alta (1.0) e o tipo de problema.
            return user_hint, X, y, 1.0, problem_type

        st.info("🔍 Analisando dataset para detectar target automaticamente...")

        scores = {}

        # Itera sobre cada coluna no DataFrame de entrada 'data'.
        for col in data.columns:
            # Chama o método estático 'analyze_column' para obter uma pontuação de "target" para a coluna atual.
            score = TargetDetector.analyze_column(data[col], col)
            # Armazena a pontuação no dicionário 'scores', usando o nome da coluna como chave.
            scores[col] = score

        # Classifica as colunas com base em suas pontuações de "target" em ordem decrescente.
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Exibe um cabeçalho para a seção de análise automática no Streamlit.
        st.write("📊 **Análise automática:**")
        # Cria um DataFrame Pandas a partir das pontuações ordenadas para exibição.
        analysis_df = pd.DataFrame(sorted_scores, columns=['Coluna', 'Score Target'])

        # Divide a interface do Streamlit em duas colunas.
        col1, col2 = st.columns(2)
        with col1:
            # Exibe as 10 principais colunas com suas pontuações de "target" em um DataFrame.
            st.dataframe(analysis_df.head(10), use_container_width=True)

        with col2:
            # Verifica se há pontuações de colunas disponíveis.
            if len(sorted_scores) > 0:
            # Obtém o nome da coluna com a maior pontuação.
            top_col = sorted_scores[0][0]
            try:
                # Tenta criar e exibir um histograma da distribuição da coluna principal.
                fig = px.histogram(data, x=top_col, title=f"Distribuição: {top_col}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                # Em caso de erro ao gerar o gráfico, exibe uma mensagem.
                st.write(f"*Não foi possível criar gráfico para {top_col}*")

        # Filtra os principais candidatos a target, pegando os 3 primeiros com score acima de 0.3.
        top_candidates = [col for col, score in sorted_scores[:3] if score > 0.3]

        # Verifica se não há candidatos de target fortes.
        if not top_candidates:
            # Exibe um aviso se a detecção automática falhou.
            st.warning("⚠️ Não consegui detectar target automaticamente.")
            # Define a última coluna como target padrão.
            target_col = data.columns[-1]
            # Define uma baixa confiança.
            confidence = 0.1
        else:
            # Exibe um cabeçalho para a seleção de target.
            st.write("🎯 **Candidatos a target (escolha ou confirme):**")
            # Permite ao usuário selecionar a coluna target a partir dos candidatos ou escolher "Nenhuma das acima".
            target_col = st.selectbox(
            "Selecione a coluna target:",
            options=top_candidates + ["⚠️ Nenhuma das acima"], # Adiciona a opção de fallback manual.
            index=0, # Define o primeiro item como padrão.
            key="auto_target_select" # Chave única para o widget Streamlit.
            )

            # Verifica se o usuário selecionou a opção de fallback manual.
            if target_col == "⚠️ Nenhuma das acima":
                # Se sim, apresenta um novo selectbox com todas as colunas para seleção manual.
                target_col = st.selectbox(
                    "Selecione manualmente:",
                    options=data.columns.tolist(), # Opções são todas as colunas do DataFrame.
                    index=len(data.columns) - 1, # Define a última coluna como padrão.
                    key="manual_fallback_select" # Chave única para o widget Streamlit.
                )
                confidence = 0.5 # Define uma confiança média para a seleção manual.
            else:
                confidence = scores[target_col] # Se uma coluna foi selecionada, usa a pontuação calculada.

        # Separa as features (X) removendo a coluna target.
        X = data.drop(columns=[target_col]).copy()
        # Separa o target (y) pegando a coluna target.
        y = data[target_col].copy()

        # Detecta o tipo de problema (classificação ou regressão) com base na série 'y'.
        problem_type = TargetDetector.detect_problem_type(y)

        # Exibe mensagens de sucesso no Streamlit com a coluna target detectada/selecionada, confiança e tipo de problema.
        st.success(f"✅ Target detectado: **{target_col}** (confiança: {confidence:.2f})")
        st.success(f"📊 Tipo de problema: **{problem_type.upper()}**")
        st.write(f"📐 Dimensões: X={X.shape}, y={y.shape}")

        # Retorna a coluna target, os dados X e y, a pontuação de confiança e o tipo de problema.
        return target_col, X, y, confidence, problem_type

    @staticmethod
    def analyze_column(column, col_name):
        """Analisa uma coluna e retorna score de ser target"""
        score = 0 # Inicializa o score da coluna como 0.

        try:
            n_unique = column.nunique() # Calcula o número de valores únicos na coluna.
            n_total = len(column) # Obtém o número total de elementos na coluna.
            unique_ratio = n_unique / n_total if n_total > 0 else 0 # Calcula a proporção de valores únicos.

            if n_unique <= 10: # Se o número de valores únicos for pequeno (até 10), sugere que pode ser uma classe.
                score += 0.3 # Adiciona 0.3 ao score.
            elif unique_ratio > 0.9: # Se a proporção de valores únicos for muito alta (quase todos únicos), sugere que não é um ID.
                score += 0.2 # Adiciona 0.2 ao score.

            target_keywords = ['target', 'label', 'class', 'score', 'rating',
                               'price', 'value', 'output', 'result', 'y'] # Define palavras-chave comuns para colunas target.
            col_lower = col_name.lower() # Converte o nome da coluna para minúsculas.
            if any(keyword in col_lower for keyword in target_keywords): # Verifica se alguma palavra-chave está no nome da coluna.
                score += 0.4 # Adiciona 0.4 ao score se encontrar uma palavra-chave.

            if n_unique > 1: # Se houver mais de um valor único.
                value_counts = column.value_counts(normalize=True) # Calcula a contagem de frequência normalizada dos valores.
                try:
                    entropy = -sum(p * np.log(p) for p in value_counts if p > 0) # Calcula a entropia da coluna.
                    max_entropy = np.log(n_unique) # Calcula a entropia máxima possível para o número de valores únicos.
                    if max_entropy > 0: # Evita divisão por zero.
                        normalized_entropy = entropy / max_entropy # Normaliza a entropia.
                        if normalized_entropy < 0.7: # Se a entropia normalizada for baixa, sugere que é um target (classes bem definidas).
                            score += 0.2 # Adiciona 0.2 ao score.
                except Exception:
                    pass # Ignora erros no cálculo da entropia.

            if pd.api.types.is_numeric_dtype(column): # Verifica se a coluna é de tipo numérico.
                try:
                    if column.abs().max() > 1000: # Se o valor máximo absoluto for muito alto, sugere regressão.
                        score += 0.1 # Adiciona 0.1 ao score.
                except Exception:
                    pass # Ignora erros ao verificar o valor máximo.

            missing_ratio = column.isna().sum() / n_total if n_total > 0 else 0 # Calcula a proporção de valores ausentes.
            if missing_ratio > 0.3: # Se houver muitos valores ausentes, diminui o score (menos provável de ser target).
                score -= 0.3 # Subtrai 0.3 do score.

            if any(x in col_lower for x in ['id', 'code', 'num', 'index', 'key']): # Verifica palavras-chave comuns para IDs.
                score -= 0.4 # Diminui o score se parecer ser um ID.

            if n_unique == n_total and n_total > 100: # Se todos os valores forem únicos e há muitas linhas, sugere um ID/identificador.
                score -= 0.5 # Diminui o score significativamente.

            date_keywords = ['date', 'time', 'day', 'month', 'year'] # Define palavras-chave comuns para datas.
            if any(x in col_lower for x in date_keywords): # Verifica se o nome da coluna contém palavras-chave de data.
                score -= 0.3 # Diminui o score se parecer ser uma coluna de data.

            score = max(0, min(1, score)) # Garante que o score esteja entre 0 e 1.

        except Exception:
            score = 0 # Em caso de qualquer erro, define o score como 0.

        return round(score, 3) # Retorna o score arredondado para 3 casas decimais.

    @staticmethod
    def detect_problem_type(y):
        """Detecta se é classificação ou regressão de forma robusta"""
        try:
            y_numeric = pd.to_numeric(y, errors='coerce')
            not_na = y_numeric.notna().sum()

            if not_na / len(y) < 0.8:
                return 'classification'

            y_clean = y_numeric.dropna()
            if len(y_clean) == 0:
                return 'classification'

            unique_vals = len(y_clean.unique())

            if unique_vals <= 5:
                try:
                    if all(y_clean.astype(int) == y_clean):
                        return 'classification'
                    else:
                        return 'regression'
                except Exception:
                    return 'classification'

            elif unique_vals <= 20:
                value_counts = y_clean.value_counts(normalize=True)
                if (value_counts > 0.25).any():
                    try:
                        if all(y_clean.astype(int) == y_clean):
                            return 'classification'
                        else:
                            return 'regression'
                    except Exception:
                        return 'classification'
                else:
                    return 'regression'
            else:
                return 'regression'

        except Exception:
            try:
                if hasattr(y, 'dtype'):
                    if y.dtype == 'object' or len(y.unique()) <= 10:
                        return 'classification'
                    else:
                        return 'regression'
                else:
                    unique_vals = len(np.unique(y))
                    if unique_vals <= 10:
                        return 'classification'
                    else:
                        return 'regression'
            except Exception:
                return 'regression'

# ========== APLICAÇÃO PRINCIPAL COM FIXES ==========
class UltraRobustApp:
    def __init__(self):
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            st.session_state.step = 1
            st.session_state.data = None
            st.session_state.processed = False
            st.session_state.processor_type = "POWERFULL"
            st.session_state.trainer_type = "ULTRA_COMPLETE"
            st.session_state.last_rerun = time.time()
            st.session_state.n_folds = 5
            st.session_state.cv_strategy = "Auto (Recomendado)"
            st.session_state.random_state = 42
            st.session_state.parallel = True

    def safe_rerun(self, delay=0.1):
        """Rerun seguro com delay"""
        current_time = time.time()
        if current_time - st.session_state.last_rerun > 0.5:
            time.sleep(delay)
            st.session_state.last_rerun = current_time
            try:
                st.rerun()
            except Exception:
                st.rerun()
        else:
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

            self.show_progress()

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
        steps = [" Upload", " Processar", " Treinar", "📊 Resultados"]
        current = st.session_state.step - 1

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
        st.header(" Upload do Dataset")

        with st.container():
            uploaded_file = st.file_uploader(
                "Escolha um arquivo CSV",
                type=['csv', 'txt', 'xlsx'],
                help="Suporta CSV, TXT e Excel",
                key="main_file_uploader"
            )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                else:
                    data = pd.read_csv(uploaded_file)

                st.success(f" Dataset carregado: {data.shape[0]} linhas × {data.shape[1]} colunas")

                if st.checkbox(" Visualizar dados", key="show_preview_upload"):
                    st.dataframe(data.head(), use_container_width=True)

                st.subheader(" Seleção do Target")

                use_auto = st.checkbox(" Usar detecção automática", value=True, key="use_auto_detect")

                if use_auto:
                    try:
                        target_col, X, y, confidence, problem_type = TargetDetector.detect_target(data)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(" Target", target_col)
                        with col2:
                            st.metric(" Tipo", problem_type.upper())

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
                    target_options = data.columns.tolist()
                    default_idx = len(target_options) - 1

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

                    X = data.drop(columns=[target_col]).copy()
                    y = data[target_col].copy()

                    try:
                        problem_type = TargetDetector.detect_problem_type(y)
                    except Exception:
                        if y.dtype == 'object' or len(y.unique()) <= 10:
                            problem_type = 'classification'
                        else:
                            problem_type = 'regression'

                    st.session_state.target_col = target_col
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.problem_type = problem_type
                    st.session_state.auto_detected = False
                    st.session_state.data = data

                    st.success(f"✅ Target selecionado: {target_col}")
                    st.success(f" Tipo: {problem_type.upper()}")

                st.markdown("---")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Novo Upload", type="secondary", key="new_upload_simple_btn"):
                        keys_to_keep = ['app_initialized', 'last_rerun', 'n_folds', 'cv_strategy', 'random_state', 'parallel']
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
                            if len(st.session_state.X) < 10:
                                st.error("❌ Muito poucas amostras (mínimo 10)")
                            else:
                                st.session_state.step = 2
                                time.sleep(0.5)
                                st.rerun()

            except Exception as e:
                st.error(f"❌ Erro ao processar arquivo: {str(e)}")

                try:
                    uploaded_file.seek(0)
                    data = pd.read_csv(uploaded_file, encoding='latin-1')
                    st.success("✅ Carregado com encoding alternativo")
                    st.session_state.data = data
                    st.rerun()
                except Exception:
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

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Amostras", st.session_state.data.shape[0])
        with col2:
            st.metric("Features", st.session_state.data.shape[1] - 1)
        with col3:
            st.metric("Target", st.session_state.target_col)

        if st.button("Treinamento", type="primary", key="process_execute_btn"):
            with st.spinner("Processando dados..."):
                try:
                    processor = PowerfulDataProcessor()

                    X, y, problem_type = processor.process(
                        st.session_state.data,
                        st.session_state.target_col
                    )

                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.problem_type = problem_type
                    st.session_state.processed = True

                    st.success("✅ Processamento concluído!")

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

                    time.sleep(1)

                except Exception as e:
                    st.error(f"❌ Erro no processamento: {str(e)}")
                    try:
                        X = st.session_state.data.drop(columns=[st.session_state.target_col]).values
                        y = st.session_state.data[st.session_state.target_col].values

                        st.session_state.X = X
                        st.session_state.y = y
                        st.session_state.processed = True

                        st.success("✅ Processamento simples realizado")
                    except Exception:
                        st.error("❌ Não foi possível processar os dados.")

        if st.session_state.get('processed', False):
            st.markdown("---")
            if st.button(" Ir para Treinamento →", type="primary", key="go_to_train_btn"):
                st.session_state.step = 3
                time.sleep(0.5)
                st.rerun()

        if st.button("⬅️ Voltar", key="back_from_process_btn"):
            st.session_state.step = 1
            time.sleep(0.5)
            st.rerun()

    def step_train(self):
        """Treinamento com fix"""
        st.header(" Treinamento com VALIDAÇÃO CRUZADA")

        if not st.session_state.get('processed', False):
            st.warning("Dados não processados.")
            if st.button("⬅️ Voltar", key="back_to_process_train"):
                st.session_state.step = 2
                time.sleep(0.1)
                st.rerun()
            return

        with st.expander(" Estatísticas do Dataset"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Amostras", len(st.session_state.X))
            with col2:
                st.metric("Features", st.session_state.X.shape[1] if hasattr(st.session_state.X, "shape") else 0)
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

        with st.container():
            st.info(" **VALIDAÇÃO CRUZADA ATIVADA**")

            with st.expander("⚙️ Configurações da Validação Cruzada"):
                col1, col2 = st.columns(2)
                with col1:
                    n_folds = st.slider(
                        "Número de folds", 3, 10, st.session_state.get('n_folds', 5),
                        help="Mais folds = mais robusto, mas mais lento",
                        key="n_folds_slider_train"
                    )
                    cv_strategy = st.selectbox(
                        "Estratégia CV",
                        ["Auto (Recomendado)", "Stratified K-Fold", "K-Fold"],
                        index=["Auto (Recomendado)", "Stratified K-Fold", "K-Fold"].index(st.session_state.get('cv_strategy', "Auto (Recomendado)")),
                        help="Auto escolhe a melhor baseado nos dados",
                        key="cv_strategy_select_train"
                    )
                with col2:
                    random_state = st.number_input(
                        "Random State", 0, 100, st.session_state.get('random_state', 42),
                        key="random_state_input_train"
                    )
                    parallel = st.checkbox(
                        "Treinamento Paralelo", value=st.session_state.get('parallel', True),
                        help="Usa todos os cores da CPU (mais rápido)",
                        key="parallel_checkbox_train"
                    )

                st.session_state.n_folds = n_folds
                st.session_state.cv_strategy = cv_strategy
                st.session_state.random_state = random_state
                st.session_state.parallel = parallel

            st.warning("⚠️ O treinamento testará **15+ modelos** e pode levar alguns minutos.")

            if st.button(" INICIAR TREINAMENTO COMPLETO", type="primary", key="start_training_main_btn"):
                self._execute_training()

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
                trainer.n_folds = int(st.session_state.get('n_folds', 5))

                results, best_model = trainer.train_safe(X, y)

                st.session_state.results = results
                st.session_state.trainer = trainer
                st.session_state.best_model = best_model

                st.success("✅ Treinamento concluído!")

                time.sleep(1)
                st.session_state.step = 4
                st.rerun()

            except Exception as e:
                st.error(f"❌ Erro no treinamento: {str(e)}")

    def step_results(self):
        """Resultados"""
        st.header(" Resultados")

        if 'results' not in st.session_state:
            st.warning("Nenhum resultado disponível.")
            if st.button("⬅️ Voltar", key="back_to_train_results"):
                st.session_state.step = 3
                time.sleep(0.1)
                st.rerun()
            return

        try:
            results = st.session_state.results
            trainer = st.session_state.trainer
            problem_type = st.session_state.problem_type

            best_name = trainer.best_model_name
            if best_name and best_name in results:
                best_metrics = results[best_name]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(" Melhor Modelo", best_name)
                with col2:
                    if problem_type == 'classification':
                        score = best_metrics.get('accuracy', best_metrics.get('f1', 0))
                    else:
                        score = best_metrics.get('r2', best_metrics.get('explained_variance', 0))
                    st.metric(" Score", f"{float(score):.4f}")
                with col3:
                    st.metric(" Modelos Treinados", len(results))

            with st.expander(" Ranking Completo"):
                ranking_df = trainer.get_ranking()
                ranking_display = ranking_df.copy()
                if 'Score' in ranking_display.columns:
                    ranking_display['Score'] = ranking_display['Score'].map(lambda x: f"{float(x):.4f}")
                st.dataframe(ranking_display, use_container_width=True)

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

            if 'best_model' in st.session_state and st.session_state.best_model is not None:
                with st.expander(" Métricas Detalhadas"):
                    model_options = list(results.keys())
                    selected_model = st.selectbox(
                        "Selecione um modelo para ver métricas detalhadas:",
                        model_options,
                        key="model_select_detailed_results"
                    )

                    if selected_model in results:
                        metrics = results[selected_model]

                        cols = st.columns(4)
                        metric_count = 0

                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float, np.floating, np.integer)) and '_std' not in metric_name:
                                with cols[metric_count % 4]:
                                    st.metric(
                                        label=metric_name.upper(),
                                        value=f"{float(value):.4f}",
                                        delta=f"± {float(metrics.get(f'{metric_name}_std', 0)):.4f}" if f'{metric_name}_std' in metrics else None
                                    )
                                metric_count += 1

                        if 'cv_type' in metrics:
                            st.write("---")
                            st.write(f"**Estratégia CV:** {metrics['cv_type']}")
                            st.write(f"**Número de folds:** {metrics.get('n_folds', 5)}")
                            st.write(f"**Tempo médio de treino:** {float(metrics.get('fit_time', 0)):.2f}s")
                            st.write(f"**Tempo médio de score:** {float(metrics.get('score_time', 0)):.2f}s")

            st.subheader(" Exportar Resultados")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(" Exportar CSV", key="export_csv_results_btn"):
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
                if st.button(" Salvar Modelo", key="save_model_results_btn"):
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
                if st.button(" Gerar Relatório", key="generate_report_btn"):
                    with st.spinner("Gerando relatório..."):
                        try:
                            data_info = {
                                'dataset_name': 'Dataset Processado',
                                'n_samples': st.session_state.X.shape[0] if 'X' in st.session_state and hasattr(st.session_state.X, 'shape') else 'N/A',
                                'n_features': st.session_state.X.shape[1] if 'X' in st.session_state and hasattr(st.session_state.X, 'shape') and len(st.session_state.X.shape) > 1 else 'N/A',
                            }

                            report_path = PDFReportGenerator.generate_report(
                                results,
                                trainer,
                                problem_type,
                                data_info
                            )

                            if report_path and os.path.exists(report_path):
                                with open(report_path, 'rb') as f:
                                    file_bytes = f.read()

                                ext = os.path.splitext(report_path)[1].lower()
                                mime_type = "application/pdf" if ext == ".pdf" else "text/plain"

                                st.download_button(
                                    "⬇️ Baixar Relatório",
                                    file_bytes,
                                    os.path.basename(report_path),
                                    mime_type,
                                    key="download_report_btn"
                                )
                            else:
                                st.warning("Não foi possível gerar o relatório.")
                        except Exception as e:
                            st.error(f"❌ Erro no relatório: {str(e)}")

            st.markdown("---")
            col1, col2 = st.columns([1, 3])

            with col1:
                if st.button("⬅️ Voltar", key="back_to_train_final"):
                    st.session_state.step = 3
                    time.sleep(0.1)
                    st.rerun()

            with col2:
                if st.button(" Novo Dataset", type="primary", key="new_dataset_btn"):
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
            if st.button(" Reiniciar Aplicação", key="restart_app_results"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    def _clear_state(self):
        """Limpa estado de forma segura"""
        keys_to_preserve = ['app_initialized', 'last_rerun', 'n_folds', 'cv_strategy', 'random_state', 'parallel']
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
    warnings.filterwarnings('ignore')

    try:
        app = UltraRobustApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Erro crítico na aplicação: {str(e)}")
        st.button(" Reiniciar Aplicação", key="restart_app_final", on_click=lambda: st.rerun())