# Esse código implementa um dashboard interativo de Machine Learning usando o framework Dash
# (com Bootstrap e Plotly).
# Ele cria uma aplicação web que permite visualizar, comparar e exportar resultados
# de modelos de Machine Learning. O usuário pode abrir o dashboard no navegador e
# interagir com gráficos, tabelas e botões.


# Importa bibliotecas para codificação, serialização e data/hora
import base64
import pickle
from datetime import datetime

# Importa o framework Dash para criar o dashboard
import dash
# Importa componentes do Dash para criar elementos interativos e layout
from dash import dcc, html, Input, Output, dash_table
# Importa componentes adicionais de estilo do Dash Bootstrap Components
import dash_bootstrap_components as dbc
# Importa o pandas para manipulação de dados
import pandas as pd
# Importa o numpy para operações numéricas
import numpy as np
# Importa objetos gráficos do Plotly
import plotly.graph_objs as go
# Importa expressões do Plotly para gráficos simplificados
import plotly.express as px
# Importa subplots do Plotly para criar gráficos com múltiplos subgráficos
from plotly.subplots import make_subplots
# Importa matriz de confusão para problemas de classificação
from sklearn.metrics import confusion_matrix


# Define a classe principal para o dashboard
class AdvancedDashboard:
    # Define métricas conhecidas para classificação
    CLASSIFICATION_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
    # Define métricas conhecidas para regressão
    REGRESSION_METRICS = ("r2", "rmse", "mae", "mse")

    # Método construtor da classe
    def __init__(self, results, models, feature_importance=None, X_test=None, y_test=None):
        # Inicializa os resultados dos modelos
        self.results = results or {}
        # Inicializa os modelos treinados
        self.models = models or {}
        # Inicializa a importância das features (opcional)
        self.feature_importance = feature_importance or {}
        # Inicializa os dados de teste (opcional)
        self.X_test = X_test
        # Inicializa os rótulos de teste (opcional)
        self.y_test = y_test
        # Cria a aplicação Dash com um tema externo
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        # Configura o layout do dashboard
        self.setup_layout()
        # Configura os callbacks do dashboard
        self.setup_callbacks()

    # Verifica se um valor é numérico válido
    def _is_numeric(self, value):
        """Verifica se o valor é numérico válido"""
        return isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value)

    # Obtém o nome do primeiro modelo disponível
    def _default_model_name(self):
        """Retorna o primeiro modelo disponível"""
        return next(iter(self.models.keys()), None)

    # Detecta o tipo de problema baseado nas métricas disponíveis nos resultados
    def detect_problem_type(self):
        """Detecta o tipo de problema baseado nas métricas"""
        # Retorna "Indeterminado" se não houver resultados
        if not self.results:
            return "Indeterminado"

        # Agrupa todas as métricas encontradas em todos os modelos
        all_keys = set()
        for metrics in self.results.values():
            if isinstance(metrics, dict):
                all_keys.update(metrics.keys())

        # Verifica se há métricas típicas de classificação
        if any(metric in all_keys for metric in self.CLASSIFICATION_METRICS):
            return "Classificação"
        # Verifica se há métricas típicas de regressão
        if any(metric in all_keys for metric in self.REGRESSION_METRICS):
            return "Regressão"

        # Retorna "Indeterminado" se nenhuma métrica conhecida for encontrada
        return "Indeterminado"

    # Obtém a métrica principal para ranquear os modelos
    def get_primary_metric(self, metrics):
        """Obtém a métrica principal para ranking"""
        # Retorna 0 se as métricas não estiverem em formato esperado
        if not isinstance(metrics, dict):
            return 0.0

        # Detecta o tipo de problema
        problem_type = self.detect_problem_type()

        # Para classificação, prioriza F1, depois ROC AUC, depois acurácia
        if problem_type == "Classificação":
            for key in ("f1", "roc_auc", "accuracy", "precision", "recall"):
                value = metrics.get(key)
                if self._is_numeric(value):
                    return float(value)

        # Para regressão, prioriza R²; se não existir, usa erro invertido
        if problem_type == "Regressão":
            # Verifica se a métrica "r2" existe e é numérica
            if self._is_numeric(metrics.get("r2")):
                # Se sim, retorna o valor de R² como float
                return float(metrics["r2"])

            # Itera sobre as métricas de erro comuns para regressão (RMSE, MAE, MSE)
            for key in ("rmse", "mae", "mse"):
                # Obtém o valor da métrica atual
                value = metrics.get(key)
                # Verifica se o valor da métrica existe e é numérico
                if self._is_numeric(value):
                    # Se sim, retorna o valor negativo da métrica de erro como float.
                    # Isso é feito para que métricas de erro (onde menor é melhor)
                    # possam ser comparadas da mesma forma que métricas de acerto (onde maior é melhor)
                    # ao buscar o "melhor" modelo usando `max()`.
                    return -float(value)


        # Retorna 0 se nenhuma métrica principal for encontrada
        return 0.0

    # Obtém informações do melhor modelo
    def _get_best_model_info(self):
        """Retorna o nome e a métrica principal do melhor modelo"""
        # Retorna valores nulos se não houver resultados
        if not self.results:
            return None, None

        # Inicializa variáveis para armazenar o nome do melhor modelo e suas métricas.
        # Usa a função `max` para encontrar o item (modelo, métricas) com a maior métrica principal.
        best_name, best_metrics = max(
            # Itera sobre os pares (nome do modelo, dicionário de métricas) dos resultados.
            self.results.items(),
            # Define a chave de comparação para a função `max`: a métrica principal de cada modelo.
            key=lambda item: self.get_primary_metric(item[1])
        )

        # Retorna nome e métrica do melhor modelo
        return best_name, self.get_primary_metric(best_metrics)

    # Ordena nomes de métricas de forma consistente
    def _ordered_metric_names(self, metric_names):
        """Ordena as métricas em uma ordem amigável"""
        priority = [
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "r2", "rmse", "mae", "mse"
        ]

        # Define uma função interna 'sort_key' que será usada para ordenar os nomes das métricas.
        def sort_key(name):
            # Verifica se o nome da métrica atual está na lista de prioridade predefinida.
            if name in priority:
            # Se estiver na lista de prioridade, retorna uma tupla (0, índice) para dar prioridade a essas métricas
            # e ordená-las pela sua posição na lista 'priority'.
                return (0, priority.index(name))
            # Se a métrica não estiver na lista de prioridade, retorna uma tupla (1, nome)
            # para que essas métricas venham depois das prioritárias e sejam ordenadas alfabeticamente.
            return (1, name)

        # Retorna a lista de nomes de métricas ordenadas usando a função 'sort_key' definida acima.
        return sorted(metric_names, key=sort_key)

    # Obtém a lista de métricas numéricas disponíveis
    def _get_numeric_metric_names(self):
        """Retorna as métricas numéricas encontradas nos resultados"""
        metric_names = set() # Inicializa um conjunto vazio para armazenar os nomes das métricas numéricas únicas.

        for metrics in self.results.values(): # Itera sobre os dicionários de métricas de cada modelo nos resultados.
            if not isinstance(metrics, dict): # Verifica se o item 'metrics' é realmente um dicionário.
                continue # Se não for, pula para o próximo item.

            for key, value in metrics.items(): # Itera sobre cada par chave-valor (nome da métrica, valor da métrica) dentro do dicionário de métricas do modelo.
                if key == "confusion_matrix": # Verifica se a chave da métrica é "confusion_matrix".
                    continue # Se for, pula para a próxima métrica, pois a matriz de confusão não é uma métrica numérica simples.
                if self._is_numeric(value): # Chama o método auxiliar '_is_numeric' para verificar se o valor da métrica é numérico.
                    metric_names.add(key) # Se o valor for numérico, adiciona o nome da métrica ao conjunto 'metric_names'.

        return self._ordered_metric_names(metric_names) # Retorna a lista de nomes de métricas numéricas, ordenada de forma consistente usando o método auxiliar '_ordered_metric_names'.

    # Formata métricas para exibição
    def _format_metric(self, value):
        """Formata o valor da métrica para exibição"""
        # Verifica se o valor é None.
        if value is None:
            # Se for None, retorna "N/A" (Not Applicable).
            return "N/A"
        # Verifica se o valor é numérico usando o método auxiliar _is_numeric.
        if self._is_numeric(value):
            # Se for numérico, formata o valor como float com 4 casas decimais e retorna como string.
            return f"{float(value):.4f}"
        # Se não for numérico e não for None, converte o valor para string e o retorna.
        return str(value)

    # Cria um gráfico vazio com mensagem central
    def _build_empty_figure(self, title, message):
        """Cria uma figura vazia com mensagem amigável"""
        fig = go.Figure()

        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16)
        )

        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=420
        )

        return fig

    # Normaliza a estrutura da importância das features
    def _normalize_feature_importance(self, model_name):
        """Converte diferentes formatos de feature importance em listas"""
        # Retorna listas vazias se o modelo não possuir feature importance
        if model_name not in self.feature_importance:
            return [], []

        fi = self.feature_importance[model_name]

        # Caso seja uma Series do pandas
        if isinstance(fi, pd.Series):
            return fi.index.astype(str).tolist(), fi.values.astype(float).tolist()

        # Caso seja um dicionário
        if isinstance(fi, dict):
            keys = list(fi.keys())
            values = [float(v) for v in fi.values()]
            return [str(k) for k in keys], values

        # Caso seja uma lista, array ou estrutura compatível
        values = np.asarray(fi).ravel().tolist()

        # Usa os nomes das colunas do X_test, se existirem
        if hasattr(self.X_test, 'columns'):
            # Obtém os nomes das colunas de X_test e as converte para string
            columns = list(map(str, self.X_test.columns))
            # Verifica se o número de nomes de colunas corresponde ao número de valores de importância
            if len(columns) == len(values):
                # Se sim, retorna os nomes das colunas e os valores de importância convertidos para float
                return columns, [float(v) for v in values]

        # Caso contrário, cria nomes genéricos de features
        features = [f'Feature_{i}' for i in range(len(values))]
        return features, [float(v) for v in values]

    # Cria a tabela detalhada de resultados
    def _build_results_table(self):
        """Cria a tabela detalhada dos resultados"""
        # Retorna um alerta caso não haja resultados
        if not self.results:
            return dbc.Alert("Nenhum resultado disponível para exibição.", color="secondary")

        rows = []
        # Obtém o melhor modelo
        best_model_name, _ = self._get_best_model_info()

        # Ordena os resultados dos modelos com base na métrica principal
        sorted_results = sorted( # Inicia a ordenação dos resultados
            self.results.items(), # Pega os itens do dicionário de resultados (nome do modelo, métricas)
            key=lambda item: self.get_primary_metric(item[1]), # Usa a métrica principal de cada modelo como chave para ordenação
            reverse=True # Ordena em ordem decrescente (do melhor para o pior)
        )

        # Monta cada linha da tabela
        for rank, (model_name, metrics) in enumerate(sorted_results, start=1):
            row = {
                "Rank": rank,
                "Modelo": model_name,
                "Melhor": "✅" if model_name == best_model_name else "",
                "Métrica_Principal": round(self.get_primary_metric(metrics), 6),
            }

            if isinstance(metrics, dict):
                # Itera sobre cada par chave-valor (nome da métrica e seu valor) no dicionário 'metrics'
                for key, value in metrics.items():
                    # Verifica se o valor da métrica é numérico usando o método auxiliar _is_numeric
                    if self._is_numeric(value):
                        # Se for numérico, converte para float, arredonda para 6 casas decimais e adiciona à linha 'row' com a chave da métrica
                        row[key] = round(float(value), 6)

            rows.append(row)

        df = pd.DataFrame(rows)

        # Retorna uma DataTable estilizada
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in df.columns],
            sort_action="native",
            filter_action="native",
            page_action="native",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "10px",
                "backgroundColor": "#1f2630",
                "color": "white",
                "border": "1px solid #3a3f44",
                "minWidth": "120px",
                "width": "120px",
                "maxWidth": "240px",
                "whiteSpace": "normal",
            },
            style_header={
                "backgroundColor": "#111827",
                "color": "white",
                "fontWeight": "bold",
                "border": "1px solid #3a3f44",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": "{Melhor} = '✅'"},
                    "backgroundColor": "#123524",
                    "color": "white",
                }
            ],
        )

    # Cria o gráfico de ranking dos modelos
    def _build_ranking_figure(self):
        """Cria o gráfico de ranking dos modelos"""
        # Retorna figura vazia se não houver resultados
        if not self.results:
            return self._build_empty_figure(
                "Ranking dos Modelos",
                "Nenhum resultado disponível."
            )

        # Ordena os resultados dos modelos com base na métrica principal
        sorted_results = sorted( # Inicia a ordenação dos resultados
            self.results.items(), # Pega os itens do dicionário de resultados (nome do modelo, métricas)
            key=lambda item: self.get_primary_metric(item[1]), # Usa a métrica principal de cada modelo como chave para ordenação
            reverse=True # Ordena em ordem decrescente (do melhor para o pior)
        )

        # Extrai nomes e pontuações
        model_names = [item[0] for item in sorted_results]
        # Extrai as pontuações da métrica principal de cada modelo na lista ordenada de resultados
        scores = [self.get_primary_metric(item[1]) for item in sorted_results]

        # Cria o gráfico de barras horizontal
        fig = go.Figure(
            data=[
                go.Bar(
                    x=scores,
                    y=model_names,
                    orientation='h',
                    marker=dict(
                        color=scores,
                        colorscale="Viridis",
                        showscale=False
                    ),
                    text=[f"{score:.4f}" for score in scores],
                    textposition="auto",
                )
            ]
        )

        # Configura o layout do gráfico
        fig.update_layout(
            title="Ranking dos Modelos",
            xaxis_title="Métrica Principal",
            yaxis_title="Modelo",
            template="plotly_dark",
            height=max(420, 60 * len(model_names)),
            margin=dict(l=40, r=20, t=60, b=40),
        )

        # Inverte a ordem do eixo Y para mostrar o melhor em cima
        fig.update_yaxes(autorange="reversed")

        return fig

    # Cria o gráfico das métricas principais
    def _build_main_metrics_figure(self):
        """Cria o gráfico comparativo das métricas principais"""
        # Retorna figura vazia se não houver resultados
        if not self.results:
            return self._build_empty_figure(
                "Comparação de Métricas",
                "Nenhum resultado disponível."
            )

        # Obtém a lista de modelos
        models = list(self.results.keys())
        # Detecta o tipo de problema
        problem_type = self.detect_problem_type()

        # Define métricas para classificação
        if problem_type == "Classificação":
            metric_defs = [
                ("accuracy", "Acurácia"),
                ("precision", "Precisão"),
                ("recall", "Recall"),
                ("f1", "F1-Score"),
            ]
        else:
            # Define métricas para regressão
            metric_defs = [
                ("r2", "R²"),
                ("rmse", "RMSE"),
                ("mae", "MAE"),
            ]

        # Cria subplots para as métricas
        fig = make_subplots(
            rows=1,
            cols=len(metric_defs),
            subplot_titles=[label for _, label in metric_defs]
        )

        # Adiciona um gráfico para cada métrica
        for i, (metric_key, metric_label) in enumerate(metric_defs, start=1):
            values = []

            for model_name in models:
                # Obtém o valor da métrica para o modelo atual; usa 0 como padrão se a métrica não existir
                value = self.results[model_name].get(metric_key, 0)
                # Converte o valor para float se for numérico válido, caso contrário, define como 0.0
                value = float(value) if self._is_numeric(value) else 0.0

                # Inverte métricas de erro para visualização
                if metric_key in {"rmse", "mae", "mse"}:
                    value = -value

                values.append(value)

            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric_label,
                    hovertemplate="Modelo: %{x}<br>Valor: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=i
            )

            fig.update_xaxes(tickangle=35, row=1, col=i)

        # Configura o layout do gráfico
        fig.update_layout(
            title="Comparação das Métricas Principais",
            template="plotly_dark",
            showlegend=False,
            height=450,
            margin=dict(l=40, r=20, t=60, b=60),
        )

        return fig

    # Cria o gráfico detalhado de métricas
    def _build_detailed_metrics_figure(self):
        """Cria gráfico com métricas detalhadas por modelo"""
        metric_names = self._get_numeric_metric_names()

        # Retorna figura vazia se não houver métricas
        if not metric_names:
            return self._build_empty_figure(
                "Métricas Detalhadas",
                "Nenhuma métrica numérica disponível."
            )

        fig = go.Figure()
        models = list(self.results.keys())

        # Adiciona um traço para cada métrica
        for metric in metric_names:
            values = []

            for model_name in models:
                # Obtém o valor da métrica para o modelo atual; usa 0 como padrão se a métrica não existir
                value = self.results[model_name].get(metric, 0)
                # Converte o valor para float se for numérico válido, caso contrário, define como 0.0
                value = float(value) if self._is_numeric(value) else 0.0

                # Inverte métricas de erro
                if metric in {"rmse", "mae", "mse"}:
                    value = -value

                values.append(value)

            fig.add_trace(go.Bar(name=metric, x=models, y=values))

        # Configura o layout
        fig.update_layout(
            title="Métricas Detalhadas por Modelo",
            template="plotly_dark",
            barmode="group",
            height=520,
            margin=dict(l=40, r=20, t=60, b=80),
        )

        fig.update_xaxes(tickangle=35)

        return fig

    # Cria o heatmap com todas as métricas
    def _build_all_metrics_heatmap(self):
        """Cria heatmap com todas as métricas numéricas"""
        metric_names = self._get_numeric_metric_names() # Obtém uma lista de nomes de métricas numéricas disponíveis

        # Retorna figura vazia se não houver métricas
        if not metric_names: # Verifica se a lista de nomes de métricas está vazia
            return self._build_empty_figure( # Retorna uma figura Plotly vazia com uma mensagem de erro
                "Comparação de Todas as Métricas", # Título da figura vazia
                "Nenhuma métrica numérica disponível." # Mensagem exibida na figura vazia
            )

        models = list(self.results.keys()) # Obtém uma lista dos nomes de todos os modelos nos resultados
        z = [] # Inicializa uma lista vazia que armazenará os dados para o heatmap (matriz Z)

        # Monta a matriz do heatmap
        for model_name in models: # Itera sobre cada nome de modelo
            row = [] # Inicializa uma lista vazia para a linha de métricas do modelo atual

            for metric in metric_names: # Itera sobre cada métrica numérica
                value = self.results[model_name].get(metric, 0) # Obtém o valor da métrica para o modelo atual, ou 0 se não existir
                value = float(value) if self._is_numeric(value) else 0.0 # Converte o valor para float se for numérico, caso contrário, usa 0.0

                # Inverte métricas de erro para que valores menores indiquem melhor performance visualmente
                if metric in {"rmse", "mae", "mse"}: # Verifica se a métrica é uma das métricas de erro
                    value = -value # Inverte o sinal do valor da métrica

                row.append(value) # Adiciona o valor (potencialmente invertido) à linha atual

            z.append(row) # Adiciona a linha completa de métricas do modelo à matriz Z

        # Cria o heatmap
        fig = go.Figure( # Cria um novo objeto de figura Plotly
            data=go.Heatmap( # Adiciona um traço do tipo Heatmap à figura
                z=z, # Define os dados da matriz Z (valores do heatmap)
                x=metric_names, # Define os rótulos do eixo X (nomes das métricas)
                y=models, # Define os rótulos do eixo Y (nomes dos modelos)
                colorscale='Viridis', # Define a escala de cores para o heatmap
                colorbar=dict(title="Valor"), # Configura a barra de cores com o título "Valor"
                hoverongaps=False, # Impede que o hoverbox apareça em células sem dados
            )
        )

        # Configura o layout
        fig.update_layout( # Atualiza as configurações de layout da figura
            title="Comparação de Todas as Métricas", # Define o título principal do gráfico
            template="plotly_dark", # Aplica o tema escuro do Plotly
            height=max(420, 60 * len(models)), # Define a altura do gráfico, ajustando dinamicamente com base no número de modelos
            margin=dict(l=40, r=20, t=60, b=60), # Define as margens do gráfico
        )

        return fig # Retorna o objeto de figura Plotly configurado

    def _build_metrics_comparison_figure(self, selected_metric):
        """Retorna o gráfico correto conforme a opção escolhida"""
        if selected_metric == 'main': # Verifica se a métrica selecionada é 'main' (métricas principais)
            return self._build_main_metrics_figure() # Retorna a figura do gráfico de métricas principais
        if selected_metric == 'detailed': # Verifica se a métrica selecionada é 'detailed' (métricas detalhadas)
            return self._build_detailed_metrics_figure() # Retorna a figura do gráfico de métricas detalhadas
        return self._build_all_metrics_heatmap() # Se nenhuma das anteriores, retorna a figura do heatmap de todas as métricas

    # Cria o gráfico de importância das features
    def _build_feature_importance_figure(self):
        """Cria o gráfico de feature importance para os top 5 modelos"""
        # Retorna figura vazia se não houver feature importance
        if not self.feature_importance: # Verifica se o dicionário de feature importance está vazio
            return self._build_empty_figure( # Retorna uma figura Plotly vazia com uma mensagem de erro
                "Feature Importance - Top 5 Modelos", # Título da figura vazia
                "Feature importance não disponível." # Mensagem exibida na figura vazia
            )


        models_with_fi = []

        # Filtra modelos que possuem feature importance
        for model_name, metrics in self.results.items():
            # Verifica se o modelo atual possui dados de feature importance
            if model_name in self.feature_importance:
                # Adiciona o nome do modelo e sua métrica principal à lista de modelos com feature importance
                models_with_fi.append((model_name, self.get_primary_metric(metrics)))

        # Retorna figura vazia se nenhum modelo tiver feature importance
        if not models_with_fi:
            return self._build_empty_figure( # Retorna uma figura vazia se nenhum modelo tiver feature importance
                "Feature Importance - Top 5 Modelos", # Título da figura vazia
                "Nenhum modelo com feature importance encontrada." # Mensagem a ser exibida na figura vazia
            )

        models_with_fi.sort(key=lambda item: item[1], reverse=True) # Ordena os modelos com feature importance pela métrica principal (do melhor para o pior)
        top_models = [name for name, _ in models_with_fi[:5]] # Seleciona os nomes dos 5 melhores modelos para exibir

        fig = make_subplots( # Cria um objeto de subplots do Plotly para múltiplos gráficos
            rows=1, # Define 1 linha de subplots
            cols=len(top_models), # Define o número de colunas como o número de top modelos
            subplot_titles=top_models, # Define os títulos dos subplots como os nomes dos top modelos
            shared_yaxes=False # Define que os eixos Y não serão compartilhados entre os subplots
        )

        for col_idx, model_name in enumerate(top_models, start=1): # Itera sobre os top modelos, com um índice de coluna
            feature_names, importances = self._normalize_feature_importance(model_name) # Normaliza e obtém os nomes das features e suas importâncias para o modelo atual

            if not feature_names or not importances: # Verifica se há nomes de features ou importâncias válidos
                continue # Pula para o próximo modelo se não houver dados

            importances = np.asarray(importances, dtype=float) # Converte as importâncias para um array NumPy de floats
            top_idx = np.argsort(importances)[-10:] # Obtém os índices das 10 features mais importantes (com base nos valores)

            top_features = [feature_names[i] for i in top_idx] # Seleciona os nomes das 10 features mais importantes
            top_values = importances[top_idx] # Seleciona os valores das importâncias das 10 features mais importantes

            fig.add_trace( # Adiciona um traço (gráfico de barras) ao subplot atual
                go.Bar( # Define o tipo de traço como barra
                    x=top_values, # Valores das barras no eixo X (importância)
                    y=top_features, # Nomes das features no eixo Y
                    orientation='h', # Orientação horizontal das barras
                    name=model_name, # Nome do modelo para a legenda (se houvesse)
                    hovertemplate="Feature: %{y}<br>Importância: %{x:.4f}<extra></extra>", # Template do texto ao passar o mouse
                ),
                row=1, # Adiciona o traço na primeira linha
                col=col_idx # Adiciona o traço na coluna correspondente ao índice do modelo
            )


        # Configura o layout
        fig.update_layout(
            title="Feature Importance - Top 5 Modelos",
            template="plotly_dark",
            height=500,
            showlegend=False,
            margin=dict(l=40, r=20, t=60, b=40),
        )

        return fig

    # Cria o gráfico de previsões vs real
    def _build_predictions_figure(self, selected_model):
        """Cria o gráfico de previsões versus valores reais"""
        # Retorna figura vazia se nenhum modelo estiver selecionado
        if not selected_model:
            return self._build_empty_figure(
                "Visualização de Previsões",
                "Nenhum modelo selecionado."
            )

        # Retorna figura vazia se o modelo não existir
        if selected_model not in self.models:
            return self._build_empty_figure(
                "Visualização de Previsões",
                "Modelo selecionado não encontrado."
            )

        # Retorna figura vazia se os dados de teste não estiverem disponíveis
        if self.X_test is None or self.y_test is None:
            return self._build_empty_figure(
                "Visualização de Previsões",
                "X_test ou y_test não disponíveis."
            )

        # Obtém o modelo
        model = self.models[selected_model]

        # Tenta gerar previsões
        try:
            # Tenta gerar previsões usando o modelo selecionado e os dados de teste (X_test)
            y_pred = model.predict(self.X_test)
        # Captura qualquer exceção que ocorra durante o processo de previsão
        except Exception as exc:
            # Em caso de erro, retorna uma figura vazia com uma mensagem de erro
            return self._build_empty_figure(
                f"Visualização de Previsões - {selected_model}",
                f"Erro ao gerar previsões: {str(exc)}"
            )

        # Converte os rótulos reais (y_test) para um array NumPy e o "achata" (ravel) para 1D
        y_true = np.asarray(self.y_test).ravel()
        # Converte as previsões (y_pred) para um array NumPy e o "achata" (ravel) para 1D
        y_pred = np.asarray(y_pred).ravel()

        # Verifica se os arrays de valores reais ou previstos estão vazios
        if len(y_true) == 0 or len(y_pred) == 0:
            # Se estiverem vazios, retorna uma figura vazia com uma mensagem de dados insuficientes
            return self._build_empty_figure(
                f"Visualização de Previsões - {selected_model}",
                "Dados insuficientes para gerar o gráfico."
            )

        # Se for regressão, exibe dispersão real vs previsto
        # Se o tipo de problema for "Regressão", cria um gráfico de dispersão de previsões vs. real
        if self.detect_problem_type() == "Regressão":
            # Calcula os valores mínimo e máximo entre os valores reais e previstos para definir os limites do eixo
            min_val = float(min(np.min(y_true), np.min(y_pred)))
            max_val = float(max(np.max(y_true), np.max(y_pred)))

            # Cria uma nova figura Plotly
            fig = go.Figure()

            # Adiciona um traço de dispersão para os pontos de previsões
            fig.add_trace(
            go.Scatter(
                x=y_true,  # Eixo X: valores reais
                y=y_pred,  # Eixo Y: valores previstos
                mode='markers',  # Define o modo como marcadores (pontos)
                name='Previsões',  # Nome da série para a legenda
                marker=dict(size=8, opacity=0.75),  # Estilo dos marcadores: tamanho e opacidade
                hovertemplate="Valor real: %{x}<br>Valor previsto: %{y}<extra></extra>",  # Modelo do tooltip ao passar o mouse
            )
            )

            # Adiciona uma linha diagonal de 45 graus que representa o cenário ideal (previsão = real)
            fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],  # Eixo X: do mínimo ao máximo
                y=[min_val, max_val],  # Eixo Y: do mínimo ao máximo
                mode='lines',  # Define o modo como linhas
                name='Ideal',  # Nome da série para a legenda
                line=dict(color='red', dash='dash'),  # Estilo da linha: cor vermelha e tracejada
                hoverinfo='skip'  # Ignora informações ao passar o mouse para esta linha
            )
            )

            # Atualiza o layout do gráfico de dispersão
            fig.update_layout(
            title=f"Previsões vs Real - {selected_model}",  # Título do gráfico, incluindo o nome do modelo
            xaxis_title="Valor Real",  # Título do eixo X
            yaxis_title="Valor Previsto",  # Título do eixo Y
            template="plotly_dark",  # Define o tema do Plotly como escuro
            height=500,  # Altura do gráfico em pixels
            margin=dict(l=40, r=20, t=60, b=40),  # Margens do gráfico (esquerda, direita, topo, baixo)
            )

            # Retorna a figura do gráfico de dispersão
            return fig

        # Se não for regressão (ou seja, é classificação), cria uma matriz de confusão
        # Obtém todos os rótulos únicos presentes nos valores reais e previstos, ordenados
        labels = pd.unique(np.concatenate([y_true, y_pred])).tolist()
        # Calcula a matriz de confusão usando os rótulos únicos
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Cria uma nova figura Plotly para a matriz de confusão (heatmap)
        fig = go.Figure(
            data=go.Heatmap(
            z=cm,  # Dados da matriz (valores da matriz de confusão)
            x=[f"Previsto: {label}" for label in labels],  # Rótulos do eixo X (previsto)
            y=[f"Real: {label}" for label in labels],  # Rótulos do eixo Y (real)
            colorscale='Blues',  # Escala de cores para o heatmap
            text=cm,  # Texto a ser exibido dentro de cada célula (os próprios valores da matriz)
            texttemplate='%{text}',  # Formato do texto: exibe o valor literal
            textfont={"size": 11},  # Tamanho da fonte do texto nas células
            hovertemplate="Real: %{y}<br>Previsto: %{x}<br>Contagem: %{z}<extra></extra>",  # Modelo do tooltip
            )
        )

        # Atualiza o layout do gráfico da matriz de confusão
        fig.update_layout(
            title=f"Matriz de Confusão - {selected_model}",  # Título do gráfico, incluindo o nome do modelo
            template="plotly_dark",  # Define o tema do Plotly como escuro
            height=500,  # Altura do gráfico em pixels
            margin=dict(l=40, r=20, t=60, b=40),  # Margens do gráfico (esquerda, direita, topo, baixo)
        )

        # Retorna a figura da matriz de confusão
        return fig

        # Define um método para construir um link de download em base64
        def _build_download_link(self, label, link_id, mime_type, filename, content_bytes):
            """Cria um link de download em base64"""
        # Codifica o conteúdo em bytes para base64 e depois decodifica para string UTF-8
        content_b64 = base64.b64encode(content_bytes).decode("utf-8")

        # Retorna um componente HTML 'a' (âncora) configurado para download
        return html.A(
            label,  # Texto visível do link
            id=link_id,  # ID único para o link HTML
            href=f"data:{mime_type};base64,{content_b64}",  # Atributo href com o URI de dados base64
            download=filename,  # Atributo download para sugerir o nome do arquivo ao baixar
            className="btn btn-success mt-2"  # Classes CSS para estilizar o link como um botão Bootstrap
        )

    # Gera as linhas do relatório em texto
    def _generate_report_lines(self):
        """Gera o conteúdo textual do relatório"""
        # Obtém o nome e a métrica principal do melhor modelo
        best_model_name, best_metric = self._get_best_model_info()
        # Detecta o tipo de problema (classificação ou regressão)
        problem_type = self.detect_problem_type()

        # Inicializa uma lista para armazenar as linhas do relatório
        lines = [
            "Relatório de Machine Learning",  # Título principal do relatório
            f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",  # Data e hora de geração
            "",  # Linha em branco para espaçamento
            f"Tipo de Problema: {problem_type}",  # Tipo de problema detectado
            f"Total de Modelos: {len(self.results)}",  # Número total de modelos avaliados
            f"Melhor Modelo: {best_model_name or 'N/A'}",  # Nome do melhor modelo
            f"Métrica Principal do Melhor Modelo: {self._format_metric(best_metric)}",  # Métrica do melhor modelo formatada
            "",  # Linha em branco para espaçamento
            "Resumo dos Modelos:",  # Subtítulo para a seção de resumo dos modelos
        ]

        # Ordena os resultados dos modelos com base na métrica principal, em ordem decrescente (do melhor para o pior)
        sorted_results = sorted(
            self.results.items(),  # Itens do dicionário de resultados (nome do modelo, métricas)
            key=lambda item: self.get_primary_metric(item[1]),  # Função chave para ordenação: a métrica principal de cada modelo
            reverse=True  # Ordena do maior para o menor valor da métrica principal
        )

        # Itera sobre os resultados ordenados para adicionar detalhes de cada modelo ao relatório
        for idx, (model_name, metrics) in enumerate(sorted_results, start=1):
            # Adiciona uma linha com o rank, nome do modelo e sua métrica principal
            lines.append(f"{idx}. {model_name} | métrica principal = {self.get_primary_metric(metrics):.4f}")

            # Verifica se as métricas do modelo são um dicionário
            if isinstance(metrics, dict):
                # Itera sobre cada par chave-valor das métricas
                for key, value in metrics.items():
                    # Verifica se o valor da métrica é numérico e não é a matriz de confusão
                    if self._is_numeric(value) and key != "confusion_matrix": # Adiciona condição para ignorar confusion_matrix
                        # Adiciona uma linha indentada com o nome da métrica e seu valor formatado
                        lines.append(f"   - {key}: {float(value):.4f}")

        # Retorna a lista de linhas que compõem o relatório
        return lines

    # Escapa caracteres especiais para PDF
    def _escape_pdf_text(self, text):
        """Escapa texto para stream PDF"""
        return (
            str(text)
            .replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )

    # Gera um PDF simples sem dependências externas
    def _generate_simple_pdf(self, lines):
        """Gera um PDF simples e válido usando apenas bibliotecas padrão"""
        # Escapa caracteres especiais em cada linha e limita a 60 linhas para evitar PDFs muito grandes
        safe_lines = [self._escape_pdf_text(line) for line in lines[:60]]

        # Lista para armazenar as partes do conteúdo do stream de texto do PDF
        content_parts = [
            "BT",          # Inicia o bloco de texto (Begin Text)
            "/F1 11 Tf",   # Define a fonte (F1) e o tamanho (11 pontos)
            "50 760 Td",   # Define a posição inicial do texto (x=50, y=760)
            "14 TL",       # Define o espaçamento entre as linhas (leading) para 14 unidades
        ]

        # Adiciona cada linha de texto ao stream
        for idx, line in enumerate(safe_lines):
            if idx > 0:
                content_parts.append("T*")  # Avança para a próxima linha (Next Line)
            content_parts.append(f"({line}) Tj")  # Adiciona o texto atual (Show Text)

        content_parts.append("ET")  # Encerra o bloco de texto (End Text)
        # Junta todas as partes do conteúdo em uma única string e a codifica para bytes (latin-1, tratando erros)
        stream = "\n".join(content_parts).encode("latin-1", errors="replace")

        # Dicionário que mapeia números de objeto PDF para seus conteúdos em bytes
        objects = {
            1: b"<< /Type /Catalog /Pages 2 0 R >>", # Objeto de Catálogo raiz do PDF
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>", # Objeto de Páginas, referenciando a página 3
            3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>", # Objeto da Página, definindo mídia, conteúdo (4) e recursos (fonte F1)
            4: b"<< /Length " + str(len(stream)).encode("latin-1") + b" >>\nstream\n" + stream + b"\nendstream", # Objeto de Conteúdo da Página, contendo o stream de texto
            5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", # Objeto de Fonte, definindo Helvetica como fonte
        }

        pdf = b"%PDF-1.4\n" # Cabeçalho do arquivo PDF, indicando a versão
        offsets = {0: 0} # Dicionário para armazenar os offsets de byte de cada objeto (offset 0 para o objeto nulo)

        # Adiciona cada objeto ao corpo do PDF e registra seus offsets
        for obj_num in range(1, 6):
            offsets[obj_num] = len(pdf) # Salva o offset atual antes de adicionar o objeto
            pdf += f"{obj_num} 0 obj\n".encode("latin-1") # Escreve o cabeçalho do objeto (ex: "1 0 obj")
            pdf += objects[obj_num] + b"\n" # Adiciona o conteúdo do objeto
            pdf += b"endobj\n" # Encerra o objeto

        xref_offset = len(pdf) # Salva o offset da tabela de referência cruzada (xref)
        pdf += b"xref\n" # Inicia a tabela xref
        pdf += b"0 6\n" # Indica que a tabela xref começa no objeto 0 e tem 6 entradas (0 a 5)
        pdf += b"0000000000 65535 f \n" # Entrada para o objeto 0 (nulo), 'f' indica que está livre

        # Adiciona as entradas para cada objeto na tabela xref
        for obj_num in range(1, 6):
            # Formata o offset do objeto para 10 dígitos com zeros à esquerda e '00000 n' (n indica que é um objeto usado)
            pdf += f"{offsets[obj_num]:010d} 00000 n \n".encode("latin-1")

        pdf += b"trailer\n" # Inicia o trailer do PDF
        pdf += b"<< /Size 6 /Root 1 0 R >>\n" # Dicionário do trailer, indicando o número total de objetos e o objeto raiz (Catálogo)
        pdf += b"startxref\n" # Indica onde a tabela xref começa
        pdf += f"{xref_offset}\n".encode("latin-1") # Adiciona o offset da xref salvo anteriormente
        pdf += b"%%EOF" # Marca o final do arquivo PDF

        return pdf # Retorna o arquivo PDF como bytes

    # Serializa o melhor modelo para download
    def _serialize_best_model(self):
        """Serializa o melhor modelo usando pickle"""
        best_model_name, _ = self._get_best_model_info()

        # Usa o melhor modelo, se houver
        if best_model_name and best_model_name in self.models:
            model_to_save = self.models[best_model_name]
        # Caso não haja melhor modelo, usa o primeiro disponível
        elif self.models:
            best_model_name = self._default_model_name()
            model_to_save = self.models[best_model_name]
        else:
            return None, "Nenhum modelo disponível para exportação."

        # Tenta serializar o modelo
        try:
            # 'protocol=pickle.HIGHEST_PROTOCOL' garante que a versão mais recente e eficiente do protocolo de pickle seja utilizada.
            payload = pickle.dumps(model_to_save, protocol=pickle.HIGHEST_PROTOCOL)
            # Retorna a sequência de bytes serializada (payload) e uma mensagem de sucesso formatada.
            return payload, f"Modelo '{best_model_name}' serializado com sucesso."
        # Captura qualquer exceção que possa ocorrer durante o processo de serialização.
        except Exception as exc:
            # Em caso de erro, retorna None como payload e uma mensagem de erro detalhada.
            return None, f"Falha ao serializar o modelo: {str(exc)}"

    # Método para configurar o layout do dashboard
    def setup_layout(self):
        """Configura o layout do dashboard"""
        # Obtém informações do melhor modelo
        best_model_name, best_metric = self._get_best_model_info()
        # Detecta o tipo de problema
        problem_type = self.detect_problem_type()

        # Define o layout principal como um container fluido
        # Define o layout principal da aplicação Dash como um container fluido (que ocupa toda a largura disponível)
        self.app.layout = dbc.Container([
            # Inicia uma linha para organizar os componentes horizontalmente
            dbc.Row([
            # Inicia uma coluna que ocupa toda a largura (12 de 12 unidades)
            dbc.Col([
                # Adiciona um título principal ao dashboard, centralizado e com margem inferior
                html.H1("Dashboard de Machine Learning Avançado",
                    className="text-center mb-4"),
                # Adiciona uma linha horizontal para separação visual
                html.Hr(),
            ], width=12)  # A coluna ocupa 12 unidades de largura
            ], className="mb-4"),  # Adiciona margem inferior à linha

            # Inicia uma nova linha para o resumo do projeto
            dbc.Row([
            # Inicia uma coluna para o card de resumo (ocupa 4 de 12 unidades)
            dbc.Col([
                # Cria um componente Card do Bootstrap
                dbc.Card([
                # Define o cabeçalho do card
                dbc.CardHeader("📊 Resumo do Projeto"),
                # Define o corpo do card
                dbc.CardBody([
                    # Exibe o número total de modelos treinados
                    html.P(f"Total de Modelos Treinados: {len(self.results)}"),
                    # Exibe o nome do melhor modelo (ou 'N/A' se não houver)
                    html.P(f"Melhor Modelo: {best_model_name or 'N/A'}"),
                    # Exibe o valor formatado da métrica do melhor modelo
                    html.P(f"Métrica do Melhor Modelo: {self._format_metric(best_metric)}"),
                ])
                ], className="mb-4")  # Adiciona margem inferior ao card
            ], width=4),  # A coluna ocupa 4 unidades de largura

            # Inicia uma coluna para o card do tipo de problema (ocupa 4 de 12 unidades)
            dbc.Col([
                # Cria um componente Card do Bootstrap
                dbc.Card([
                # Define o cabeçalho do card
                dbc.CardHeader("🎯 Tipo de Problema"),
                # Define o corpo do card
                dbc.CardBody([
                    # Exibe o tipo de problema detectado (Classificação/Regressão), centralizado
                    html.H3(problem_type, id="problem-type",
                        className="text-center"),
                    # Adiciona uma descrição auxiliar, centralizada e em texto "muted"
                    html.P("Classificação/Regressão detectada automaticamente",
                       className="text-muted text-center")
                ])
                ], className="mb-4")  # Adiciona margem inferior ao card
            ], width=4),  # A coluna ocupa 4 unidades de largura

            # Inicia uma coluna para o card de estatísticas (ocupa 4 de 12 unidades)
            dbc.Col([
                # Cria um componente Card do Bootstrap
                dbc.Card([
                # Define o cabeçalho do card
                dbc.CardHeader("📈 Estatísticas"),
                # Define o corpo do card
                dbc.CardBody([
                    # Exibe o número de modelos otimizados (detectados pelo nome)
                    html.P(
                    f"Modelos Otimizados: {len([m for m in self.models.keys() if 'optimized' in m.lower() or 'otimizado' in m.lower()])}"
                    ),
                    # Informa se há modelos de ensemble (detectados pelo nome)
                    html.P(
                    f"Inclui Ensemble: {'Sim' if any('ensemble' in m.lower() for m in self.models.keys()) else 'Não'}"
                    ),
                    # Exibe o status geral (se há resultados ou não)
                    html.P(
                    f"Status: {'✅ Completo' if self.results else '⚠️ Sem resultados'}"
                    )
                ])
                ], className="mb-4")  # Adiciona margem inferior ao card
            ], width=4)  # A coluna ocupa 4 unidades de largura
            ], className="mb-4"),  # Adiciona margem inferior à linha

            # Inicia uma nova linha para o gráfico de ranking dos modelos
            dbc.Row([
            # Inicia uma coluna que ocupa toda a largura (12 de 12 unidades)
            dbc.Col([
                # Cria um componente Card do Bootstrap
                dbc.Card([
                # Define o cabeçalho do card
                dbc.CardHeader("🏆 Ranking dos Modelos (do melhor para o pior)"),
                # Define o corpo do card
                dbc.CardBody([
                    # Adiciona um componente gráfico (Plotly)
                    dcc.Graph(
                    id='ranking-plot',  # ID único para o gráfico
                    figure=self._build_ranking_figure()  # Define a figura inicial chamando um método interno
                    )
                ])
                ])
            ], width=12)  # A coluna ocupa 12 unidades de largura
            ], className="mb-4"),  # Adiciona margem inferior à linha

            # Inicia uma nova linha para os gráficos de comparação de métricas
            dbc.Row([
            # Inicia uma coluna que ocupa toda a largura (12 de 12 unidades)
            dbc.Col([
                # Cria um componente Card do Bootstrap
                dbc.Card([
                # Define o cabeçalho do card
                dbc.CardHeader("📊 Comparação de Métricas"),
                # Define o corpo do card
                dbc.CardBody([
                    # Adiciona um dropdown para selecionar o tipo de comparação de métricas
                    dcc.Dropdown(
                    id='metric-selector',  # ID único para o dropdown
                    options=[  # Define as opções do dropdown
                        {'label': 'Todas as Métricas', 'value': 'all'},
                        {'label': 'Acurácia/F1/R2', 'value': 'main'},
                        {'label': 'Métricas Detalhadas', 'value': 'detailed'}
                    ],
                    value='main',  # Define a opção padrão selecionada
                    className="mb-3"  # Adiciona margem inferior ao dropdown
                    ),
                    # Adiciona um componente gráfico para exibir a comparação de métricas
                    dcc.Graph(
                    id='metrics-comparison',  # ID único para o gráfico
                    figure=self._build_metrics_comparison_figure('main')  # Define a figura inicial com a opção padrão
                    )
                ])
                ])
            ], width=12)  # A coluna ocupa 12 unidades de largura
            ], className="mb-4"),  # Adiciona margem inferior à linha

            # Inicia uma nova linha para o gráfico de importância das features
            dbc.Row([
            # Inicia uma coluna que ocupa toda a largura (12 de 12 unidades)
            dbc.Col([
                # Cria um componente Card do Bootstrap
                dbc.Card([
                # Define o cabeçalho do card
                dbc.CardHeader("🔍 Feature Importance - Top 5 Modelos"),
                # Define o corpo do card
                dbc.CardBody([
                    # Adiciona um componente gráfico para exibir a importância das features
                    dcc.Graph(
                    id='feature-importance-plot',  # ID único para o gráfico
                    figure=self._build_feature_importance_figure()  # Define a figura inicial
                    )
                ])
                ])
            ], width=12)  # A coluna ocupa 12 unidades de largura
            ], className="mb-4"),  # Adiciona margem inferior à linha

            # Inicia uma nova linha para a visualização de previsões
            dbc.Row([
            # Inicia uma coluna que ocupa toda a largura (12 de 12 unidades)
            dbc.Col([
                # Cria um componente Card do Bootstrap
                dbc.Card([
                # Define o cabeçalho do card
                dbc.CardHeader("🔮 Visualização de Previsões vs Real"),
                # Define o corpo do card
                dbc.CardBody([
                    # Adiciona um dropdown para selecionar o modelo a ser visualizado
                    dcc.Dropdown(
                    id='model-selector',  # ID único para o dropdown
                    options=[{'label': m, 'value': m}  # Gera opções a partir dos nomes dos modelos
                         for m in self.models.keys()],
                    value=self._default_model_name(),  # Define o modelo padrão selecionado
                    className="mb-3"  # Adiciona margem inferior ao dropdown
                    ),
                    # Adiciona um componente gráfico para exibir as previsões
                    dcc.Graph(
                    id='predictions-plot',  # ID único para o gráfico
                    # Define a figura inicial com o modelo padrão
                    figure=self._build_predictions_figure(self._default_model_name())
                    )
                ])
                ])
            ], width=12)  # A coluna ocupa 12 unidades de largura
            ], className="mb-4"),  # Adiciona margem inferior à linha

            # Inicia uma nova linha para a seção de download de relatórios e exportação
            dbc.Row([
            # Inicia uma coluna que ocupa toda a largura (12 de 12 unidades)
            dbc.Col([
                # Cria um componente Card do Bootstrap
                dbc.Card([
                # Define o cabeçalho do card
                dbc.CardHeader("📥 Relatório e Exportação"),
                # Define o corpo do card
                dbc.CardBody([
                    # Cria um contêiner flexível para os botões de download, centralizados
                    html.Div([
                    # Botão para gerar relatório PDF
                    dbc.Button("📄 Gerar Relatório PDF",
                           id="generate-pdf",  # ID único para o botão
                           color="primary",  # Cor primária do Bootstrap
                           className="me-2"),  # Margem à direita
                    # Botão para exportar resultados em CSV
                    dbc.Button("💾 Exportar Resultados CSV",
                           id="export-csv",  # ID único para o botão
                           color="success",  # Cor de sucesso do Bootstrap
                           className="me-2"),  # Margem à direita
                    # Botão para salvar o melhor modelo
                    dbc.Button("🤖 Salvar Melhor Modelo",
                           id="save-model",  # ID único para o botão
                           color="warning"),  # Cor de aviso do Bootstrap
                    ], className="d-flex justify-content-center"),  # Classes para layout flexível e centralização

                    # Div oculta para armazenar o link de download do PDF (será populado por callback)
                    html.Div(id='pdf-download', style={'display': 'none'}),
                    # Div oculta para armazenar o link de download do CSV (será populado por callback)
                    html.Div(id='csv-download', style={'display': 'none'}),
                    # Div oculta para armazenar o link de download do modelo (será populado por callback)
                    html.Div(id='model-download', style={'display': 'none'}),

                    # Barra de progresso para indicar o status das operações
                    dbc.Progress(id="progress-bar", value=0,  # ID e valor inicial (0%)
                         striped=True, animated=True,  # Estilos visuais
                         className="mt-3"),  # Margem superior

                    # Div para exibir mensagens de status ao usuário
                    html.Div(id="status-message",
                         className="mt-2 text-center")  # Margem superior e centralização
                ])
                ])
            ], width=12)  # A coluna ocupa 12 unidades de largura
            ], className="mb-4"),  # Adiciona margem inferior à linha

            # Inicia uma nova linha para a tabela detalhada de resultados
            dbc.Row([
            # Inicia uma coluna que ocupa toda a largura (12 de 12 unidades)
            dbc.Col([
                # Cria um componente Card do Bootstrap
                dbc.Card([
                # Define o cabeçalho do card
                dbc.CardHeader("📋 Tabela Detalhada de Resultados"),
                # Define o corpo do card
                dbc.CardBody([
                    # Div que conterá a tabela de resultados (será construída por um método interno)
                    html.Div(
                    id='results-table',  # ID único para a div
                    children=self._build_results_table()  # Conteúdo inicial da tabela
                    )
                ])
                ])
            ], width=12)  # A coluna ocupa 12 unidades de largura
            ]) # Nenhuma margem inferior para a última linha do layout
        ], fluid=True) # O container ocupa toda a largura disponível (fluido)

    # Método para configurar os callbacks do dashboard
    def setup_callbacks(self):
        """Configura os callbacks do dashboard"""

        # Callback para atualizar o gráfico de comparação de métricas
        @self.app.callback(  # Decorador que registra a função 'update_metrics_comparison' como um callback Dash
            Output('metrics-comparison', 'figure'),  # Define o componente de saída: a propriedade 'figure' do gráfico com ID 'metrics-comparison'
            Input('metric-selector', 'value')  # Define o componente de entrada: a propriedade 'value' do dropdown com ID 'metric-selector'
        )
        def update_metrics_comparison(selected_metric):  # Define a função de callback que será executada quando o valor do 'metric-selector' mudar
            return self._build_metrics_comparison_figure(selected_metric)  # Chama um método interno para construir o gráfico de comparação de métricas com base na opção selecionada e retorna a figura

        # Callback para atualizar o gráfico de previsões
        @self.app.callback(  # Decorador que registra a função 'update_predictions_plot' como um callback Dash
            Output('predictions-plot', 'figure'),  # Define o componente de saída: a propriedade 'figure' do gráfico com ID 'predictions-plot'
            Input('model-selector', 'value')  # Define o componente de entrada: a propriedade 'value' do dropdown com ID 'model-selector'
        )
        def update_predictions_plot(selected_model):  # Define a função de callback que será executada quando o valor do 'model-selector' mudar
            # A função recebe o 'value' do 'model-selector' como argumento 'selected_model'
            return self._build_predictions_figure(selected_model)  # Chama um método interno para construir o gráfico de previsões com base no modelo selecionado e retorna a figura

        # Registra um callback para o aplicativo Dash
        @self.app.callback(
            # Define as saídas que serão atualizadas por este callback
            [Output('results-table', 'children'),    # O conteúdo da tabela de resultados
             Output('pdf-download', 'children'),     # O link de download para PDF
             Output('csv-download', 'children'),     # O link de download para CSV
             Output('model-download', 'children'),   # O link de download para o modelo
             Output('progress-bar', 'value'),        # O valor da barra de progresso
             Output('status-message', 'children')],  # A mensagem de status exibida ao usuário
            # Define as entradas que acionarão este callback
            [Input('generate-pdf', 'n_clicks'),      # O número de cliques no botão 'Gerar Relatório PDF'
             Input('export-csv', 'n_clicks'),        # O número de cliques no botão 'Exportar Resultados CSV'
             Input('save-model', 'n_clicks')],       # O número de cliques no botão 'Salvar Melhor Modelo'
            # Impede que o callback seja acionado na inicialização da aplicação
            prevent_initial_call=True
        )
        def handle_downloads(pdf_clicks, csv_clicks, model_clicks):
            # Obtém o contexto do callback para identificar o botão clicado
            ctx = dash.callback_context

            # Define a resposta padrão sem atualização
            default_response = (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

            # Retorna sem atualização se nada foi acionado
            if not ctx.triggered:
                return default_response

            # Identifica o botão clicado
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Gera o relatório em PDF
            if button_id == 'generate-pdf':  # Verifica se o botão 'generate-pdf' foi clicado
                try:  # Inicia um bloco try-except para tratamento de erros durante a geração do PDF
                    report_lines = self._generate_report_lines()  # Gera as linhas de texto para o relatório
                    pdf_bytes = self._generate_simple_pdf(report_lines)  # Gera o conteúdo do PDF em bytes a partir das linhas do relatório

                    download_link = self._build_download_link(  # Cria um link de download para o arquivo PDF
                        label="📥 Baixar Relatório PDF",  # Texto exibido no botão de download
                        link_id="pdf-download-link",  # ID único para o link de download
                        mime_type="application/pdf",  # Tipo MIME para arquivo PDF
                        filename="relatorio_ml.pdf",  # Nome do arquivo a ser baixado
                        content_bytes=pdf_bytes  # Conteúdo do arquivo em bytes (o PDF)
                    )

                    return (  # Retorna para o callback com o link de download e uma mensagem de sucesso
                        dash.no_update,  # Não atualiza a tabela de resultados
                        download_link,  # Atualiza o link de download de PDF
                        dash.no_update,  # Não atualiza o link de download de CSV
                        dash.no_update,  # Não atualiza o link de download do modelo
                        100,  # Define o valor da barra de progresso como 100 (concluído)
                        "✅ PDF gerado com sucesso!"  # Exibe a mensagem de sucesso
                    )
                except Exception as exc:  # Captura qualquer exceção que ocorra durante o processo
                    return (  # Retorna para o callback com uma mensagem de erro
                        dash.no_update,  # Não atualiza a tabela de resultados
                        dash.no_update,  # Não atualiza o link de download de PDF
                        dash.no_update,  # Não atualiza o link de download de CSV
                        dash.no_update,  # Não atualiza o link de download do modelo
                        0,  # Define o valor da barra de progresso como 0
                        f"❌ Erro ao gerar PDF: {str(exc)}"  # Exibe a mensagem de erro formatada
                    )

            # Exporta os resultados em CSV
            if button_id == 'export-csv':  # Verifica se o botão 'export-csv' foi clicado
                try:  # Inicia um bloco try-except para tratamento de erros durante a exportação
                    rows = []  # Inicializa uma lista vazia para armazenar as linhas de dados do CSV

                    # Ordena os resultados dos modelos com base na métrica principal, do melhor para o pior
                    sorted_results = sorted(
                        self.results.items(),  # Itera sobre os itens do dicionário de resultados
                        key=lambda item: self.get_primary_metric(item[1]),  # Usa a métrica principal para ordenação
                        reverse=True  # Ordena em ordem decrescente (melhor primeiro)
                    )

                    # Itera sobre os resultados ordenados para construir cada linha do CSV
                    for rank, (model_name, metrics) in enumerate(sorted_results, start=1):
                        row = {  # Cria um dicionário para a linha atual
                            "rank": rank,  # Adiciona a posição no ranking
                            "model": model_name,  # Adiciona o nome do modelo
                            "primary_metric": self.get_primary_metric(metrics),  # Adiciona a métrica principal
                        }

                        if isinstance(metrics, dict):  # Verifica se as métricas são um dicionário
                            for key, value in metrics.items():  # Itera sobre cada métrica do modelo
                                if self._is_numeric(value):  # Verifica se o valor da métrica é numérico
                                    row[key] = float(value)  # Adiciona a métrica numérica à linha

                        rows.append(row)  # Adiciona a linha (dicionário) à lista de linhas

                    results_df = pd.DataFrame(rows)  # Converte a lista de dicionários em um DataFrame do pandas
                    # Converte o DataFrame para CSV como string e depois para bytes, ignorando o índice
                    csv_bytes = results_df.to_csv(index=False).encode("utf-8")

                    # Cria um link de download para o arquivo CSV
                    download_link = self._build_download_link(
                        label="💾 Baixar CSV",  # Texto exibido no botão de download
                        link_id="csv-download-link",  # ID único para o link de download
                        mime_type="text/csv",  # Tipo MIME para arquivo CSV
                        filename="resultados_ml.csv",  # Nome do arquivo a ser baixado
                        content_bytes=csv_bytes  # Conteúdo do arquivo em bytes (o CSV)
                    )

                    return (  # Retorna para o callback com o link de download e uma mensagem de sucesso
                        dash.no_update,  # Não atualiza a tabela de resultados
                        dash.no_update,  # Não atualiza o link de download de PDF
                        download_link,  # Atualiza o link de download de CSV
                        dash.no_update,  # Não atualiza o link de download do modelo
                        100,  # Define o valor da barra de progresso como 100 (concluído)
                        "✅ CSV exportado com sucesso!"  # Exibe a mensagem de sucesso
                    )
                except Exception as exc:  # Captura qualquer exceção que ocorra durante o processo
                    return (  # Retorna para o callback com uma mensagem de erro
                        dash.no_update,  # Não atualiza a tabela de resultados
                        dash.no_update,  # Não atualiza o link de download de PDF
                        dash.no_update,  # Não atualiza o link de download de CSV
                        dash.no_update,  # Não atualiza o link de download do modelo
                        0,  # Define o valor da barra de progresso como 0
                        f"❌ Erro ao exportar CSV: {str(exc)}"  # Exibe a mensagem de erro formatada
                    )

            # Serializa e exporta o melhor modelo
            if button_id == 'save-model':  # Verifica se o botão 'save-model' foi clicado
                model_bytes, message = self._serialize_best_model()  # Tenta serializar o melhor modelo e obtém os bytes e uma mensagem de status

                if model_bytes is None:  # Se a serialização falhou (model_bytes é None)
                    return (  # Retorna para o callback com uma mensagem de erro
                        dash.no_update,  # Não atualiza a tabela de resultados
                        dash.no_update,  # Não atualiza o link de download de PDF
                        dash.no_update,  # Não atualiza o link de download de CSV
                        dash.no_update,  # Não atualiza o link de download do modelo
                        0,  # Define o valor da barra de progresso como 0
                        f"❌ {message}"  # Exibe a mensagem de erro formatada
                    )

                download_link = self._build_download_link(  # Se a serialização foi bem-sucedida, cria um link de download
                    label="🤖 Baixar Modelo",  # Texto exibido no botão de download
                    link_id="model-download-link",  # ID único para o link de download
                    mime_type="application/octet-stream",  # Tipo MIME para arquivo binário genérico
                    filename="melhor_modelo.pkl",  # Nome do arquivo a ser baixado
                    content_bytes=model_bytes  # Conteúdo do arquivo em bytes (o modelo serializado)
                )

                return (  # Retorna para o callback com o link de download e uma mensagem de sucesso
                    dash.no_update,  # Não atualiza a tabela de resultados
                    dash.no_update,  # Não atualiza o link de download de PDF
                    dash.no_update,  # Não atualiza o link de download de CSV
                    download_link,  # Atualiza o link de download do modelo
                    100,  # Define o valor da barra de progresso como 100 (concluído)
                    f"✅ {message}"  # Exibe a mensagem de sucesso formatada
                )

            # Retorna padrão se nenhuma ação válida for executada
            return default_response

    # Método para executar o dashboard
    def run(self, port=8050):
        """Executa o dashboard"""
        # Exibe a URL do dashboard no console
        print(f"Dashboard rodando em http://localhost:{port}")
        # Inicia o servidor do Dash
        self.app.run(debug=True, port=port)

