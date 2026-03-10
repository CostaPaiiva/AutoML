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
            if self._is_numeric(metrics.get("r2")):
                return float(metrics["r2"])

            for key in ("rmse", "mae", "mse"):
                value = metrics.get(key)
                if self._is_numeric(value):
                    return -float(value)

        # Retorna 0 se nenhuma métrica principal for encontrada
        return 0.0

    # Obtém informações do melhor modelo
    def _get_best_model_info(self):
        """Retorna o nome e a métrica principal do melhor modelo"""
        # Retorna valores nulos se não houver resultados
        if not self.results:
            return None, None

        # Seleciona o melhor modelo com base na métrica principal
        best_name, best_metrics = max(
            self.results.items(),
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

        def sort_key(name):
            if name in priority:
                return (0, priority.index(name))
            return (1, name)

        return sorted(metric_names, key=sort_key)

    # Obtém a lista de métricas numéricas disponíveis
    def _get_numeric_metric_names(self):
        """Retorna as métricas numéricas encontradas nos resultados"""
        metric_names = set()

        for metrics in self.results.values():
            if not isinstance(metrics, dict):
                continue

            for key, value in metrics.items():
                if key == "confusion_matrix":
                    continue
                if self._is_numeric(value):
                    metric_names.add(key)

        return self._ordered_metric_names(metric_names)

    # Formata métricas para exibição
    def _format_metric(self, value):
        """Formata o valor da métrica para exibição"""
        if value is None:
            return "N/A"
        if self._is_numeric(value):
            return f"{float(value):.4f}"
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
            columns = list(map(str, self.X_test.columns))
            if len(columns) == len(values):
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

        # Ordena os modelos pela métrica principal
        sorted_results = sorted(
            self.results.items(),
            key=lambda item: self.get_primary_metric(item[1]),
            reverse=True
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
                for key, value in metrics.items():
                    if self._is_numeric(value):
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
        sorted_results = sorted(
            self.results.items(),
            key=lambda item: self.get_primary_metric(item[1]),
            reverse=True
        )

        # Extrai nomes e pontuações
        model_names = [item[0] for item in sorted_results]
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
                value = self.results[model_name].get(metric_key, 0)
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
                value = self.results[model_name].get(metric, 0)
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
        metric_names = self._get_numeric_metric_names()

        # Retorna figura vazia se não houver métricas
        if not metric_names:
            return self._build_empty_figure(
                "Comparação de Todas as Métricas",
                "Nenhuma métrica numérica disponível."
            )

        models = list(self.results.keys())
        z = []

        # Monta a matriz do heatmap
        for model_name in models:
            row = []

            for metric in metric_names:
                value = self.results[model_name].get(metric, 0)
                value = float(value) if self._is_numeric(value) else 0.0

                # Inverte métricas de erro
                if metric in {"rmse", "mae", "mse"}:
                    value = -value

                row.append(value)

            z.append(row)

        # Cria o heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=metric_names,
                y=models,
                colorscale='Viridis',
                colorbar=dict(title="Valor"),
                hoverongaps=False,
            )
        )

        # Configura o layout
        fig.update_layout(
            title="Comparação de Todas as Métricas",
            template="plotly_dark",
            height=max(420, 60 * len(models)),
            margin=dict(l=40, r=20, t=60, b=60),
        )

        return fig

    # Escolhe qual gráfico de comparação de métricas deve ser exibido
    def _build_metrics_comparison_figure(self, selected_metric):
        """Retorna o gráfico correto conforme a opção escolhida"""
        if selected_metric == 'main':
            return self._build_main_metrics_figure()
        if selected_metric == 'detailed':
            return self._build_detailed_metrics_figure()
        return self._build_all_metrics_heatmap()

    # Cria o gráfico de importância das features
    def _build_feature_importance_figure(self):
        """Cria o gráfico de feature importance para os top 5 modelos"""
        # Retorna figura vazia se não houver feature importance
        if not self.feature_importance:
            return self._build_empty_figure(
                "Feature Importance - Top 5 Modelos",
                "Feature importance não disponível."
            )

        models_with_fi = []

        # Filtra modelos que possuem feature importance
        for model_name, metrics in self.results.items():
            if model_name in self.feature_importance:
                models_with_fi.append((model_name, self.get_primary_metric(metrics)))

        # Retorna figura vazia se nenhum modelo tiver feature importance
        if not models_with_fi:
            return self._build_empty_figure(
                "Feature Importance - Top 5 Modelos",
                "Nenhum modelo com feature importance encontrada."
            )

        # Ordena os modelos pela métrica principal
        models_with_fi.sort(key=lambda item: item[1], reverse=True)
        top_models = [name for name, _ in models_with_fi[:5]]

        # Cria subplots para os top modelos
        fig = make_subplots(
            rows=1,
            cols=len(top_models),
            subplot_titles=top_models,
            shared_yaxes=False
        )

        # Adiciona as barras de importância para cada modelo
        for col_idx, model_name in enumerate(top_models, start=1):
            feature_names, importances = self._normalize_feature_importance(model_name)

            if not feature_names or not importances:
                continue

            importances = np.asarray(importances, dtype=float)
            top_idx = np.argsort(importances)[-10:]

            top_features = [feature_names[i] for i in top_idx]
            top_values = importances[top_idx]

            fig.add_trace(
                go.Bar(
                    x=top_values,
                    y=top_features,
                    orientation='h',
                    name=model_name,
                    hovertemplate="Feature: %{y}<br>Importância: %{x:.4f}<extra></extra>",
                ),
                row=1,
                col=col_idx
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
            y_pred = model.predict(self.X_test)
        except Exception as exc:
            return self._build_empty_figure(
                f"Visualização de Previsões - {selected_model}",
                f"Erro ao gerar previsões: {str(exc)}"
            )

        y_true = np.asarray(self.y_test).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # Retorna figura vazia se os dados estiverem vazios
        if len(y_true) == 0 or len(y_pred) == 0:
            return self._build_empty_figure(
                f"Visualização de Previsões - {selected_model}",
                "Dados insuficientes para gerar o gráfico."
            )

        # Se for regressão, exibe dispersão real vs previsto
        if self.detect_problem_type() == "Regressão":
            min_val = float(min(np.min(y_true), np.min(y_pred)))
            max_val = float(max(np.max(y_true), np.max(y_pred)))

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=y_true,
                    y=y_pred,
                    mode='markers',
                    name='Previsões',
                    marker=dict(size=8, opacity=0.75),
                    hovertemplate="Valor real: %{x}<br>Valor previsto: %{y}<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Ideal',
                    line=dict(color='red', dash='dash'),
                    hoverinfo='skip'
                )
            )

            fig.update_layout(
                title=f"Previsões vs Real - {selected_model}",
                xaxis_title="Valor Real",
                yaxis_title="Valor Previsto",
                template="plotly_dark",
                height=500,
                margin=dict(l=40, r=20, t=60, b=40),
            )

            return fig

        # Se for classificação, exibe matriz de confusão
        labels = pd.unique(np.concatenate([y_true, y_pred])).tolist()
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Previsto: {label}" for label in labels],
                y=[f"Real: {label}" for label in labels],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 11},
                hovertemplate="Real: %{y}<br>Previsto: %{x}<br>Contagem: %{z}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Matriz de Confusão - {selected_model}",
            template="plotly_dark",
            height=500,
            margin=dict(l=40, r=20, t=60, b=40),
        )

        return fig

    # Cria um link de download a partir de bytes
    def _build_download_link(self, label, link_id, mime_type, filename, content_bytes):
        """Cria um link de download em base64"""
        content_b64 = base64.b64encode(content_bytes).decode("utf-8")

        return html.A(
            label,
            id=link_id,
            href=f"data:{mime_type};base64,{content_b64}",
            download=filename,
            className="btn btn-success mt-2"
        )

    # Gera as linhas do relatório em texto
    def _generate_report_lines(self):
        """Gera o conteúdo textual do relatório"""
        best_model_name, best_metric = self._get_best_model_info()
        problem_type = self.detect_problem_type()

        lines = [
            "Relatório de Machine Learning",
            f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Tipo de Problema: {problem_type}",
            f"Total de Modelos: {len(self.results)}",
            f"Melhor Modelo: {best_model_name or 'N/A'}",
            f"Métrica Principal do Melhor Modelo: {self._format_metric(best_metric)}",
            "",
            "Resumo dos Modelos:",
        ]

        sorted_results = sorted(
            self.results.items(),
            key=lambda item: self.get_primary_metric(item[1]),
            reverse=True
        )

        for idx, (model_name, metrics) in enumerate(sorted_results, start=1):
            lines.append(f"{idx}. {model_name} | métrica principal = {self.get_primary_metric(metrics):.4f}")

            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if self._is_numeric(value):
                        lines.append(f"   - {key}: {float(value):.4f}")

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
        safe_lines = [self._escape_pdf_text(line) for line in lines[:60]]

        content_parts = [
            "BT",
            "/F1 11 Tf",
            "50 760 Td",
            "14 TL",
        ]

        for idx, line in enumerate(safe_lines):
            if idx > 0:
                content_parts.append("T*")
            content_parts.append(f"({line}) Tj")

        content_parts.append("ET")
        stream = "\n".join(content_parts).encode("latin-1", errors="replace")

        objects = {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            4: b"<< /Length " + str(len(stream)).encode("latin-1") + b" >>\nstream\n" + stream + b"\nendstream",
            5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        }

        pdf = b"%PDF-1.4\n"
        offsets = {0: 0}

        for obj_num in range(1, 6):
            offsets[obj_num] = len(pdf)
            pdf += f"{obj_num} 0 obj\n".encode("latin-1")
            pdf += objects[obj_num] + b"\n"
            pdf += b"endobj\n"

        xref_offset = len(pdf)
        pdf += b"xref\n"
        pdf += b"0 6\n"
        pdf += b"0000000000 65535 f \n"

        for obj_num in range(1, 6):
            pdf += f"{offsets[obj_num]:010d} 00000 n \n".encode("latin-1")

        pdf += b"trailer\n"
        pdf += b"<< /Size 6 /Root 1 0 R >>\n"
        pdf += b"startxref\n"
        pdf += f"{xref_offset}\n".encode("latin-1")
        pdf += b"%%EOF"

        return pdf

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
            payload = pickle.dumps(model_to_save, protocol=pickle.HIGHEST_PROTOCOL)
            return payload, f"Modelo '{best_model_name}' serializado com sucesso."
        except Exception as exc:
            return None, f"Falha ao serializar o modelo: {str(exc)}"

    # Método para configurar o layout do dashboard
    def setup_layout(self):
        """Configura o layout do dashboard"""
        # Obtém informações do melhor modelo
        best_model_name, best_metric = self._get_best_model_info()
        # Detecta o tipo de problema
        problem_type = self.detect_problem_type()

        # Define o layout principal como um container fluido
        self.app.layout = dbc.Container([
            # Linha para o cabeçalho
            dbc.Row([
                # Coluna contendo o título do dashboard
                dbc.Col([
                    html.H1("Dashboard de Machine Learning Avançado",
                            className="text-center mb-4"),
                    html.Hr(),
                ], width=12)
            ], className="mb-4"),

            # Linha para o resumo do projeto
            dbc.Row([
                # Coluna para o card de resumo do projeto
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Resumo do Projeto"),
                        dbc.CardBody([
                            html.P(f"Total de Modelos Treinados: {len(self.results)}"),
                            html.P(f"Melhor Modelo: {best_model_name or 'N/A'}"),
                            html.P(f"Métrica do Melhor Modelo: {self._format_metric(best_metric)}"),
                        ])
                    ], className="mb-4")
                ], width=4),

                # Coluna para o card do tipo de problema
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 Tipo de Problema"),
                        dbc.CardBody([
                            html.H3(problem_type, id="problem-type",
                                    className="text-center"),
                            html.P("Classificação/Regressão detectada automaticamente",
                                   className="text-muted text-center")
                        ])
                    ], className="mb-4")
                ], width=4),

                # Coluna para o card de estatísticas
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Estatísticas"),
                        dbc.CardBody([
                            html.P(
                                f"Modelos Otimizados: {len([m for m in self.models.keys() if 'optimized' in m.lower() or 'otimizado' in m.lower()])}"
                            ),
                            html.P(
                                f"Inclui Ensemble: {'Sim' if any('ensemble' in m.lower() for m in self.models.keys()) else 'Não'}"
                            ),
                            html.P(
                                f"Status: {'✅ Completo' if self.results else '⚠️ Sem resultados'}"
                            )
                        ])
                    ], className="mb-4")
                ], width=4)
            ], className="mb-4"),

            # Linha para o ranking dos modelos
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🏆 Ranking dos Modelos (do melhor para o pior)"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='ranking-plot',
                                figure=self._build_ranking_figure()
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Linha para gráficos de comparação de métricas
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Comparação de Métricas"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='metric-selector',
                                options=[
                                    {'label': 'Todas as Métricas', 'value': 'all'},
                                    {'label': 'Acurácia/F1/R2', 'value': 'main'},
                                    {'label': 'Métricas Detalhadas', 'value': 'detailed'}
                                ],
                                value='main',
                                className="mb-3"
                            ),
                            dcc.Graph(
                                id='metrics-comparison',
                                figure=self._build_metrics_comparison_figure('main')
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Linha para o gráfico de importância das features
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔍 Feature Importance - Top 5 Modelos"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='feature-importance-plot',
                                figure=self._build_feature_importance_figure()
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Linha para visualização de previsões
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔮 Visualização de Previsões vs Real"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='model-selector',
                                options=[{'label': m, 'value': m}
                                         for m in self.models.keys()],
                                value=self._default_model_name(),
                                className="mb-3"
                            ),
                            dcc.Graph(
                                id='predictions-plot',
                                figure=self._build_predictions_figure(self._default_model_name())
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Linha para download de relatórios e exportação
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📥 Relatório e Exportação"),
                        dbc.CardBody([
                            html.Div([
                                dbc.Button("📄 Gerar Relatório PDF",
                                           id="generate-pdf",
                                           color="primary",
                                           className="me-2"),
                                dbc.Button("💾 Exportar Resultados CSV",
                                           id="export-csv",
                                           color="success",
                                           className="me-2"),
                                dbc.Button("🤖 Salvar Melhor Modelo",
                                           id="save-model",
                                           color="warning"),
                            ], className="d-flex justify-content-center"),

                            # Div para armazenar os links de download
                            html.Div(id='pdf-download', style={'display': 'none'}),
                            html.Div(id='csv-download', style={'display': 'none'}),
                            html.Div(id='model-download', style={'display': 'none'}),

                            # Barra de progresso
                            dbc.Progress(id="progress-bar", value=0,
                                         striped=True, animated=True,
                                         className="mt-3"),

                            # Mensagem de status
                            html.Div(id="status-message",
                                     className="mt-2 text-center")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Linha para tabela detalhada de resultados
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📋 Tabela Detalhada de Resultados"),
                        dbc.CardBody([
                            html.Div(
                                id='results-table',
                                children=self._build_results_table()
                            )
                        ])
                    ])
                ], width=12)
            ])
        ], fluid=True)

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

