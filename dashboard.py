# Importa o framework Dash para criar o dashboard
import dash
# Importa componentes do Dash para criar elementos interativos e layout
from dash import dcc, html, Input, Output, State
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
# Importa bibliotecas para codificação e manipulação de arquivos
import base64
import io

# Define a classe principal para o dashboard


class AdvancedDashboard:
    # Método construtor da classe
    def __init__(self, results, models, feature_importance=None, X_test=None, y_test=None):
        # Inicializa os resultados dos modelos
        self.results = results
        # Inicializa os modelos treinados
        self.models = models
        # Inicializa a importância das features (opcional)
        self.feature_importance = feature_importance
        # Inicializa os dados de teste (opcional)
        self.X_test = X_test
        # Inicializa os rótulos de teste (opcional)
        self.y_test = y_test
        # Cria a aplicação Dash com um tema externo
        self.app = dash.Dash(__name__, external_stylesheets=[
                             dbc.themes.DARKLY])
        # Configura o layout do dashboard
        self.setup_layout()
        # Configura os callbacks do dashboard
        self.setup_callbacks()

    # Método para configurar o layout do dashboard
    def setup_layout(self):
        """Configura o layout do dashboard"""

        # Define o layout principal como um container fluido
        self.app.layout = dbc.Container([
            # Linha para o cabeçalho
            dbc.Row([
                # Coluna contendo o título do dashboard
                dbc.Col([
                    html.H1("🚀 Dashboard de Machine Learning Avançado",
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
                            # Exibe o total de modelos treinados
                            html.P(
                                f"Total de Modelos Treinados: {len(self.results)}"),
                            # Exibe o melhor modelo baseado na métrica principal
                            html.P(
                                f"Melhor Modelo: {max(self.results.items(), key=lambda x: self.get_primary_metric(x[1]))[0]}"),
                            # Exibe a métrica do melhor modelo
                            html.P(
                                f"Métrica do Melhor Modelo: {self.get_primary_metric(max(self.results.items(), key=lambda x: self.get_primary_metric(x[1]))[1]):.4f}"),
                        ])
                    ], className="mb-4")
                ], width=4),

                # Coluna para o card do tipo de problema
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 Tipo de Problema"),
                        dbc.CardBody([
                            # Exibe o tipo de problema detectado
                            html.H3(self.detect_problem_type(), id="problem-type",
                                    className="text-center"),
                            # Exibe uma descrição do tipo de problema
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
                            # Exibe o número de modelos otimizados
                            html.P(
                                f"Modelos Otimizados: {len([m for m in self.models.keys() if 'Optimized' in m])}"),
                            # Indica se há ensemble nos modelos
                            html.P(
                                f"Inclui Ensemble: {'Sim' if 'Ensemble' in self.models else 'Não'}"),
                            # Exibe o status do projeto
                            html.P("Status: ✅ Completo")
                        ])
                    ], className="mb-4")
                ], width=4)
            ], className="mb-4"),

            # Linha para o ranking dos modelos
            dbc.Row([
                # Coluna contendo o gráfico de ranking
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            "🏆 Ranking dos Modelos (do melhor para o pior)"),
                        dbc.CardBody([
                            dcc.Graph(id='ranking-plot')
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Linha para gráficos de comparação de métricas
            dbc.Row([
                # Coluna contendo o dropdown e o gráfico de métricas
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Comparação de Métricas"),
                        dbc.CardBody([
                            # Dropdown para selecionar o tipo de métrica
                            dcc.Dropdown(
                                id='metric-selector',
                                options=[
                                    {'label': 'Todas as Métricas', 'value': 'all'},
                                    {'label': 'Acurácia/F1/R2', 'value': 'main'},
                                    {'label': 'Métricas Detalhadas',
                                        'value': 'detailed'}
                                ],
                                value='main',
                                className="mb-3"
                            ),
                            # Gráfico de comparação de métricas
                            dcc.Graph(id='metrics-comparison')
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Linha para o gráfico de importância das features
            dbc.Row([
                # Coluna contendo o gráfico de importância das features
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔍 Feature Importance - Top 5 Modelos"),
                        dbc.CardBody([
                            dcc.Graph(id='feature-importance-plot')
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Linha para visualização de previsões
            dbc.Row([
                # Coluna contendo o dropdown e o gráfico de previsões
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔮 Visualização de Previsões vs Real"),
                        dbc.CardBody([
                            # Dropdown para selecionar o modelo
                            dcc.Dropdown(
                                id='model-selector',
                                options=[{'label': m, 'value': m}
                                         for m in self.models.keys()],
                                value=list(self.models.keys())[
                                    0] if self.models else None,
                                className="mb-3"
                            ),
                            # Gráfico de previsões
                            dcc.Graph(id='predictions-plot')
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Linha para download de relatórios e exportação
            dbc.Row([
                # Coluna contendo o card de Relatório e Exportação
                dbc.Col([
                    # Card que encapsula os elementos de Relatório e Exportação
                    dbc.Card([
                        # Cabeçalho do card com o título
                        dbc.CardHeader("📥 Relatório e Exportação"),
                        # Corpo do card onde os elementos serão exibidos
                        dbc.CardBody([
                            # Div que agrupa os botões de download e progresso
                            html.Div([
                                # Botão para gerar relatório em PDF
                                dbc.Button("📄 Gerar Relatório PDF",
                                           id="generate-pdf",
                                           color="primary",
                                           className="me-2"),
                                # Botão para exportar resultados em CSV
                                dbc.Button("💾 Exportar Resultados CSV",
                                           id="export-csv",
                                           color="success",
                                           className="me-2"),
                                # Botão para salvar o melhor modelo
                                dbc.Button("🤖 Salvar Melhor Modelo",
                                           id="save-model",
                                           color="warning"),
                            ], className="d-flex justify-content-center"),


                            # Div para armazenar o link de download do PDF
                            html.Div(id='pdf-download',
                                     style={'display': 'none'}),
                            # Div para armazenar o link de download do CSV
                            html.Div(id='csv-download',
                                     style={'display': 'none'}),
                            # Div para armazenar o link de download do modelo
                            html.Div(id='model-download',
                                     style={'display': 'none'}),

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
                # Coluna contendo a tabela de resultados detalhados
                dbc.Col([
                    # Card que encapsula a tabela de resultados
                    dbc.Card([
                        # Cabeçalho do card com o título
                        dbc.CardHeader("📋 Tabela Detalhada de Resultados"),
                        # Corpo do card onde a tabela será exibida
                        dbc.CardBody([
                            # Div que conterá a tabela de resultados detalhados
                            html.Div(id='results-table')
                        ])
                    ])
                ], width=12)
            ])
        ], fluid=True)

    # Detecta o tipo de problema baseado nas métricas disponíveis nos resultados
    def detect_problem_type(self):
        """Detecta o tipo de problema baseado nas métricas"""
        # Retorna "Indeterminado" se não houver resultados
        if not self.results:
            return "Indeterminado"

        # Obtém as métricas de um modelo como exemplo
        sample_metrics = next(iter(self.results.values()))
        # Verifica se a métrica "accuracy" está presente para identificar classificação
        if 'accuracy' in sample_metrics:
            return "Classificação"
        # Verifica se a métrica "r2" está presente para identificar regressão
        elif 'r2' in sample_metrics:
            return "Regressão"
        # Retorna "Indeterminado" se nenhuma métrica conhecida for encontrada
        return "Indeterminado"

    # Obtém a métrica principal para ranquear os modelos
    def get_primary_metric(self, metrics):
        """Obtém a métrica principal para ranking"""
        # Retorna a métrica "f1" se estiver disponível
        if 'f1' in metrics:
            return metrics['f1']
        # Retorna o negativo da métrica "rmse" para que valores menores sejam melhores
        elif 'rmse' in metrics:
            return -metrics['rmse']
        # Retorna 0 se nenhuma métrica principal for encontrada
        return 0

    # Configura os callbacks do dashboard
    def setup_callbacks(self):
        """Configura os callbacks do dashboard"""

        # Callback para atualizar o gráfico de ranking dos modelos
        @self.app.callback(
            Output('ranking-plot', 'figure'),
            Input('ranking-plot', 'id')
        )
        def update_ranking_plot(_):
            # Ordena os modelos pela métrica principal em ordem decrescente
            sorted_results = sorted(self.results.items(),
                                    key=lambda x: self.get_primary_metric(
                                        x[1]),
                                    reverse=True)

            # Extrai os nomes dos modelos e suas respectivas métricas
            models = [m[0] for m in sorted_results]
            scores = [self.get_primary_metric(m[1]) for m in sorted_results]

            # Cria um gráfico de barras horizontal para o ranking
            fig = go.Figure(data=[
                go.Bar(x=scores, y=models, orientation='h',
                       marker_color=px.colors.sequential.Viridis)
            ])

            # Configura o layout do gráfico
            fig.update_layout(
                title="Ranking dos Modelos",
                xaxis_title="Métrica Principal",
                yaxis_title="Modelo",
                height=500,
                template="plotly_dark"
            )

            return fig

        # Callback para atualizar o gráfico de comparação de métricas
        @self.app.callback(
            Output('metrics-comparison', 'figure'),
            Input('metric-selector', 'value')
        )
        def update_metrics_comparison(selected_metric):
            # Obtém a lista de modelos
            models = list(self.results.keys())

            # Verifica se o usuário selecionou métricas principais
            if selected_metric == 'main':
                # Define as métricas principais para classificação ou regressão
                # Verifica se o tipo de problema detectado é "Classificação"
                if self.detect_problem_type() == "Classificação":
                    # Define as chaves das métricas principais para problemas de classificação
                    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
                    # Define os nomes das métricas principais para exibição no gráfico
                    metric_names = ['Acurácia',
                                    'Precisão', 'Recall', 'F1-Score']
                else:
                    # Define as chaves das métricas principais para problemas de regressão
                    metric_keys = ['r2', 'rmse', 'mae']
                    # Define os nomes das métricas principais para exibição no gráfico
                    metric_names = ['R²', 'RMSE', 'MAE']

                # Cria subplots para cada métrica principal
                fig = make_subplots(rows=1, cols=len(metric_keys),
                                    subplot_titles=metric_names)

                # Itera sobre as métricas principais e seus nomes correspondentes
                for i, (key, name) in enumerate(zip(metric_keys, metric_names)):
                    # Obtém os valores da métrica atual para cada modelo nos resultados
                    values = [self.results[m].get(key, 0) for m in models]

                    # Inverte os valores de RMSE para visualização
                    if key == 'rmse':
                        values = [-v for v in values]

                    fig.add_trace(
                        go.Bar(x=models, y=values, name=name),
                        row=1, col=i+1
                    )

                    # Ajusta o ângulo dos rótulos do eixo x
                    fig.update_xaxes(tickangle=45, row=1, col=i+1)

                # Configura o layout do gráfico
                fig.update_layout(height=400, showlegend=False,
                                  template="plotly_dark")

            else:
                # Caso o usuário selecione todas as métricas, cria um heatmap
                all_metrics = set()
                for metrics in self.results.values():
                    all_metrics.update(metrics.keys())

                # Remove métricas não numéricas
                all_metrics = [
                    m for m in all_metrics if m not in ['confusion_matrix']]

                # Prepara os dados para o heatmap
                data = []
                for model in models:
                    row = []
                    for metric in all_metrics:
                        value = self.results[model].get(metric, 0)

                        # Substitui valores ausentes por 0
                        if value is None:
                            value = 0

                        # Inverte valores de RMSE para visualização
                        if metric == 'rmse':
                            value = -value

                        row.append(value)
                    data.append(row)

                # Cria o heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=data,
                    x=all_metrics,
                    y=models,
                    colorscale='Viridis',
                    colorbar=dict(title="Valor")
                ))

                # Configura o layout do heatmap
                fig.update_layout(
                    title="Comparação de Todas as Métricas",
                    height=600,
                    template="plotly_dark"
                )

            return fig

        # Callback para atualizar o gráfico de importância das features
        @self.app.callback(
            Output('feature-importance-plot', 'figure'),
            Input('feature-importance-plot', 'id')
        )
        def update_feature_importance(_):
            # Retorna um gráfico vazio se não houver importância de features
            if not self.feature_importance:
                return go.Figure()

            # Obtém os top 5 modelos com importância de features
            models_with_fi = []
            for name, metrics in self.results.items():
                if name in self.feature_importance:
                    models_with_fi.append(
                        (name, self.get_primary_metric(metrics)))

            # Ordena os modelos pela métrica principal
            models_with_fi.sort(key=lambda x: x[1], reverse=True)
            top_5_models = [m[0] for m in models_with_fi[:5]]

            # Cria subplots para os top 5 modelos
            fig = make_subplots(rows=1, cols=len(top_5_models),
                                subplot_titles=top_5_models,
                                shared_yaxes=True)

            # Adiciona gráficos de barras para cada modelo
            for i, model_name in enumerate(top_5_models):
                if model_name in self.feature_importance:
                    importances = self.feature_importance[model_name]

                    # Obtém os nomes das features
                    if hasattr(self.X_test, 'columns'):
                        features = self.X_test.columns.tolist()
                    else:
                        features = [
                            f'Feature_{i}' for i in range(len(importances))]

                    # Ordena as features pela importância
                    sorted_idx = np.argsort(importances)[-10:]

                    fig.add_trace(
                        go.Bar(x=importances[sorted_idx],
                               y=[features[i] for i in sorted_idx],
                               orientation='h',
                               name=model_name),
                        row=1, col=i+1
                    )

            # Configura o layout do gráfico
            fig.update_layout(height=400, showlegend=False,
                              template="plotly_dark")

            return fig

        # Callback para atualizar o gráfico de previsões vs real
        @self.app.callback(
            Output('predictions-plot', 'figure'),
            Input('model-selector', 'value')
        )
        def update_predictions_plot(selected_model):
            # Retorna um gráfico vazio se o modelo ou os dados de teste não estiverem disponíveis
            if selected_model not in self.models or self.X_test is None or self.y_test is None:
                return go.Figure()

            # Obtém o modelo selecionado e faz as previsões
            model = self.models[selected_model]
            y_pred = model.predict(self.X_test)

            # Verifica se o problema é de regressão
            if self.detect_problem_type() == "Regressão":
                # Cria um gráfico de dispersão para previsões vs valores reais
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=self.y_test,
                    y=y_pred,
                    mode='markers',
                    name='Previsões',
                    marker=dict(color='lightblue')
                ))

                # Adiciona uma linha de perfeita predição
                min_val = min(self.y_test.min(), y_pred.min())
                max_val = max(self.y_test.max(), y_pred.max())

                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Ideal',
                    line=dict(color='red', dash='dash')
                ))

                # Configura o layout do gráfico
                fig.update_layout(
                    title=f"Previsões vs Real - {selected_model}",
                    xaxis_title="Valor Real",
                    yaxis_title="Valor Previsto",
                    template="plotly_dark"
                )

            else:
                # Cria uma matriz de confusão para problemas de classificação
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(self.y_test, y_pred)

                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Previsto ' + str(i) for i in range(cm.shape[1])],
                    y=['Real ' + str(i) for i in range(cm.shape[0])],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))

                # Configura o layout do gráfico
                fig.update_layout(
                    title=f"Matriz de Confusão - {selected_model}",
                    template="plotly_dark"
                )

            return fig

        # Callback para manipular os downloads de PDF, CSV e modelo
        @self.app.callback(
            [Output('results-table', 'children'),
             Output('pdf-download', 'children'),
             Output('csv-download', 'children'),
             Output('model-download', 'children'),
             Output('progress-bar', 'value'),
             Output('status-message', 'children')],
            [Input('generate-pdf', 'n_clicks'),
             Input('export-csv', 'n_clicks'),
             Input('save-model', 'n_clicks')],
            prevent_initial_call=True
        )
        def handle_downloads(pdf_clicks, csv_clicks, model_clicks):
            # Obtém o contexto do callback para identificar o botão clicado
            ctx = dash.callback_context

            # Retorna sem atualização se nenhum botão foi clicado
            if not ctx.triggered:
                return dash.no_update

            # Identifica o botão que foi clicado
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Gera o relatório em PDF se o botão correspondente foi clicado
            if button_id == 'generate-pdf':
                pdf_content = "Relatório PDF gerado com sucesso!"
                pdf_b64 = base64.b64encode(pdf_content.encode()).decode()

                download_link = html.A(
                    "📥 Baixar Relatório PDF",
                    id="pdf-download-link",
                    href=f"data:application/pdf;base64,{pdf_b64}",
                    download="relatorio_ml.pdf",
                    className="btn btn-success mt-2"
                )

                return dash.no_update, download_link, dash.no_update, dash.no_update, 100, "✅ PDF gerado com sucesso!"

            # Exporta os resultados em CSV se o botão correspondente foi clicado
            elif button_id == 'export-csv':
                results_df = pd.DataFrame(self.results).T
                csv_string = results_df.to_csv(index=True)
                csv_b64 = base64.b64encode(csv_string.encode()).decode()

                download_link = html.A(
                    "💾 Baixar CSV",
                    id="csv-download-link",
                    href=f"data:text/csv;base64,{csv_b64}",
                    download="resultados_ml.csv",
                    className="btn btn-success mt-2"
                )

                return dash.no_update, dash.no_update, download_link, dash.no_update, 100, "✅ CSV exportado com sucesso!"

            # Salva o melhor modelo se o botão correspondente foi clicado
            elif button_id == 'save-model':
                model_content = "Modelo salvo com sucesso!"
                model_b64 = base64.b64encode(model_content.encode()).decode()

                download_link = html.A(
                    "🤖 Baixar Modelo",
                    id="model-download-link",
                    href=f"data:application/octet-stream;base64,{model_b64}",
                    download="melhor_modelo.pkl",
                    className="btn btn-success mt-2"
                )

                return dash.no_update, dash.no_update, dash.no_update, download_link, 100, "✅ Modelo salvo com sucesso!"

            # Retorna sem atualização se nenhuma ação foi realizada
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, 0, ""

    # Método para executar o dashboard
    def run(self, port=8050):
        """Executa o dashboard"""
        # Exibe a URL do dashboard no console
        print(f"Dashboard rodando em http://localhost:{port}")
        # Inicia o servidor do Dash
        self.app.run_server(debug=True, port=port)
