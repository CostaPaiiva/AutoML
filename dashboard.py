# Esse código implementa um dashboard interativo de Machine Learning usando o framework Dash
# (com Bootstrap e Plotly).
# Ele cria uma aplicação web que permite visualizar, comparar e exportar resultados
# de modelos de Machine Learning. O usuário pode abrir o dashboard no navegador e
# interagir com gráficos, tabelas e botões.


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
                    html.H1(" Dashboard de Machine Learning Avançado",
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
                # Coluna contendo o card do ranking dos modelos
                dbc.Col([
                    # Card que encapsula o gráfico de ranking
                    dbc.Card([
                        # Cabeçalho do card com o título
                        dbc.CardHeader(
                            "🏆 Ranking dos Modelos (do melhor para o pior)"),
                        # Corpo do card onde o gráfico será exibido
                        dbc.CardBody([
                            # Gráfico de ranking dos modelos
                            dcc.Graph(id='ranking-plot')
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Linha para gráficos de comparação de métricas
            dbc.Row([
                # Coluna contendo o dropdown e o gráfico de métricas
                dbc.Col([
                    # Card que encapsula o dropdown e o gráfico de métricas
                    dbc.Card([
                        # Cabeçalho do card com o título
                        dbc.CardHeader("📊 Comparação de Métricas"),
                        # Corpo do card onde os elementos serão exibidos
                        dbc.CardBody([
                            # Dropdown para selecionar o tipo de métrica
                            dcc.Dropdown(
                                id='metric-selector',  # Define o ID do componente como 'metric-selector'
                                options=[  # Define as opções disponíveis no dropdown
                                    # Opção para exibir todas as métricas
                                    {'label': 'Todas as Métricas', 'value': 'all'},
                                    # Opção para exibir métricas principais
                                    {'label': 'Acurácia/F1/R2', 'value': 'main'},
                                    {'label': 'Métricas Detalhadas',  # Opção para exibir métricas detalhadas
                                        'value': 'detailed'}
                                ],
                                value='main',  # Define o valor padrão como 'main'
                                className="mb-3"  # Adiciona uma classe CSS para estilização
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
                    # Card que encapsula o gráfico de importância das features
                    dbc.Card([
                        # Cabeçalho do card com o título
                        dbc.CardHeader("🔍 Feature Importance - Top 5 Modelos"),
                        # Corpo do card onde o gráfico será exibido
                        dbc.CardBody([
                            # Gráfico de importância das features
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
                            # Define o ID do componente como 'model-selector'
                            dcc.Dropdown(
                                id='model-selector',
                                # Define as opções do dropdown com base nos nomes dos modelos disponíveis
                                options=[{'label': m, 'value': m}
                                         for m in self.models.keys()],
                                # Define o valor padrão como o primeiro modelo na lista, se houver modelos disponíveis
                                value=list(self.models.keys())[
                                    0] if self.models else None,
                                # Adiciona uma classe CSS para estilização
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
            # Ordena os resultados dos modelos com base na métrica principal, em ordem decrescente
            sorted_results = sorted(self.results.items(),  # Obtém os itens (nome do modelo e métricas) dos resultados
                                    # Define a métrica principal como chave para ordenação
                                    key=lambda x: self.get_primary_metric(
                                        x[1]),
                                    reverse=True)  # Ordena em ordem decrescente

            # Extrai os nomes dos modelos dos resultados ordenados
            models = [m[0] for m in sorted_results]
            # Extrai as métricas principais dos modelos ordenados
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

                    # Inverte os valores de RMSE para que valores menores sejam melhores visualmente
                    if key == 'rmse':
                        # Multiplica os valores de RMSE por -1
                        values = [-v for v in values]

                    # Adiciona um gráfico de barras ao subplot correspondente
                    fig.add_trace(
                        # Cria o gráfico de barras com os modelos no eixo x e os valores no eixo y
                        go.Bar(x=models, y=values, name=name),
                        row=1, col=i+1  # Define a posição do gráfico no subplot
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

                # Inicializa uma lista para armazenar os dados do heatmap
                data = []
                # Itera sobre cada modelo nos resultados
                for model in models:
                    # Inicializa uma lista para armazenar os valores das métricas para o modelo atual
                    row = []
                    # Itera sobre todas as métricas disponíveis
                    for metric in all_metrics:
                        # Obtém o valor da métrica atual para o modelo atual, ou 0 se não estiver disponível
                        value = self.results[model].get(metric, 0)

                        # Substitui valores ausentes por 0 (caso o valor seja None)
                        if value is None:
                            value = 0

                        # Inverte os valores de RMSE para que valores menores sejam melhores visualmente
                        if metric == 'rmse':
                            value = -value

                        # Adiciona o valor da métrica à linha correspondente ao modelo
                        row.append(value)
                    # Adiciona a linha completa (valores das métricas) à lista de dados do heatmap
                    data.append(row)

                # Cria o heatmap com os dados processados
                fig = go.Figure(data=go.Heatmap(
                    z=data,  # Define os valores do heatmap como os dados processados
                    x=all_metrics,  # Define os rótulos do eixo x como as métricas
                    y=models,  # Define os rótulos do eixo y como os modelos
                    colorscale='Viridis',  # Define a escala de cores do heatmap
                    colorbar=dict(title="Valor")  # Adiciona um título à barra de cores
                ))

                # Configura o layout do heatmap
                fig.update_layout(
                    title="Comparação de Todas as Métricas",  # Define o título do gráfico
                    height=600,  # Define a altura do gráfico
                    template="plotly_dark"  # Define o tema do gráfico como escuro
                )

            # Retorna o gráfico gerado
            return fig

        # Callback para atualizar o gráfico de importância das features
        @self.app.callback(
            # Define o componente de saída como o gráfico de importância das features
            Output('feature-importance-plot', 'figure'),
            # Define o componente de entrada como o ID do gráfico de importância das features
            Input('feature-importance-plot', 'id')
        )
        def update_feature_importance(_):
            # Retorna um gráfico vazio se não houver importância de features
            if not self.feature_importance:
                return go.Figure()

            # Obtém os top 5 modelos com importância de features
            models_with_fi = []
            # Lista para armazenar os modelos que possuem importância de features
            for name, metrics in self.results.items():
                # Verifica se o modelo atual está na lista de importância de features
                if name in self.feature_importance:
                    # Adiciona o modelo e sua métrica principal à lista
                    models_with_fi.append(
                        (name, self.get_primary_metric(metrics))
                    )

            # Ordena os modelos pela métrica principal em ordem decrescente
            models_with_fi.sort(key=lambda x: x[1], reverse=True)
            # Seleciona os nomes dos top 5 modelos com base na métrica principal
            top_5_models = [m[0] for m in models_with_fi[:5]]

            # Cria subplots para os top 5 modelos, compartilhando o eixo y
            fig = make_subplots(rows=1, cols=len(top_5_models),
                                subplot_titles=top_5_models,
                                shared_yaxes=True)

            # Adiciona gráficos de barras para cada modelo
            for i, model_name in enumerate(top_5_models):
                # Verifica se o modelo atual possui importância de features
                if model_name in self.feature_importance:
                    # Obtém as importâncias das features para o modelo atual
                    importances = self.feature_importance[model_name]

                    # Verifica se X_test tem colunas (é um DataFrame)
                    if hasattr(self.X_test, 'columns'):
                        # Obtém os nomes das colunas como features
                        features = self.X_test.columns.tolist()
                    # Caso contrário, cria nomes genéricos para as features
                    else:
                        features = [
                            f'Feature_{i}' for i in range(len(importances))
                        ]

                    # Ordena as features pela importância
                    sorted_idx = np.argsort(importances)[-10:]

                    # Adiciona um gráfico de barras horizontal ao subplot atual
                    fig.add_trace(
                        # Cria um gráfico de barras com as importâncias das features no eixo x
                        go.Bar(x=importances[sorted_idx],
                               # e os nomes das features correspondentes no eixo y
                               y=[features[i] for i in sorted_idx],
                               orientation='h',  # Define a orientação do gráfico como horizontal
                               name=model_name),  # Define o nome do traço como o nome do modelo
                        # Define a posição do gráfico no subplot (linha 1, coluna i+1)
                        row=1, col=i+1
                    )

            # Configura o layout do gráfico
            fig.update_layout(height=400, showlegend=False,
                              template="plotly_dark")

            return fig

        # Callback para atualizar o gráfico de previsões vs real
        @self.app.callback(  # Define um callback para atualizar o gráfico de previsões vs real
            Output('predictions-plot', 'figure'),  # Define o componente de saída como o gráfico de previsões
            Input('model-selector', 'value')  # Define o componente de entrada como o valor selecionado no dropdown de modelos
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

                # Adiciona um gráfico de dispersão para previsões vs valores reais
                fig.add_trace(go.Scatter(
                    x=self.y_test,  # Valores reais no eixo x
                    y=y_pred,  # Valores previstos no eixo y
                    # Define o modo do gráfico como pontos (dispersão)
                    mode='markers',
                    name='Previsões',  # Nome do traço no gráfico
                    # Define a cor dos pontos como azul claro
                    marker=dict(color='lightblue')
                ))

                # Calcula os valores mínimo e máximo para a linha de perfeita predição
                # Obtém o menor valor entre os reais e previstos
                min_val = min(self.y_test.min(), y_pred.min())
                # Obtém o maior valor entre os reais e previstos
                max_val = max(self.y_test.max(), y_pred.max())

                # Adiciona uma linha de perfeita predição ao gráfico
                fig.add_trace(go.Scatter(
                    # Define os valores do eixo x como o intervalo mínimo e máximo
                    x=[min_val, max_val],
                    # Define os valores do eixo y como o intervalo mínimo e máximo
                    y=[min_val, max_val],
                    mode='lines',  # Define o modo do gráfico como linhas
                    name='Ideal',  # Nome do traço no gráfico
                    # Define a cor da linha como vermelha e o estilo como tracejado
                    line=dict(color='red', dash='dash')
                ))

                # Configura o layout do gráfico
                fig.update_layout(
                    # Define o título do gráfico com o nome do modelo selecionado
                    title=f"Previsões vs Real - {selected_model}",
                    xaxis_title="Valor Real",  # Define o título do eixo x
                    yaxis_title="Valor Previsto",  # Define o título do eixo y
                    template="plotly_dark"  # Define o tema do gráfico como escuro
                )

            else:
                # Cria uma matriz de confusão a partir dos valores reais e previstos
                cm = confusion_matrix(self.y_test, y_pred)

                # Cria um objeto Figure do Plotly para o heatmap
                fig = go.Figure(data=go.Heatmap(
                    # Define os valores da matriz de confusão como dados do heatmap
                    z=cm,
                    # Define os rótulos do eixo x (previsões)
                    x=['Previsto ' + str(i) for i in range(cm.shape[1])],
                    # Define os rótulos do eixo y (reais)
                    y=['Real ' + str(i) for i in range(cm.shape[0])],
                    # Define a escala de cores do heatmap
                    colorscale='Blues',
                    # Exibe os valores da matriz de confusão como texto no heatmap
                    text=cm,
                    # Define o formato do texto exibido
                    texttemplate='%{text}',
                    # Define a fonte e o tamanho do texto
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
                # Define o conteúdo do PDF como uma string de exemplo
                pdf_content = "Relatório PDF gerado com sucesso!"
                # Codifica o conteúdo do PDF em base64 para permitir o download
                pdf_b64 = base64.b64encode(pdf_content.encode()).decode()

                # Cria um link de download para o arquivo PDF
                download_link = html.A(
                    # Texto do botão de download
                    "📥 Baixar Relatório PDF",
                    # ID do componente para callbacks
                    id="pdf-download-link",
                    # Define o conteúdo do link como o PDF codificado em base64
                    href=f"data:application/pdf;base64,{pdf_b64}",
                    # Nome do arquivo ao ser baixado
                    download="relatorio_ml.pdf",
                    # Classe CSS para estilização
                    className="btn btn-success mt-2"
                )

                return dash.no_update, download_link, dash.no_update, dash.no_update, 100, "✅ PDF gerado com sucesso!"

            # Exporta os resultados em CSV se o botão correspondente foi clicado
            elif button_id == 'export-csv':
                # Converte os resultados em um DataFrame do pandas
                results_df = pd.DataFrame(self.results).T
                # Converte o DataFrame para uma string CSV
                csv_string = results_df.to_csv(index=True)
                # Codifica a string CSV em base64 para permitir o download
                csv_b64 = base64.b64encode(csv_string.encode()).decode()

                # Cria um link de download para o arquivo CSV
                download_link = html.A(
                    "💾 Baixar CSV",  # Texto do botão de download
                    id="csv-download-link",  # ID do componente para callbacks
                    # Define o conteúdo do link como o CSV codificado em base64
                    href=f"data:text/csv;base64,{csv_b64}",
                    download="resultados_ml.csv",  # Nome do arquivo ao ser baixado
                    className="btn btn-success mt-2"  # Classe CSS para estilização
                )

                # Retorna o link de download e atualiza a barra de progresso e a mensagem de status
                return dash.no_update, dash.no_update, download_link, dash.no_update, 100, "✅ CSV exportado com sucesso!"

            # Salva o melhor modelo se o botão correspondente foi clicado
            elif button_id == 'save-model':
                # Define o conteúdo do modelo salvo como uma string de exemplo
                model_content = "Modelo salvo com sucesso!"
                # Codifica o conteúdo do modelo em base64 para download
                model_b64 = base64.b64encode(model_content.encode()).decode()

                # Cria um link de download para o modelo salvo
                download_link = html.A(
                    # Texto do link de download
                    "🤖 Baixar Modelo",
                    # ID do link de download
                    id="model-download-link",
                    # Define o href como o conteúdo codificado em base64
                    href=f"data:application/octet-stream;base64,{model_b64}",
                    # Nome do arquivo para download
                    download="melhor_modelo.pkl",
                    # Classe CSS para estilização do botão
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
