
# Classe PDFReportGenerator: gera relatórios PDF de ML.
# 1. Configura identidade visual (cores, logo, estilos).
# 2. Cria capa com título, subtítulo e informações do dataset.
# 3. Adiciona seção de informações gerais sobre modelos e problema.
# 4. Gera resumo executivo destacando o melhor modelo.
# 5. Mostra métricas detalhadas do modelo vencedor.
# 6. Cria ranking completo dos modelos avaliados.
# 7. Exibe métricas adicionais dos outros modelos.
# 8. Inclui recomendações práticas e próximos passos.
# 9. Adiciona rodapé com marca e informações finais.
# 10. Método generate_report() monta todo o PDF e salva o arquivo.


from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import os


class PDFReportGenerator:
    def __init__(self, results, models, best_model_name, problem_type, data_info=None):
        self.results = results
        self.models = models
        self.best_model_name = best_model_name
        self.problem_type = problem_type
        self.data_info = data_info or {}
        self.styles = getSampleStyleSheet()
        self.setup_brand()
        self.setup_custom_styles()

    def setup_brand(self):
        """Configura identidade visual da plataforma"""
        self.PLATFORM_NAME = "AutoML"
        self.PRIMARY_COLOR = "#1E88E5"
        self.SECONDARY_COLOR = "#0D47A1"
        self.SUBTITLE = "Sistema Inteligente de Machine Learning"
        self.LOGO_PATH = "logo.png"

        self.primary_color = colors.HexColor(self.PRIMARY_COLOR)
        self.secondary_color = colors.HexColor(self.SECONDARY_COLOR)
        self.accent_color = colors.HexColor("#F39C12")
        self.success_color = colors.HexColor("#27AE60")
        self.light_bg = colors.HexColor("#F4F8FD")
        self.border_color = colors.HexColor("#D6E4F0")
        self.text_dark = colors.HexColor("#1C2833")
        self.text_muted = colors.HexColor("#5D6D7E")

    def setup_custom_styles(self):
        """Configura estilos personalizados para o PDF"""
        if 'CoverTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CoverTitle',
                parent=self.styles['Heading1'],
                fontSize=30,
                leading=34,
                textColor=self.primary_color,
                alignment=1,
                spaceAfter=12
            ))

        if 'CoverSubtitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CoverSubtitle',
                parent=self.styles['Normal'],
                fontSize=13,
                leading=17,
                textColor=self.text_muted,
                alignment=1,
                spaceAfter=8
            ))

        if 'SectionTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionTitle',
                parent=self.styles['Heading2'],
                fontSize=16,
                leading=20,
                textColor=self.secondary_color,
                spaceAfter=10,
                spaceBefore=8
            ))

        if 'BodyTextCustom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BodyTextCustom',
                parent=self.styles['Normal'],
                fontSize=10,
                leading=14,
                textColor=self.text_dark,
                spaceAfter=6
            ))

        if 'MutedText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='MutedText',
                parent=self.styles['Normal'],
                fontSize=9,
                leading=12,
                textColor=self.text_muted,
                spaceAfter=6
            ))

        if 'SubtitleCustom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SubtitleCustom',
                parent=self.styles['Normal'],
                fontSize=12,
                leading=14,
                textColor=self.text_muted,
                spaceAfter=6
            ))

        if 'TableCellCustom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='TableCellCustom',
                parent=self.styles['Normal'],
                fontSize=8,
                leading=10,
                textColor=colors.black
            ))

        if 'TableCellHeaderCustom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='TableCellHeaderCustom',
                parent=self.styles['Normal'],
                fontSize=9,
                leading=11,
                textColor=colors.whitesmoke,
                alignment=1
            ))

        if 'FooterCustom' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='FooterCustom',
                parent=self.styles['Normal'],
                fontSize=8,
                leading=10,
                textColor=colors.grey,
                alignment=1
            ))

        if 'BrandBadge' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BrandBadge',
                parent=self.styles['Normal'],
                fontSize=10,
                leading=12,
                textColor=self.secondary_color,
                alignment=1,
                spaceAfter=10
            ))

    def generate_report(self, filename="relatorio_ml.pdf"):
        """Gera o relatório PDF completo"""
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        story = []

        story.extend(self._create_cover_page())
        story.append(PageBreak())

        story.extend(self._create_general_info())
        story.append(Spacer(1, 14))

        story.extend(self._create_executive_summary())
        story.append(Spacer(1, 14))

        story.extend(self._create_best_model_section())
        story.append(Spacer(1, 14))

        story.extend(self._create_ranking_table())
        story.append(PageBreak())

        story.extend(self._create_metrics_section())
        story.append(Spacer(1, 14))

        story.extend(self._create_recommendations())
        story.append(Spacer(1, 16))

        story.extend(self._create_footer())

        doc.build(
            story,
            onFirstPage=self._add_page_decor,
            onLaterPages=self._add_page_decor
        )
        print(f"Relatório gerado: {filename}")
        return filename

    def _add_page_decor(self, canvas, doc):
        """Cabeçalho e rodapé com identidade visual"""
        canvas.saveState()

        canvas.setStrokeColor(self.primary_color)
        canvas.setLineWidth(1.2)
        canvas.line(36, A4[1] - 24, A4[0] - 36, A4[1] - 24)

        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(self.secondary_color)
        canvas.drawString(36, A4[1] - 18, self.PLATFORM_NAME)

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(36, 18, f"Relatório gerado automaticamente pelo {self.PLATFORM_NAME}")
        canvas.drawRightString(A4[0] - 36, 18, f"Página {canvas.getPageNumber()}")

        canvas.restoreState()

    def _safe_float(self, value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _safe_text(self, value, default="N/A"):
        try:
            if value is None:
                return default
            return str(value)
        except Exception:
            return default

    def _truncate_text(self, text, max_chars=42):
        text = self._safe_text(text, "")
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."

    def _metric_label_for_display(self):
        if self.problem_type == 'classification':
            return "F1-Score"
        return "RMSE"

    def _create_cover_page(self):
        """Capa do relatório personalizada com marca"""
        elements = []

        elements.append(Spacer(1, 45))

        if os.path.exists(self.LOGO_PATH):
            try:
                logo = Image(self.LOGO_PATH, width=1.4 * inch, height=1.4 * inch)
                logo.hAlign = 'CENTER'
                elements.append(logo)
                elements.append(Spacer(1, 18))
            except Exception:
                pass

        elements.append(Paragraph(self.PLATFORM_NAME, self.styles['CoverTitle']))
        elements.append(Paragraph(self.SUBTITLE, self.styles['CoverSubtitle']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("RELATÓRIO EXECUTIVO DE MACHINE LEARNING", self.styles['BrandBadge']))
        elements.append(Spacer(1, 24))

        subtitle = (
            f"Análise automatizada de modelos para problema de "
            f"{self._safe_text(self.problem_type).upper()}"
        )
        elements.append(Paragraph(subtitle, self.styles['CoverSubtitle']))
        elements.append(Spacer(1, 30))

        cover_data = [
            ["Melhor Modelo", self._safe_text(self.best_model_name)],
            ["Quantidade de Modelos", self._safe_text(len(self.models))],
            ["Data de Geração", datetime.now().strftime("%d/%m/%Y %H:%M")],
        ]

        if self.data_info:
            cover_data.extend([
                ["Dataset", self._safe_text(self.data_info.get('dataset_name', 'N/A'))],
                ["Amostras", self._safe_text(self.data_info.get('n_samples', 'N/A'))],
                ["Features", self._safe_text(self.data_info.get('n_features', 'N/A'))],
            ])

        table = Table(cover_data, colWidths=[2.2 * inch, 3.3 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), self.primary_color),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('BACKGROUND', (1, 0), (1, -1), self.light_bg),
            ('TEXTCOLOR', (1, 0), (1, -1), self.text_dark),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.75, self.border_color),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 35))

        elements.append(Paragraph(
            f"Este documento foi gerado automaticamente pelo {self.PLATFORM_NAME} e consolida "
            "os resultados do pipeline de Machine Learning, incluindo ranking dos modelos, "
            "métricas de desempenho e recomendações práticas para produção.",
            self.styles['BodyTextCustom']
        ))

        return elements

    def _create_general_info(self):
        """Cria seção de informações gerais"""
        elements = []

        elements.append(Paragraph("INFORMAÇÕES GERAIS", self.styles['SectionTitle']))

        info_data = [
            ["Plataforma", self.PLATFORM_NAME],
            ["Tipo de Problema", self._safe_text(self.problem_type).upper()],
            ["Total de Modelos", self._safe_text(len(self.models))],
            ["Melhor Modelo", self._safe_text(self.best_model_name)],
            ["Status", "PROCESSAMENTO COMPLETO"],
            ["Data do Relatório", datetime.now().strftime("%d/%m/%Y %H:%M")],
        ]

        if self.data_info:
            info_data.extend([
                ["Dataset", self._safe_text(self.data_info.get('dataset_name', 'N/A'))],
                ["Amostras", self._safe_text(self.data_info.get('n_samples', 'N/A'))],
                ["Features", self._safe_text(self.data_info.get('n_features', 'N/A'))],
            ])

        table = Table(info_data, colWidths=[2.0 * inch, 4.0 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EAF3FC')),
            ('TEXTCOLOR', (0, 0), (-1, -1), self.text_dark),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.7, self.border_color),
        ]))

        elements.append(table)
        return elements

    def _create_executive_summary(self):
        """Cria resumo executivo"""
        elements = []

        elements.append(Paragraph("RESUMO EXECUTIVO", self.styles['SectionTitle']))

        summary_text = f"""
        Este relatório apresenta os resultados do processo automatizado conduzido pelo
        <b>{self.PLATFORM_NAME}</b>, com avaliação de <b>{len(self.models)}</b> modelos para um
        problema de <b>{self._safe_text(self.problem_type).upper()}</b>.

        O modelo <b>{self._safe_text(self.best_model_name)}</b> apresentou o melhor desempenho geral
        segundo os critérios definidos pelo sistema, sendo o principal candidato para implantação.

        Nas próximas seções, você encontrará o ranking completo dos modelos avaliados,
        análise detalhada de métricas e recomendações práticas para os próximos passos.
        """

        elements.append(Paragraph(summary_text, self.styles['BodyTextCustom']))
        return elements
        def _create_best_model_section(self):
            """Cria seção detalhada do melhor modelo"""
            elements = []  # Inicializa uma lista vazia para armazenar os elementos do PDF.

            elements.append(Paragraph("MELHOR MODELO IDENTIFICADO", self.styles['SectionTitle']))  # Adiciona um título de seção à lista de elementos.

            if self.best_model_name in self.results:  # Verifica se o nome do melhor modelo existe nos resultados.
                metrics = self.results[self.best_model_name]  # Obtém as métricas do melhor modelo.

                if self.problem_type == 'classification':  # Verifica se o tipo de problema é classificação.
                    metrics_data = [  # Define os dados da tabela de métricas para classificação.
                        ["Métrica", "Valor"],  # Cabeçalho da tabela.
                        ["Acurácia", f"{self._safe_float(metrics.get('accuracy', 0)):.4f}"],  # Adiciona a acurácia formatada.
                        ["Precisão", f"{self._safe_float(metrics.get('precision', 0)):.4f}"],  # Adiciona a precisão formatada.
                        ["Recall", f"{self._safe_float(metrics.get('recall', 0)):.4f}"],  # Adiciona o recall formatado.
                        ["F1-Score", f"{self._safe_float(metrics.get('f1', 0)):.4f}"],  # Adiciona o F1-Score formatado.
                        ["ROC AUC", f"{self._safe_float(metrics.get('roc_auc', 0)):.4f}"],  # Adiciona o ROC AUC formatado.
                    ]
                else:  # Se não for classificação, assume que é regressão.
                    metrics_data = [  # Define os dados da tabela de métricas para regressão.
                        ["Métrica", "Valor"],  # Cabeçalho da tabela.
                        ["R² Score", f"{self._safe_float(metrics.get('r2', 0)):.4f}"],  # Adiciona o R² Score formatado.
                        ["RMSE", f"{self._safe_float(metrics.get('rmse', 0)):.4f}"],  # Adiciona o RMSE formatado.
                        ["MAE", f"{self._safe_float(metrics.get('mae', 0)):.4f}"],  # Adiciona o MAE formatado.
                        ["MAPE", f"{self._safe_float(metrics.get('mape', 0)):.2f}%"],  # Adiciona o MAPE formatado como porcentagem.
                    ]
                # Cria um objeto Table com os dados das métricas.
                # 'metrics_data' contém as métricas e seus valores.
                # 'repeatRows=1' garante que a linha de cabeçalho da tabela se repita se a tabela se estender por várias páginas.
                # 'colWidths' define a largura de cada coluna na tabela.
                table = Table(metrics_data, repeatRows=1, colWidths=[2.2 * inch, 1.8 * inch])
                # Aplica um estilo visual à tabela usando TableStyle.
                table.setStyle(TableStyle([
                    # Define a cor de fundo para a linha do cabeçalho (primeira linha, índice 0) como a cor secundária.
                    ('BACKGROUND', (0, 0), (-1, 0), self.secondary_color),
                    # Define a cor do texto para a linha do cabeçalho como branco fumaça.
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    # Define a cor de fundo para as linhas de dados (a partir da segunda linha, índice 1, até o final) como um tom claro.
                    ('BACKGROUND', (0, 1), (-1, -1), self.light_bg),
                    # Define a cor do texto para as linhas de dados como um tom escuro.
                    ('TEXTCOLOR', (0, 1), (-1, -1), self.text_dark),
                    # Alinha todo o conteúdo da tabela (células) ao centro.
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    # Alinha verticalmente todo o conteúdo da tabela ao meio.
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    # Define a fonte para a linha do cabeçalho como negrito.
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    # Define o tamanho da fonte para todas as células da tabela como 10.
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    # Adiciona um preenchimento inferior de 8 unidades para todas as células.
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    # Adiciona um preenchimento superior de 8 unidades para todas as células.
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    # Adiciona uma grade a todas as células da tabela, com largura de linha de 0.75 e cor de borda definida.
                    ('GRID', (0, 0), (-1, -1), 0.75, self.border_color)
                ]))
            # Cria um parágrafo que serve como uma recomendação para o melhor modelo.
            recommendation = Paragraph(
                # Define o texto da recomendação, incluindo o nome do melhor modelo e a plataforma.
                f"<b>Recomendação:</b> O modelo <b>{self._safe_text(self.best_model_name)}</b> "
                f"é o mais indicado para uso em produção dentro do {self.PLATFORM_NAME}, "
                "pois apresentou o melhor desempenho entre os algoritmos avaliados.",
                # Aplica um estilo de texto personalizado (BodyTextCustom) ao parágrafo.
                self.styles['BodyTextCustom']
            )
            # Adiciona um bloco de elementos (tabela, espaço e recomendação) à lista principal de elementos do PDF.
            # O KeepTogether garante que esses elementos permaneçam juntos na mesma página, se possível.
            elements.append(KeepTogether([
                table,  # A tabela de métricas detalhadas do melhor modelo.
                Spacer(1, 10),  # Um espaço vertical de 10 unidades para separação.
                recommendation  # O parágrafo com a recomendação sobre o melhor modelo.
            ]))

        return elements
        def _create_ranking_table(self):
            """Cria tabela de ranking dos modelos""" # Docstring: Descreve o propósito da função.
            elements = [] # Inicializa uma lista vazia para armazenar os elementos do PDF.

            elements.append(Paragraph("RANKING COMPLETO DE MODELOS", self.styles['SectionTitle'])) # Adiciona um título de seção à lista de elementos.

            sorted_results = sorted( # Inicia a ordenação dos resultados dos modelos.
                self.results.items(), # Converte o dicionário de resultados em uma lista de itens (pares chave-valor).
                key=lambda x: self._get_primary_metric(x[1]), # Define a chave de ordenação usando uma função lambda que chama '_get_primary_metric' com as métricas do modelo.
                reverse=True # Ordena em ordem decrescente, de modo que o melhor desempenho fique no topo.
            )

            table_data = [[ # Inicializa a lista de dados para a tabela, começando com a linha do cabeçalho.
                Paragraph("Posição", self.styles['TableCellHeaderCustom']), # Adiciona o cabeçalho 'Posição' com estilo.
                Paragraph("Modelo", self.styles['TableCellHeaderCustom']), # Adiciona o cabeçalho 'Modelo' com estilo.
                Paragraph(self._metric_label_for_display(), self.styles['TableCellHeaderCustom']), # Adiciona o cabeçalho da métrica principal (ex: F1-Score, RMSE) com estilo.
                Paragraph("Status", self.styles['TableCellHeaderCustom']), # Adiciona o cabeçalho 'Status' com estilo.
            ]]

            for i, (model_name, metrics) in enumerate(sorted_results, 1): # Itera sobre os resultados ordenados, atribuindo um índice 'i' (posição).
                display_metric = self._get_display_metric(metrics) # Obtém o valor da métrica principal formatado para exibição.
                status = "⭐ RECOMENDADO" if model_name == self.best_model_name else "✅" # Define o status do modelo (recomendado ou padrão).

                table_data.append([ # Adiciona uma nova linha de dados à lista para a tabela.
                    Paragraph(str(i), self.styles['TableCellCustom']), # Adiciona a posição do modelo com estilo.
                    Paragraph(self._truncate_text(model_name, 42), self.styles['TableCellCustom']), # Adiciona o nome do modelo truncado com estilo.
                    Paragraph(display_metric, self.styles['TableCellCustom']), # Adiciona a métrica de exibição do modelo com estilo.
                    Paragraph(status, self.styles['TableCellCustom']), # Adiciona o status do modelo com estilo.
                ])

            table = Table( # Cria um objeto Table com os dados preparados.
                table_data, # Fornece os dados da tabela.
                colWidths=[0.75 * inch, 3.0 * inch, 1.3 * inch, 1.15 * inch], # Define as larguras das colunas.
                repeatRows=1 # Garante que a primeira linha (cabeçalho) se repita em novas páginas.
            )

        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (3, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.6, self.border_color),
        ])
        # Itera sobre as linhas da tabela, começando da segunda linha (índice 1), para aplicar estilos condicionalmente.
        for i, row in enumerate(table_data[1:], 1):
            # Extrai o texto da coluna 'Status' da linha atual. Verifica se o objeto tem o atributo 'text' para evitar erros.
            status_text = row[3].text if hasattr(row[3], 'text') else ""
            # Verifica se a palavra "RECOMENDADO" está presente no texto da coluna 'Status'.
            if "RECOMENDADO" in status_text:
            # Se for o modelo recomendado, define a cor de fundo da linha como um azul claro.
                table_style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#E8F4FD'))
                # Define a cor do texto para esta linha como um tom escuro.
                table_style.add('TEXTCOLOR', (0, i), (-1, i), self.text_dark)
                # Define a fonte para esta linha como negrito.
                table_style.add('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold')
                table.setStyle(table_style) # Aplica o estilo de tabela configurado 'table_style' à tabela.
                elements.append(table) # Adiciona a tabela de ranking à lista de elementos do PDF.

        return elements
        def _create_metrics_section(self):
            """Cria seção de métricas detalhadas""" # Comentário: Docstring para a função que cria a seção de métricas detalhadas.
            elements = [] # Inicializa uma lista vazia para armazenar os elementos do PDF.

            elements.append(Paragraph("ANÁLISE DETALHADA DE MÉTRICAS", self.styles['SectionTitle'])) # Adiciona um título de seção à lista de elementos.

            other_models = [ # Inicializa uma lista para armazenar os modelos que não são o melhor modelo.
                (model_name, metrics) # Tupla contendo o nome do modelo e suas métricas.
                for model_name, metrics in self.results.items() # Itera sobre todos os modelos e suas métricas nos resultados.
                if model_name != self.best_model_name # Filtra para incluir apenas os modelos que não são o melhor.
            ]

            if not other_models: # Verifica se não há outros modelos após a filtragem.
                elements.append(Paragraph( # Adiciona um parágrafo informando que não há modelos adicionais para exibir.
                    "Não há modelos adicionais para exibir nesta seção.", # Texto da mensagem.
                    self.styles['BodyTextCustom'] # Aplica o estilo de texto personalizado.
                ))
                return elements # Retorna a lista de elementos (com a mensagem de nenhum modelo adicional).

            for idx, (model_name, metrics) in enumerate(other_models, 1): # Itera sobre cada modelo restante com um índice.
                block = [] # Inicializa uma lista para armazenar elementos de um bloco de modelo.

                block.append(Paragraph( # Adiciona um parágrafo com o nome do modelo.
                    f"Modelo: {self._safe_text(model_name)}", # Formata o nome do modelo (seguro para texto).
                    self.styles['SubtitleCustom'] # Aplica o estilo de subtítulo personalizado.
                ))

                if self.problem_type == 'classification': # Verifica se o tipo de problema é classificação.
                    metrics_text = ( # Constrói uma string com métricas específicas para classificação.
                        f"Acurácia: {self._safe_float(metrics.get('accuracy', 0)):.4f} | " # Formata e adiciona a acurácia.
                        f"F1-Score: {self._safe_float(metrics.get('f1', 0)):.4f} | " # Formata e adiciona o F1-Score.
                        f"Precisão: {self._safe_float(metrics.get('precision', 0)):.4f}" # Formata e adiciona a precisão.
                    )
                else: # Se não for classificação, assume que é regressão.
                    metrics_text = ( # Constrói uma string com métricas específicas para regressão.
                        f"R²: {self._safe_float(metrics.get('r2', 0)):.4f} | " # Formata e adiciona o R².
                        f"RMSE: {self._safe_float(metrics.get('rmse', 0)):.4f} | " # Formata e adiciona o RMSE.
                        f"MAE: {self._safe_float(metrics.get('mae', 0)):.4f}" # Formata e adiciona o MAE.
                    )

                block.append(Paragraph(metrics_text, self.styles['BodyTextCustom'])) # Adiciona as métricas formatadas como um parágrafo ao bloco.
                block.append(Spacer(1, 6)) # Adiciona um espaço vertical após as métricas.

                elements.append(KeepTogether(block)) # Adiciona o bloco de modelo e suas métricas à lista principal de elementos, mantendo-os juntos na mesma página.

                if idx % 12 == 0: # Verifica se 12 modelos foram adicionados, para adicionar uma quebra de página.
                    elements.append(PageBreak()) # Adiciona uma quebra de página.

            return elements # Retorna a lista completa de elementos da seção de métricas.

        def _create_recommendations(self):
            """Cria seção de recomendações""" # Comentário: Docstring para a função que cria a seção de recomendações.
            elements = [] # Inicializa uma lista vazia para armazenar os elementos do PDF.

        elements.append(Paragraph("RECOMENDAÇÕES E PRÓXIMOS PASSOS", self.styles['SectionTitle']))

        recommendations = [
            "1. Implementar o modelo recomendado em ambiente de produção.",
            "2. Monitorar continuamente o desempenho após a implantação.",
            "3. Re-treinar o modelo periodicamente com novos dados.",
            "4. Validar os resultados em ambiente controlado antes do deploy definitivo.",
            "5. Documentar parâmetros, métricas e decisões para garantir reprodutibilidade.",
            "6. Avaliar interpretabilidade, monitoramento e governança do modelo.",
            "7. Considerar evolução futura com ensemble, explainability e automação contínua."
        ]
        # Itera sobre cada recomendação na lista 'recommendations'.
        for rec in recommendations:
            # Adiciona um parágrafo contendo a recomendação atual à lista de elementos do relatório.
            # O estilo 'BodyTextCustom' é aplicado a este parágrafo.
            elements.append(Paragraph(rec, self.styles['BodyTextCustom']))
            # Adiciona um espaço vertical de 3 unidades após cada recomendação para melhor legibilidade.
            elements.append(Spacer(1, 3))

        return elements
        def _create_footer(self):
            """Cria rodapé final"""
            elements = []  # Inicializa uma lista para armazenar os elementos do rodapé.

            elements.append(Spacer(1, 20))  # Adiciona um espaço vertical de 20 unidades.
            elements.append(Paragraph(  # Adiciona um parágrafo ao rodapé.
                f"Documento gerado automaticamente pelo {self.PLATFORM_NAME}.",  # Texto informativo sobre a origem do documento.
                self.styles['MutedText']  # Aplica o estilo 'MutedText' ao parágrafo.
            ))
            elements.append(Paragraph(  # Adiciona outro parágrafo ao rodapé.
                "Este relatório resume os principais resultados, métricas e recomendações obtidos durante a execução.",  # Texto de resumo do relatório.
                self.styles['FooterCustom']  # Aplica o estilo 'FooterCustom' ao parágrafo.
            ))

            return elements  # Retorna a lista de elementos que compõem o rodapé.

        def _get_primary_metric(self, metrics):
            """Obtém a métrica principal para ordenação"""
            if self.problem_type == 'classification':  # Verifica se o tipo de problema é classificação.
                return self._safe_float(metrics.get('f1', 0))  # Retorna o F1-Score para classificação, garantindo um float seguro.
            else:  # Se não for classificação, assume que é regressão.
                return -self._safe_float(metrics.get('rmse', 0))  # Retorna o negativo do RMSE para regressão (menor RMSE é melhor, então negativo o torna maior para ordenação decrescente).

        def _get_display_metric(self, metrics):
            """Obtém a métrica principal para exibição amigável"""
            if self.problem_type == 'classification':  # Verifica se o tipo de problema é classificação.
                value = self._safe_float(metrics.get('f1', 0))  # Obtém o valor do F1-Score de forma segura.
                return f"{value:.4f}"  # Formata o F1-Score para exibição com 4 casas decimais.
            else:  # Se não for classificação, assume que é regressão.
                value = self._safe_float(metrics.get('rmse', 0))  # Obtém o valor do RMSE de forma segura.
                return f"{value:.4f}"  # Formata o RMSE para exibição com 4 casas decimais.
