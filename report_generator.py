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
        elements = []

        elements.append(Paragraph("MELHOR MODELO IDENTIFICADO", self.styles['SectionTitle']))

        if self.best_model_name in self.results:
            metrics = self.results[self.best_model_name]

            if self.problem_type == 'classification':
                metrics_data = [
                    ["Métrica", "Valor"],
                    ["Acurácia", f"{self._safe_float(metrics.get('accuracy', 0)):.4f}"],
                    ["Precisão", f"{self._safe_float(metrics.get('precision', 0)):.4f}"],
                    ["Recall", f"{self._safe_float(metrics.get('recall', 0)):.4f}"],
                    ["F1-Score", f"{self._safe_float(metrics.get('f1', 0)):.4f}"],
                    ["ROC AUC", f"{self._safe_float(metrics.get('roc_auc', 0)):.4f}"],
                ]
            else:
                metrics_data = [
                    ["Métrica", "Valor"],
                    ["R² Score", f"{self._safe_float(metrics.get('r2', 0)):.4f}"],
                    ["RMSE", f"{self._safe_float(metrics.get('rmse', 0)):.4f}"],
                    ["MAE", f"{self._safe_float(metrics.get('mae', 0)):.4f}"],
                    ["MAPE", f"{self._safe_float(metrics.get('mape', 0)):.2f}%"],
                ]

            table = Table(metrics_data, repeatRows=1, colWidths=[2.2 * inch, 1.8 * inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.secondary_color),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('BACKGROUND', (0, 1), (-1, -1), self.light_bg),
                ('TEXTCOLOR', (0, 1), (-1, -1), self.text_dark),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.75, self.border_color)
            ]))

            recommendation = Paragraph(
                f"<b>Recomendação:</b> O modelo <b>{self._safe_text(self.best_model_name)}</b> "
                f"é o mais indicado para uso em produção dentro do {self.PLATFORM_NAME}, "
                "pois apresentou o melhor desempenho entre os algoritmos avaliados.",
                self.styles['BodyTextCustom']
            )

            elements.append(KeepTogether([
                table,
                Spacer(1, 10),
                recommendation
            ]))

        return elements

    def _create_ranking_table(self):
        """Cria tabela de ranking dos modelos"""
        elements = []

        elements.append(Paragraph("RANKING COMPLETO DE MODELOS", self.styles['SectionTitle']))

        sorted_results = sorted(
            self.results.items(),
            key=lambda x: self._get_primary_metric(x[1]),
            reverse=True
        )

        table_data = [[
            Paragraph("Posição", self.styles['TableCellHeaderCustom']),
            Paragraph("Modelo", self.styles['TableCellHeaderCustom']),
            Paragraph(self._metric_label_for_display(), self.styles['TableCellHeaderCustom']),
            Paragraph("Status", self.styles['TableCellHeaderCustom']),
        ]]

        for i, (model_name, metrics) in enumerate(sorted_results, 1):
            display_metric = self._get_display_metric(metrics)
            status = "⭐ RECOMENDADO" if model_name == self.best_model_name else "✅"

            table_data.append([
                Paragraph(str(i), self.styles['TableCellCustom']),
                Paragraph(self._truncate_text(model_name, 42), self.styles['TableCellCustom']),
                Paragraph(display_metric, self.styles['TableCellCustom']),
                Paragraph(status, self.styles['TableCellCustom']),
            ])

        table = Table(
            table_data,
            colWidths=[0.75 * inch, 3.0 * inch, 1.3 * inch, 1.15 * inch],
            repeatRows=1
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

        for i, row in enumerate(table_data[1:], 1):
            status_text = row[3].text if hasattr(row[3], 'text') else ""
            if "RECOMENDADO" in status_text:
                table_style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#E8F4FD'))
                table_style.add('TEXTCOLOR', (0, i), (-1, i), self.text_dark)
                table_style.add('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold')

        table.setStyle(table_style)
        elements.append(table)

        return elements

    def _create_metrics_section(self):
        """Cria seção de métricas detalhadas"""
        elements = []

        elements.append(Paragraph("ANÁLISE DETALHADA DE MÉTRICAS", self.styles['SectionTitle']))

        other_models = [
            (model_name, metrics)
            for model_name, metrics in self.results.items()
            if model_name != self.best_model_name
        ]

        if not other_models:
            elements.append(Paragraph(
                "Não há modelos adicionais para exibir nesta seção.",
                self.styles['BodyTextCustom']
            ))
            return elements

        for idx, (model_name, metrics) in enumerate(other_models, 1):
            block = []

            block.append(Paragraph(
                f"Modelo: {self._safe_text(model_name)}",
                self.styles['SubtitleCustom']
            ))

            if self.problem_type == 'classification':
                metrics_text = (
                    f"Acurácia: {self._safe_float(metrics.get('accuracy', 0)):.4f} | "
                    f"F1-Score: {self._safe_float(metrics.get('f1', 0)):.4f} | "
                    f"Precisão: {self._safe_float(metrics.get('precision', 0)):.4f}"
                )
            else:
                metrics_text = (
                    f"R²: {self._safe_float(metrics.get('r2', 0)):.4f} | "
                    f"RMSE: {self._safe_float(metrics.get('rmse', 0)):.4f} | "
                    f"MAE: {self._safe_float(metrics.get('mae', 0)):.4f}"
                )

            block.append(Paragraph(metrics_text, self.styles['BodyTextCustom']))
            block.append(Spacer(1, 6))

            elements.append(KeepTogether(block))

            if idx % 12 == 0:
                elements.append(PageBreak())

        return elements

    def _create_recommendations(self):
        """Cria seção de recomendações"""
        elements = []

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

        for rec in recommendations:
            elements.append(Paragraph(rec, self.styles['BodyTextCustom']))
            elements.append(Spacer(1, 3))

        return elements

    def _create_footer(self):
        """Cria rodapé final"""
        elements = []

        elements.append(Spacer(1, 20))
        elements.append(Paragraph(
            f"Documento gerado automaticamente pelo {self.PLATFORM_NAME}.",
            self.styles['MutedText']
        ))
        elements.append(Paragraph(
            "Este relatório resume os principais resultados, métricas e recomendações obtidos durante a execução.",
            self.styles['FooterCustom']
        ))

        return elements

    def _get_primary_metric(self, metrics):
        """Obtém a métrica principal para ordenação"""
        if self.problem_type == 'classification':
            return self._safe_float(metrics.get('f1', 0))
        else:
            return -self._safe_float(metrics.get('rmse', 0))

    def _get_display_metric(self, metrics):
        """Obtém a métrica principal para exibição amigável"""
        if self.problem_type == 'classification':
            value = self._safe_float(metrics.get('f1', 0))
            return f"{value:.4f}"
        else:
            value = self._safe_float(metrics.get('rmse', 0))
            return f"{value:.4f}"