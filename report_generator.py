from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime

class PDFReportGenerator:
    def __init__(self, results, models, best_model_name, problem_type, data_info=None):
        self.results = results
        self.models = models
        self.best_model_name = best_model_name
        self.problem_type = problem_type
        self.data_info = data_info or {}
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Configura estilos personalizados para o PDF"""
        self.styles.add(ParagraphStyle(
            name='Title',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='Heading2',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#3498db'),
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='NormalCenter',
            parent=self.styles['Normal'],
            alignment=1,  # Center
            spaceAfter=12
        ))
    
    def generate_report(self, filename="relatorio_ml.pdf"):
        """Gera o relatório PDF completo"""
        doc = SimpleDocTemplate(filename, pagesize=A4)
        story = []
        
        # Título
        story.append(Paragraph("RELATÓRIO DE MACHINE LEARNING", self.styles['Title']))
        story.append(Spacer(1, 20))
        
        # Data e informações gerais
        story.append(self._create_general_info())
        story.append(Spacer(1, 20))
        
        # Resumo executivo
        story.append(self._create_executive_summary())
        story.append(Spacer(1, 20))
        
        # Melhor modelo
        story.append(self._create_best_model_section())
        story.append(Spacer(1, 20))
        
        # Ranking completo
        story.append(self._create_ranking_table())
        story.append(Spacer(1, 20))
        
        # Métricas detalhadas
        story.append(self._create_metrics_section())
        story.append(Spacer(1, 20))
        
        # Recomendações
        story.append(self._create_recommendations())
        story.append(Spacer(1, 20))
        
        # Rodapé
        story.append(self._create_footer())
        
        # Construir PDF
        doc.build(story)
        print(f"Relatório gerado: {filename}")
        return filename
    
    def _create_general_info(self):
        """Cria seção de informações gerais"""
        elements = []
        
        elements.append(Paragraph("INFORMAÇÕES GERAIS", self.styles['Heading2']))
        
        info_data = [
            ["Data do Relatório:", datetime.now().strftime("%d/%m/%Y %H:%M")],
            ["Tipo de Problema:", self.problem_type.upper()],
            ["Total de Modelos:", str(len(self.models))],
            ["Melhor Modelo:", self.best_model_name],
            ["Status:", "PROCESSAMENTO COMPLETO"],
        ]
        
        if self.data_info:
            info_data.extend([
                ["Dataset:", self.data_info.get('dataset_name', 'N/A')],
                ["Amostras:", str(self.data_info.get('n_samples', 'N/A'))],
                ["Features:", str(self.data_info.get('n_features', 'N/A'))],
            ])
        
        table = Table(info_data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        elements.append(table)
        return elements
    
    def _create_executive_summary(self):
        """Cria resumo executivo"""
        elements = []
        
        elements.append(Paragraph("RESUMO EXECUTIVO", self.styles['Heading2']))
        
        summary_text = f"""
        Este relatório apresenta os resultados de um processo avançado de Machine Learning
        que envolveu o treinamento e avaliação de {len(self.models)} modelos diferentes.
        
        O sistema identificou automaticamente o problema como {self.problem_type.upper()} e aplicou
        técnicas sofisticadas de pré-processamento, seleção de features e otimização de hiperparâmetros.
        
        O modelo {self.best_model_name} demonstrou o melhor desempenho geral, sendo recomendado para
        implementação em produção.
        """
        
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        return elements
    
    def _create_best_model_section(self):
        """Cria seção detalhada do melhor modelo"""
        elements = []
        
        elements.append(Paragraph("MELHOR MODELO IDENTIFICADO", self.styles['Heading2']))
        
        if self.best_model_name in self.results:
            metrics = self.results[self.best_model_name]
            
            if self.problem_type == 'classification':
                metrics_data = [
                    ["Métrica", "Valor"],
                    ["Acurácia", f"{metrics.get('accuracy', 0):.4f}"],
                    ["Precisão", f"{metrics.get('precision', 0):.4f}"],
                    ["Recall", f"{metrics.get('recall', 0):.4f}"],
                    ["F1-Score", f"{metrics.get('f1', 0):.4f}"],
                    ["ROC AUC", f"{metrics.get('roc_auc', 0):.4f}"],
                ]
            else:
                metrics_data = [
                    ["Métrica", "Valor"],
                    ["R² Score", f"{metrics.get('r2', 0):.4f}"],
                    ["RMSE", f"{metrics.get('rmse', 0):.4f}"],
                    ["MAE", f"{metrics.get('mae', 0):.4f}"],
                    ["MAPE", f"{metrics.get('mape', 0):.2f}%"],
                ]
            
            table = Table(metrics_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            
            elements.append(table)
            
            # Adicionar recomendação
            recommendation = Paragraph(
                f"<b>Recomendação:</b> O modelo {self.best_model_name} deve ser utilizado em produção "
                "devido ao seu desempenho superior em todas as métricas avaliadas.",
                self.styles['Normal']
            )
            elements.append(Spacer(1, 12))
            elements.append(recommendation)
        
        return elements
    
    def _create_ranking_table(self):
        """Cria tabela de ranking dos modelos"""
        elements = []
        
        elements.append(Paragraph("RANKING COMPLETO DE MODELOS", self.styles['Heading2']))
        
        # Ordenar modelos
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: self._get_primary_metric(x[1]), 
                               reverse=True)
        
        # Preparar dados da tabela
        table_data = [["Posição", "Modelo", "Métrica Principal", "Status"]]
        
        for i, (model_name, metrics) in enumerate(sorted_results, 1):
            primary_metric = self._get_primary_metric(metrics)
            status = "⭐ RECOMENDADO" if model_name == self.best_model_name else "✅"
            
            table_data.append([
                str(i),
                model_name,
                f"{primary_metric:.4f}",
                status
            ])
        
        table = Table(table_data, colWidths=[0.8*inch, 2.5*inch, 1.5*inch, 1.2*inch])
        
        # Estilizar tabela
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ])
        
        # Destacar linha do melhor modelo
        for i, row in enumerate(table_data[1:], 1):
            if row[3] == "⭐ RECOMENDADO":
                table_style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f39c12'))
                table_style.add('TEXTCOLOR', (0, i), (-1, i), colors.white)
                table_style.add('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold')
        
        table.setStyle(table_style)
        elements.append(table)
        
        return elements
    
    def _create_metrics_section(self):
        """Cria seção de métricas detalhadas"""
        elements = []
        
        elements.append(Paragraph("ANÁLISE DETALHADA DE MÉTRICAS", self.styles['Heading2']))
        
        # Para cada modelo, mostrar métricas principais
        for model_name, metrics in self.results.items():
            if model_name == self.best_model_name:
                continue  # Já mostramos o melhor modelo
            
            elements.append(Paragraph(f"Modelo: {model_name}", 
                                     ParagraphStyle(name='Subtitle', 
                                                   fontSize=12, 
                                                   textColor=colors.HexColor('#7f8c8d'),
                                                   spaceAfter=6)))
            
            if self.problem_type == 'classification':
                metrics_text = f"""
                Acurácia: {metrics.get('accuracy', 0):.4f} | 
                F1-Score: {metrics.get('f1', 0):.4f} | 
                Precisão: {metrics.get('precision', 0):.4f}
                """
            else:
                metrics_text = f"""
                R²: {metrics.get('r2', 0):.4f} | 
                RMSE: {metrics.get('rmse', 0):.4f} | 
                MAE: {metrics.get('mae', 0):.4f}
                """
            
            elements.append(Paragraph(metrics_text, self.styles['Normal']))
            elements.append(Spacer(1, 8))
        
        return elements
    
    def _create_recommendations(self):
        """Cria seção de recomendações"""
        elements = []
        
        elements.append(Paragraph("RECOMENDAÇÕES E PRÓXIMOS PASSOS", self.styles['Heading2']))
        
        recommendations = [
            "1. Implementar o modelo recomendado em ambiente de produção",
            "2. Monitorar continuamente o desempenho do modelo",
            "3. Re-treinar o modelo periodicamente com novos dados",
            "4. Considerar técnicas de ensemble para melhorar ainda mais a performance",
            "5. Validar os resultados com testes A/B em produção",
            "6. Documentar todo o processo para reprodutibilidade",
            "7. Considerar técnicas de explainable AI (XAI) para interpretabilidade"
        ]
        
        for rec in recommendations:
            elements.append(Paragraph(rec, self.styles['Normal']))
            elements.append(Spacer(1, 4))
        
        return elements
    
    def _create_footer(self):
        """Cria rodapé do relatório"""
        elements = []
        
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("=" * 80, 
                                 ParagraphStyle(name='Line', alignment=1)))
        
        footer_text = """
        Relatório gerado automaticamente pelo Sistema Avançado de Machine Learning
        Para mais informações ou suporte técnico, consulte a documentação do sistema
        """
        
        elements.append(Paragraph(footer_text, 
                                 ParagraphStyle(name='Footer', 
                                               fontSize=8, 
                                               textColor=colors.grey,
                                               alignment=1)))
        
        return elements
    
    def _get_primary_metric(self, metrics):
        """Obtém a métrica principal para ordenação"""
        if self.problem_type == 'classification':
            return metrics.get('f1', 0)
        else:
            return -metrics.get('rmse', 0)  # Negativo porque menor RMSE é melhor