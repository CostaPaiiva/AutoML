#Este código é responsável por:

# 1. Treinar dezenas de modelos automaticamente

# Classificação ou regressão

# Mais de 30 algoritmos diferentes

# Inclui modelos lineares, árvores, ensembles, boosting e redes neurais

# 2. Avaliar corretamente cada modelo

# Usa validação cruzada

# Calcula métricas completas:

# classificação: accuracy, precision, recall, f1, roc_auc

# regressão: RMSE, MAE, R², MAPE

# 3. Ranqueia os modelos

# Define uma métrica principal

# Ordena do melhor para o pior

# Mantém rastreabilidade total dos resultados

# 4. Cria ensembles automaticamente

# Seleciona os melhores modelos

# Cria VotingClassifier / VotingRegressor

# Usa pesos baseados em performance real

# 5. Otimiza hiperparâmetros com Optuna

# Aplica otimização inteligente

# Foca nos modelos mais promissores

# Re-treina e reavalia

# 6. Escolhe o melhor modelo final

# Baseado em métrica objetiva

# Sem viés manual

# Pronto para produção

# Importa a biblioteca NumPy para operações matemáticas e manipulação de arrays
import numpy as np

# Importa a biblioteca Pandas para manipulação de dados em DataFrames
import pandas as pd

# Importa funções de divisão de dados e validação cruzada do scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
# train_test_split: divide dados em treino e teste
# cross_val_score: calcula métricas usando validação cruzada
# StratifiedKFold: validação cruzada estratificada (mantém proporção de classes)
# KFold: validação cruzada simples

# Importa o LabelEncoder para transformar variáveis categóricas em números
from sklearn.preprocessing import LabelEncoder

# Importa warnings para controlar mensagens de aviso
import warnings
warnings.filterwarnings('ignore')  # Ignora avisos para não poluir a saída

# Importa joblib para salvar e carregar modelos treinados
import joblib

# Importa optuna para otimização automática de hiperparâmetros
import optuna

# Importa todas as métricas do scikit-learn (accuracy, f1, mse, etc.)
from sklearn.metrics import *

# Importa o XGBoost, biblioteca de gradient boosting altamente eficiente
import xgboost as xgb

# Importa o LightGBM, outra biblioteca de gradient boosting otimizada para velocidade
import lightgbm as lgb

# Importa o CatBoost, biblioteca de boosting que lida bem com variáveis categóricas
from catboost import CatBoostClassifier, CatBoostRegressor
# CatBoostClassifier: para problemas de classificação
# CatBoostRegressor: para problemas de regressão

# Importa VotingClassifier e VotingRegressor para combinar múltiplos modelos
from sklearn.ensemble import VotingClassifier, VotingRegressor
# VotingClassifier: combina classificadores (ensemble)
# VotingRegressor: combina regressões (ensemble)

# Importa GridSearchCV e RandomizedSearchCV para busca de hiperparâmetros
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# GridSearchCV: busca exaustiva em grade
# RandomizedSearchCV: busca aleatória mais rápida

# Define a classe principal para treinar modelos avançados
class AdvancedModelTrainer:
    def __init__(self, problem_type):
        self.problem_type = problem_type        # Tipo de problema: 'classification' ou 'regression'
        self.models = {}                        # Dicionário para armazenar modelos treinados
        self.results = {}                       # Dicionário para armazenar resultados de avaliação
        self.best_model = None                  # Melhor modelo encontrado
        self.best_model_name = ""               # Nome do melhor modelo
        self.feature_importance = {}            # Importância das features (quando disponível)

        
    def get_all_models(self):
        """Retorna mais de 30 modelos de ML"""
        # Verifica se o problema é de classificação
        if self.problem_type == 'classification':
            # Importa modelos lineares
            from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
            # Importa modelos de máquinas de vetores de suporte (SVM)
            from sklearn.svm import SVC, NuSVC, LinearSVC
            # Importa modelos baseados em vizinhos mais próximos
            from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
            # Importa modelos baseados em árvores de decisão
            from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
            # Importa modelos ensemble (conjuntos de modelos)
            from sklearn.ensemble import (
                RandomForestClassifier, GradientBoostingClassifier, 
                AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                HistGradientBoostingClassifier
            )
            # Importa modelos Naive Bayes
            from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
            # Importa modelos de análise discriminante
            from sklearn.discriminant_analysis import (
                LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
            )
            # Importa redes neurais artificiais
            from sklearn.neural_network import MLPClassifier
            
            # Define um dicionário com os modelos disponíveis para classificação
            models = {
                # Modelos lineares
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'RidgeClassifier': RidgeClassifier(random_state=42),
                'SGDClassifier': SGDClassifier(random_state=42),
                
                # SVM
                'SVC': SVC(probability=True, random_state=42),
                'NuSVC': NuSVC(probability=True, random_state=42),
                'LinearSVC': LinearSVC(random_state=42),
                
                # Vizinhos
                'KNeighborsClassifier': KNeighborsClassifier(),
                'RadiusNeighborsClassifier': RadiusNeighborsClassifier(),
                
                # Árvores
                'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
                'ExtraTreeClassifier': ExtraTreeClassifier(random_state=42),
                
                # Ensemble
                'RandomForestClassifier': RandomForestClassifier(random_state=42, n_estimators=100),
                'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
                'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
                'BaggingClassifier': BaggingClassifier(random_state=42),
                'ExtraTreesClassifier': ExtraTreesClassifier(random_state=42),
                'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state=42),
                
                # Naive Bayes
                'GaussianNB': GaussianNB(),
                'BernoulliNB': BernoulliNB(),
                'MultinomialNB': MultinomialNB(),
                
                # Análise Discriminante
                'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
                'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
                
                # Redes Neurais
                'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000),
                
                # Gradient Boosting Avançado
                'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
                
                # Modelos Ensemble Avançados
                'VotingClassifier': None  # Será criado posteriormente
            }
            
        else:  # Regression
            # Importa modelos lineares para regressão
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
            # Importa modelos de máquinas de vetores de suporte (SVM) para regressão
            from sklearn.svm import SVR, NuSVR, LinearSVR
            # Importa modelos baseados em vizinhos mais próximos para regressão
            from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
            # Importa modelos baseados em árvores de decisão para regressão
            from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
            # Importa modelos ensemble (conjuntos de modelos) para regressão
            from sklearn.ensemble import (
                RandomForestRegressor, GradientBoostingRegressor, 
                AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                HistGradientBoostingRegressor
            )
            # Importa Kernel Ridge para regressão
            from sklearn.kernel_ridge import KernelRidge
            # Importa redes neurais artificiais para regressão
            from sklearn.neural_network import MLPRegressor
            
            # Define um dicionário com os modelos disponíveis para regressão
            models = {
                # Modelos lineares
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'ElasticNet': ElasticNet(random_state=42),
                'SGDRegressor': SGDRegressor(random_state=42),
                
                # SVM
                'SVR': SVR(),
                'NuSVR': NuSVR(),
                'LinearSVR': LinearSVR(random_state=42),
                
                # Vizinhos
                'KNeighborsRegressor': KNeighborsRegressor(),
                'RadiusNeighborsRegressor': RadiusNeighborsRegressor(),
                
                # Árvores
                'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
                'ExtraTreeRegressor': ExtraTreeRegressor(random_state=42),
                
                # Ensemble
                'RandomForestRegressor': RandomForestRegressor(random_state=42, n_estimators=100),
                'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
                'AdaBoostRegressor': AdaBoostRegressor(random_state=42),
                'BaggingRegressor': BaggingRegressor(random_state=42),
                'ExtraTreesRegressor': ExtraTreesRegressor(random_state=42),
                'HistGradientBoostingRegressor': HistGradientBoostingRegressor(random_state=42),
                
                # Kernel
                'KernelRidge': KernelRidge(),
                
                # Redes Neurais
                'MLPRegressor': MLPRegressor(random_state=42, max_iter=1000),
                
                # Gradient Boosting Avançado
                'XGBoost': xgb.XGBRegressor(random_state=42),
                'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
                'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
                
                # Modelos Ensemble Avançados
                'VotingRegressor': None  # Será criado posteriormente
            }
        
        # Retorna todos os modelos disponíveis para classificação ou regressão
        return models

    # Função para otimizar hiperparâmetros de um modelo usando Optuna
    def optimize_with_optuna(self, model_name, X_train, y_train, n_trials=50):
        """Otimização hiperparâmetros com Optuna"""
        print(f"Otimizando {model_name} com Optuna...")

        # Define a função objetivo para a otimização
        def objective(trial):
            # Verifica se o problema é de classificação
            if self.problem_type == 'classification':
                # Configura os hiperparâmetros para o modelo XGBoost
                if model_name == 'XGBoost':
                    # Define os hiperparâmetros a serem otimizados
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),  # Número de árvores
                        'max_depth': trial.suggest_int('max_depth', 3, 10),  # Profundidade máxima da árvore
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # Taxa de aprendizado
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Proporção de amostras usadas para cada árvore
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Proporção de features usadas por árvore
                    }
                    # Cria o modelo XGBoost com os parâmetros sugeridos
                    model = xgb.XGBClassifier(**param, random_state=42, use_label_encoder=False, eval_metric='logloss')
                
                # Configura os hiperparâmetros para o modelo RandomForestClassifier
                elif model_name == 'RandomForestClassifier':
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),  # Número de árvores
                        'max_depth': trial.suggest_int('max_depth', 3, 20),  # Profundidade máxima
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),  # Mínimo de amostras para dividir
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),  # Mínimo de amostras por folha
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  # Seleção de features
                    }
                    # Cria o modelo RandomForestClassifier com os parâmetros sugeridos e uma semente aleatória para reprodutibilidade
                    model = RandomForestClassifier(**param, random_state=42)
                
            else:  # Regression
                # Configuração de hiperparâmetros para o modelo XGBoost
                if model_name == 'XGBoost':
                    # Define os hiperparâmetros a serem otimizados
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),  # Número de árvores
                        'max_depth': trial.suggest_int('max_depth', 3, 10),  # Profundidade máxima da árvore
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # Taxa de aprendizado
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Proporção de amostras usadas para cada árvore
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Proporção de features usadas por árvore
                    }
                    model = xgb.XGBRegressor(**param, random_state=42)
                    
                # Configuração de hiperparâmetros para o modelo RandomForestRegressor
                elif model_name == 'RandomForestRegressor':
                    # Define os hiperparâmetros a serem otimizados
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),  # Número de árvores
                        'max_depth': trial.suggest_int('max_depth', 3, 20),  # Profundidade máxima da árvore
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),  # Mínimo de amostras para dividir um nó
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),  # Mínimo de amostras em uma folha
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  # Estratégia de seleção de features
                    }
                    # Cria um modelo RandomForestRegressor com os parâmetros sugeridos e uma semente aleatória para reprodutibilidade
                    model = RandomForestRegressor(**param, random_state=42)

            # Realiza validação cruzada com 3 divisões, utilizando a métrica de avaliação definida e paralelizando o processo
            score = cross_val_score(model, X_train, y_train, 
                                   cv=3, scoring=self.get_scoring_metric(), n_jobs=-1)
            # Retorna a média das pontuações obtidas na validação cruzada
            return score.mean()
        
        # Cria um estudo Optuna para maximizar a métrica de avaliação
        study = optuna.create_study(direction='maximize')
        # Otimiza os hiperparâmetros do modelo com base na função objetivo definida
        study.optimize(objective, n_trials=n_trials)
        
        # Retorna os melhores parâmetros encontrados pelo estudo
        return study.best_params
    
    # Define a métrica de avaliação com base no tipo de problema (classificação ou regressão)
    def get_scoring_metric(self):
        """Retorna a métrica de avaliação baseada no tipo de problema"""
        if self.problem_type == 'classification':
            # Para classificação, utiliza a métrica F1 ponderada
            return 'f1_weighted'  # ou 'accuracy', 'roc_auc', etc.
        else:
            # Para regressão, utiliza o erro quadrático médio negativo
            return 'neg_root_mean_squared_error'
    
    def train_models(self, X, y, optimize_top_n=5):
        """Treina todos os modelos com validação cruzada"""
        # Função principal para treinar todos os modelos com validação cruzada
        print(f"Iniciando treinamento de modelos ({self.problem_type})...")
        
        # Divisão dos dados em treino e teste
        if self.problem_type == 'classification':
            # Para classificação, utiliza divisão estratificada para manter proporção das classes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            # Para regressão, divisão simples dos dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Obter todos os modelos disponíveis para o tipo de problema
        all_models = self.get_all_models()
        
        # Loop para treinar cada modelo individualmente
        for name, model in all_models.items():
            if model is not None:  # Ignorar modelos que ainda não foram criados (ex: ensembles)
                try:
                    print(f"Treinando {name}...")
                    
                    # Configuração de validação cruzada
                    if self.problem_type == 'classification':
                        # Para classificação, utiliza validação cruzada estratificada
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    else:
                        # Para regressão, utiliza validação cruzada simples
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    
                    # Calcula as métricas de validação cruzada
                    cv_scores = cross_val_score(model, X_train, y_train, 
                                               cv=cv, scoring=self.get_scoring_metric(),
                                               n_jobs=-1)
                    
                    # Treina o modelo no conjunto completo de treino
                    model.fit(X_train, y_train)
                    
                    # Realiza previsões no conjunto de teste
                    y_pred = model.predict(X_test)
                    
                    # Calcular métricas do modelo treinado no conjunto de teste
                    metrics = self.calculate_metrics(y_test, y_pred)
                    # Adiciona a média e o desvio padrão das métricas de validação cruzada
                    metrics['cv_mean'] = cv_scores.mean()  # Média das pontuações da validação cruzada
                    metrics['cv_std'] = cv_scores.std()  # Desvio padrão das pontuações da validação cruzada
                    
                    # Armazena o modelo treinado no dicionário de modelos
                    self.models[name] = model
                    # Armazena os resultados de avaliação no dicionário de resultados
                    self.results[name] = metrics
                    
                    # Verifica se o modelo possui importância das features e armazena
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = model.feature_importances_
                    
                    # Exibe a métrica principal do modelo treinado
                    print(f"{name}: {self.get_primary_metric(metrics)}")
                    
                except Exception as e:
                    # Captura e exibe erros durante o treinamento do modelo
                    print(f"Erro ao treinar {name}: {str(e)}")
        
        # Cria um ensemble dos melhores modelos treinados
        self.create_ensemble(X_train, y_train, X_test, y_test)
        
        # Otimiza os hiperparâmetros dos top N modelos mais promissores
        if optimize_top_n > 0:
            self.optimize_top_models(optimize_top_n, X_train, y_train, X_test, y_test)
        
        # Determina o melhor modelo com base na métrica principal
        self.determine_best_model()
        
        # Retorna os resultados de avaliação e o nome do melhor modelo encontrado
        return self.results, self.best_model_name
    
    # Calcular métricas do modelo treinado no conjunto de teste
    def calculate_metrics(self, y_true, y_pred):
        """Calcula todas as métricas relevantes"""
        metrics = {}

        if self.problem_type == 'classification':
            # Métricas para problemas de classificação
            metrics['accuracy'] = accuracy_score(y_true, y_pred)  # Acurácia
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')  # Precisão ponderada
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')  # Recall ponderado
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')  # F1 ponderado
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred, multi_class='ovr') if len(np.unique(y_true)) > 2 else roc_auc_score(y_true, y_pred)  # AUC ROC

            # Matriz de confusão (será usada posteriormente)
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm

        else:  # Regressão
            # Métricas para problemas de regressão
            metrics['mse'] = mean_squared_error(y_true, y_pred)  # Erro quadrático médio
            metrics['rmse'] = np.sqrt(metrics['mse'])  # Raiz do erro quadrático médio
            metrics['mae'] = mean_absolute_error(y_true, y_pred)  # Erro absoluto médio
            metrics['r2'] = r2_score(y_true, y_pred)  # Coeficiente de determinação R²
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-10, None))) * 100  # Erro percentual absoluto médio

        return metrics
    
    # Função para obter a métrica principal usada para ranquear os modelos
    def get_primary_metric(self, metrics):
        """Retorna a métrica principal para ranking"""
        if self.problem_type == 'classification':
            return metrics['f1']  # Para classificação, utiliza a métrica F1
        else:
            return -metrics['rmse']  # Para regressão, utiliza o RMSE negativo (maximização)

    # Função para criar um ensemble dos melhores modelos treinados
    def create_ensemble(self, X_train, y_train, X_test, y_test):
        """Cria ensemble dos melhores modelos"""
        print("Criando ensemble dos melhores modelos...")
        
        # Seleciona os top 5 modelos com base na métrica principal
        sorted_models = sorted(self.results.items(), 
                              key=lambda x: self.get_primary_metric(x[1]), 
                              reverse=True)[:5]
        
        ensemble_models = []  # Lista para armazenar os modelos do ensemble
        weights = []  # Lista para armazenar os pesos dos modelos
        
        for name, metrics in sorted_models:
            if name in self.models:
                ensemble_models.append((name, self.models[name]))  # Adiciona o modelo ao ensemble
                weights.append(self.get_primary_metric(metrics))  # Adiciona o peso baseado na métrica principal
        
        if len(ensemble_models) >= 3:  # Cria o ensemble apenas se houver pelo menos 3 modelos
            # Normaliza os pesos para que a soma seja 1
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            if self.problem_type == 'classification':
                # Cria um VotingClassifier para problemas de classificação
                ensemble = VotingClassifier(
                    estimators=ensemble_models,
                    voting='soft',
                    weights=weights.tolist()
                )
            else:
                # Cria um VotingRegressor para problemas de regressão
                ensemble = VotingRegressor(
                    estimators=ensemble_models,
                    weights=weights.tolist()
                )
            
            # Treina o ensemble no conjunto de treino
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)  # Faz previsões no conjunto de teste
            
            # Calcula as métricas do ensemble
            metrics = self.calculate_metrics(y_test, y_pred)
            
            self.models['Ensemble'] = ensemble  # Armazena o ensemble nos modelos
            self.results['Ensemble'] = metrics  # Armazena os resultados do ensemble
            
            print(f"Ensemble criado com {len(ensemble_models)} modelos")
    
    # Função para otimizar os N melhores modelos usando Optuna
    def optimize_top_models(self, n_models, X_train, y_train, X_test, y_test):
        """Otimiza os N melhores modelos"""
        print(f"Otimizando os {n_models} melhores modelos...")
        
        # Ordena os modelos com base na métrica principal em ordem decrescente e seleciona os N melhores
        sorted_models = sorted(self.results.items(), 
                              key=lambda x: self.get_primary_metric(x[1]), 
                              reverse=True)[:n_models]
        
        # Loop que percorre todos os modelos já ordenados (provavelmente por desempenho ou score)
        for name, _ in sorted_models:
            
            # Verifica se o nome do modelo atual está entre os modelos específicos:
            # XGBoost, RandomForestClassifier ou RandomForestRegressor
            if name in ['XGBoost', 'RandomForestClassifier', 'RandomForestRegressor']:
                
                try:
                    # Início de um bloco "try": aqui o código vai tentar executar alguma operação
                    # especial para esses modelos (como calcular importância das features).
                    # Se ocorrer erro, o "except" correspondente vai tratar.

                    # Otimiza os hiperparâmetros do modelo usando Optuna
                    best_params = self.optimize_with_optuna(name, X_train, y_train, n_trials=30)
                    
                    # Recria o modelo com os melhores parâmetros encontrados
                    model_class = type(self.models[name])
                    optimized_model = model_class(**best_params, random_state=42)
                    
                    if 'XGB' in name and self.problem_type == 'classification':
                        # Configurações específicas para XGBoost em classificação
                        optimized_model.set_params(use_label_encoder=False, eval_metric='logloss')
                    
                    # Treina o modelo otimizado
                    optimized_model.fit(X_train, y_train)
                    y_pred = optimized_model.predict(X_test)  # Faz previsões no conjunto de teste
                    
                    # Calcula as métricas do modelo otimizado
                    metrics = self.calculate_metrics(y_test, y_pred)
                    
                    # Atualiza os resultados com o modelo otimizado
                    self.models[f'{name}_Optimized'] = optimized_model
                    self.results[f'{name}_Optimized'] = metrics
                    
                    print(f"{name} otimizado: {self.get_primary_metric(metrics)}")
                    
                except Exception as e:
                    # Captura e exibe erros durante a otimização
                    print(f"Erro ao otimizar {name}: {str(e)}")
    
    # Função para determinar o melhor modelo com base nas métricas
    def determine_best_model(self):
        """Determina o melhor modelo baseado nas métricas"""
        if not self.results:  # Verifica se há resultados disponíveis
            return
        
        # Seleciona o modelo com a melhor métrica principal
        best_model_name = max(self.results.items(), 
                             key=lambda x: self.get_primary_metric(x[1]))[0]
        
        self.best_model_name = best_model_name  # Armazena o nome do melhor modelo
        self.best_model = self.models.get(best_model_name)  # Armazena o melhor modelo
        
        print(f"\n⭐ MELHOR MODELO: {best_model_name}")
        print(f"Métrica principal: {self.get_primary_metric(self.results[best_model_name]):.4f}")
    
    # Função para obter os modelos ranqueados do melhor para o pior
    def get_ranked_models(self):
        """Retorna modelos ordenados do melhor para o pior"""
        # Ordena os modelos com base na métrica principal
        ranked = sorted(
            self.results.items(),   # Pega todos os itens do dicionário self.results (modelo -> métricas)

            # Define a chave de ordenação: para cada item (x),
            # aplica a função self.get_primary_metric(x[1]),
            # que extrai a métrica principal do dicionário de métricas do modelo.
            key=lambda x: self.get_primary_metric(x[1]),

            # reverse=True significa que a ordenação será em ordem decrescente,
            # ou seja, os melhores modelos (maior valor da métrica) ficam no topo.
            reverse=True
        )

        
        # Cria um DataFrame com os modelos e suas métricas
        # Esse DataFrame será construído a partir de uma lista de dicionários
        ranking_df = pd.DataFrame([
            
            # Para cada modelo avaliado, cria um dicionário com informações relevantes
            {
                'Modelo': name,   # Nome do modelo (ex: RandomForest, XGBoost, etc.)
                
                # Chama a função self.get_primary_metric(metrics) para extrair a métrica principal
                # (ex: accuracy para classificação ou RMSE para regressão)
                'Métrica Principal': self.get_primary_metric(metrics),
                
                # Armazena todas as métricas calculadas para o modelo (ex: accuracy, f1, recall, etc.)
                'Detalhes': metrics
            }
            
            # Loop que percorre a lista ranked, que contém pares (name, metrics)
            # name = nome do modelo, metrics = dicionário de métricas desse modelo
            for name, metrics in ranked
        ])

        return ranking_df  # Retorna o DataFrame com o ranking dos modelos
    
    # Função para salvar os modelos treinados e seus resultados
    def save_models(self, path='models/'):
        """Salva todos os modelos treinados"""
        import os
        os.makedirs(path, exist_ok=True)  # Cria o diretório se não existir
        
        # Loop que percorre todos os modelos armazenados no dicionário self.models
        for name, model in self.models.items():
            
            # Salva cada modelo em um arquivo .pkl usando joblib
            # O nome do arquivo será o nome do modelo, dentro do diretório especificado em 'path'
            joblib.dump(model, f'{path}/{name}.pkl')
            
            # Exibe mensagem confirmando que o modelo foi salvo com sucesso
            print(f"Modelo {name} salvo em {path}/{name}.pkl")
        
        # Converte o dicionário de resultados (self.results) em um DataFrame do pandas
        # O .T (transpose) é usado para inverter linhas e colunas, deixando os resultados organizados
        results_df = pd.DataFrame(self.results).T
        
        # Salva os resultados em um arquivo CSV dentro do diretório 'path'
        results_df.to_csv(f'{path}/model_results.csv')

        
        # Exibe uma mensagem no terminal informando ao usuário
        # que os resultados foram salvos em um arquivo CSV.
        # O caminho do arquivo é construído dinamicamente usando a variável 'path'.
        print(f"Resultados salvos em {path}/model_results.csv")
