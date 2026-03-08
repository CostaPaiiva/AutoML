# Este código é responsável por:
# 1. Treinar dezenas de modelos automaticamente
# 2. Avaliar corretamente cada modelo
# 3. Ranqueia os modelos
# 4. Cria ensembles automaticamente
# 5. Otimiza hiperparâmetros com Optuna
# 6. Escolhe o melhor modelo final

# Importa a biblioteca NumPy para operações numéricas
import numpy as np
# Importa a biblioteca Pandas para manipulação e análise de dados
import pandas as pd
# Importa a biblioteca warnings para controlar avisos
import warnings
# Ignora todos os avisos para manter a saída limpa
warnings.filterwarnings('ignore')

# Importa a biblioteca joblib para serialização de objetos Python (salvar e carregar modelos)
import joblib
# Importa a biblioteca Optuna para otimização de hiperparâmetros
import optuna
# Importa a biblioteca XGBoost para modelos de gradient boosting
import xgboost as xgb
# Importa a biblioteca LightGBM para modelos de gradient boosting
import lightgbm as lgb

# Importa as classes CatBoostClassifier e CatBoostRegressor da biblioteca CatBoost
from catboost import CatBoostClassifier, CatBoostRegressor
# Importa todas as métricas de avaliação do scikit-learn
from sklearn.metrics import *
# Importa funções para divisão de dados, validação cruzada e criação de folds do scikit-learn
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold
)
# Importa a classe LabelEncoder para codificar rótulos categóricos
from sklearn.preprocessing import LabelEncoder
# Importa as classes VotingClassifier e VotingRegressor para criar modelos de ensemble
from sklearn.ensemble import VotingClassifier, VotingRegressor
# Importa as classes GridSearchCV e RandomizedSearchCV para busca de hiperparâmetros
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Classe para treinar modelos de Machine Learning de forma avançada
class AdvancedModelTrainer:
    # Construtor da classe, inicializa atributos
    def __init__(self, problem_type):
        # Define o tipo de problema (classificação ou regressão)
        self.problem_type = problem_type
        # Dicionário para armazenar os modelos treinados
        self.models = {}
        # Dicionário para armazenar os resultados de avaliação de cada modelo
        self.results = {}
        # Armazena o melhor modelo encontrado
        self.best_model = None
        # Armazena o nome do melhor modelo encontrado
        self.best_model_name = ""
        # Dicionário para armazenar a importância das features de cada modelo
        self.feature_importance = {}

    # Método para obter todos os modelos disponíveis
    def get_all_models(self):
        """Retorna mais de 30 modelos de ML"""
        # Verifica se o problema é de classificação
        if self.problem_type == 'classification':
            # Importa modelos de classificação do scikit-learn
            from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
            from sklearn.svm import SVC, NuSVC, LinearSVC
            from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
            from sklearn.ensemble import (
                RandomForestClassifier, GradientBoostingClassifier,
                AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                HistGradientBoostingClassifier
            )
            from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
            from sklearn.discriminant_analysis import (
                LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
            )
            from sklearn.neural_network import MLPClassifier

            # Dicionário de modelos de classificação
            models = {
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'RidgeClassifier': RidgeClassifier(random_state=42),
                'SGDClassifier': SGDClassifier(random_state=42),

                'SVC': SVC(probability=True, random_state=42),
                'NuSVC': NuSVC(probability=True, random_state=42),
                'LinearSVC': LinearSVC(random_state=42),

                'KNeighborsClassifier': KNeighborsClassifier(),
                'RadiusNeighborsClassifier': RadiusNeighborsClassifier(),

                'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
                'ExtraTreeClassifier': ExtraTreeClassifier(random_state=42),

                'RandomForestClassifier': RandomForestClassifier(random_state=42, n_estimators=100),
                'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
                'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
                'BaggingClassifier': BaggingClassifier(random_state=42),
                'ExtraTreesClassifier': ExtraTreesClassifier(random_state=42),
                'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state=42),

                'GaussianNB': GaussianNB(),
                'BernoulliNB': BernoulliNB(),
                'MultinomialNB': MultinomialNB(),

                'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
                'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),

                'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000),

                'XGBoost': xgb.XGBClassifier(
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                ),
                'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'CatBoost': CatBoostClassifier(random_state=42, verbose=0),

                'VotingClassifier': None # Placeholder para ensemble
            }

        # Caso contrário, o problema é de regressão
        else:
            # Importa modelos de regressão do scikit-learn
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
            from sklearn.svm import SVR, NuSVR, LinearSVR
            from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
            from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
            from sklearn.ensemble import (
                RandomForestRegressor, GradientBoostingRegressor,
                AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                HistGradientBoostingRegressor
            )
            from sklearn.kernel_ridge import KernelRidge
            from sklearn.neural_network import MLPRegressor

            # Dicionário de modelos de regressão
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'ElasticNet': ElasticNet(random_state=42),
                'SGDRegressor': SGDRegressor(random_state=42),

                'SVR': SVR(),
                'NuSVR': NuSVR(),
                'LinearSVR': LinearSVR(random_state=42),

                'KNeighborsRegressor': KNeighborsRegressor(),
                'RadiusNeighborsRegressor': RadiusNeighborsRegressor(),

                'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
                'ExtraTreeRegressor': ExtraTreeRegressor(random_state=42),

                'RandomForestRegressor': RandomForestRegressor(random_state=42, n_estimators=100),
                'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
                'AdaBoostRegressor': AdaBoostRegressor(random_state=42),
                'BaggingRegressor': BaggingRegressor(random_state=42),
                'ExtraTreesRegressor': ExtraTreesRegressor(random_state=42),
                'HistGradientBoostingRegressor': HistGradientBoostingRegressor(random_state=42),

                'KernelRidge': KernelRidge(),
                'MLPRegressor': MLPRegressor(random_state=42, max_iter=1000),

                'XGBoost': xgb.XGBRegressor(random_state=42),
                'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
                'CatBoost': CatBoostRegressor(random_state=42, verbose=0),

                'VotingRegressor': None # Placeholder para ensemble
            }

        # Retorna o dicionário de modelos
        return models

    # Método para otimizar hiperparâmetros com Optuna
    def optimize_with_optuna(self, model_name, X_train, y_train, n_trials=50):
        """Otimização hiperparâmetros com Optuna"""
        print(f"Otimizando {model_name} com Optuna...")

        # Função objetivo para o Optuna
        def objective(trial):
            model = None

            # Verifica se o problema é de classificação
            if self.problem_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier

                # Otimização para XGBoost
                if model_name == 'XGBoost':
                    # Define os parâmetros a serem otimizados para o XGBoost Classifier
                    param = {
                        # Sugere um número de estimadores (árvores) entre 50 e 300
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        # Sugere a profundidade máxima da árvore entre 3 e 10
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        # Sugere a taxa de aprendizado entre 0.01 e 0.3 (escala logarítmica)
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        # Sugere a fração de amostras a serem usadas para treinar cada árvore (subsample) entre 0.5 e 1.0
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        # Sugere a fração de features a serem usadas para treinar cada árvore (colsample_bytree) entre 0.5 e 1.0
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    }
                    # Cria uma instância do XGBoost Classifier com os parâmetros sugeridos pelo Optuna
                    model = xgb.XGBClassifier(
                        **param, # Desempacota o dicionário de parâmetros
                        random_state=42, # Define a semente aleatória para reprodutibilidade
                        use_label_encoder=False, # Desabilita o uso do LabelEncoder (recomendado para versões mais recentes do XGBoost)
                        eval_metric='logloss' # Define a métrica de avaliação a ser usada durante o treinamento
                    )

                # Otimização para RandomForestClassifier
                elif model_name == 'RandomForestClassifier':
                    # Define os parâmetros a serem otimizados para o RandomForestClassifier
                    param = {
                        # Sugere um número de estimadores (árvores) entre 50 e 300
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        # Sugere a profundidade máxima da árvore entre 3 e 20
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        # Sugere o número mínimo de amostras para dividir um nó interno entre 2 e 20
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        # Sugere o número mínimo de amostras em um nó folha entre 1 e 10
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        # Sugere a estratégia para escolher o número de features a serem consideradas ao procurar a melhor divisão
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    }
                    # Cria uma instância do RandomForestClassifier com os parâmetros sugeridos pelo Optuna
                    model = RandomForestClassifier(**param, random_state=42)

            # Caso contrário, o problema é de regressão
            else:
                from sklearn.ensemble import RandomForestRegressor

                # Otimização para XGBoost
                if model_name == 'XGBoost':
                    # Define os parâmetros a serem otimizados para o XGBoost Regressor
                    param = {
                        # Sugere um número de estimadores (árvores) entre 50 e 300
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        # Sugere a profundidade máxima da árvore entre 3 e 10
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        # Sugere a taxa de aprendizado entre 0.01 e 0.3 (escala logarítmica)
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        # Sugere a fração de amostras a serem usadas para treinar cada árvore (subsample) entre 0.5 e 1.0
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        # Sugere a fração de features a serem usadas para treinar cada árvore (colsample_bytree) entre 0.5 e 1.0
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    }
                    # Cria uma instância do XGBoost Regressor com os parâmetros sugeridos pelo Optuna
                    model = xgb.XGBRegressor(**param, random_state=42)

                # Otimização para RandomForestRegressor
                elif model_name == 'RandomForestRegressor':
                    # Define os parâmetros a serem otimizados para o RandomForestRegressor
                    param = {
                        # Sugere um número de estimadores (árvores) entre 50 e 300
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        # Sugere a profundidade máxima da árvore entre 3 e 20
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        # Sugere o número mínimo de amostras para dividir um nó interno entre 2 e 20
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        # Sugere o número mínimo de amostras em um nó folha entre 1 e 10
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        # Sugere a estratégia para escolher o número de features a serem consideradas ao procurar a melhor divisão
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    }
                    # Cria uma instância do RandomForestRegressor com os parâmetros sugeridos pelo Optuna
                    model = RandomForestRegressor(**param, random_state=42)


            # Se o modelo não foi definido, retorna infinito negativo
            if model is None:
                return -np.inf

            # Define a estratégia de validação cruzada
            if self.problem_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)

            # Calcula a pontuação média da validação cruzada
            score = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=self.get_scoring_metric(),
                n_jobs=-1
            )
            # Retorna a média das pontuações
            return score.mean()

        # Cria um estudo Optuna com direção de maximização
        study = optuna.create_study(direction='maximize')
        # Otimiza a função objetivo
        study.optimize(objective, n_trials=n_trials)

        # Retorna os melhores hiperparâmetros encontrados
        return study.best_params

    # Método para obter a métrica de avaliação
    def get_scoring_metric(self):
        """Retorna a métrica de avaliação baseada no tipo de problema"""
        # Se for classificação, retorna 'f1_weighted'
        if self.problem_type == 'classification':
            return 'f1_weighted'
        # Se for regressão, retorna 'neg_root_mean_squared_error'
        else:
            return 'neg_root_mean_squared_error'

    # Método para treinar todos os modelos
    def train_models(self, X, y, optimize_top_n=5):
        """Treina todos os modelos com validação cruzada"""
        print(f"Iniciando treinamento de modelos ({self.problem_type})...")

        # Divide os dados em treino e teste
        if self.problem_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Obtém todos os modelos disponíveis
        all_models = self.get_all_models()

        # Itera sobre todos os modelos
        for name, model in all_models.items():
            # Verifica se o modelo não é None (placeholder de ensemble)
            if model is not None:
                try:
                    print(f"Treinando {name}...")

                    # Define a estratégia de validação cruzada
                    if self.problem_type == 'classification':
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    else:
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)

                    # Calcula as pontuações da validação cruzada
                    cv_scores = cross_val_score(
                        model,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=self.get_scoring_metric(),
                        n_jobs=-1
                    )

                    # Treina o modelo com os dados de treino
                    model.fit(X_train, y_train)

                    # Faz previsões nos dados de teste
                    y_pred = model.predict(X_test)

                    # Obtém os scores de predição
                    y_score = self.get_prediction_scores(model, X_test)
                    # Calcula as métricas de avaliação
                    metrics = self.calculate_metrics(y_test, y_pred, y_score=y_score)

                    # Adiciona as métricas de validação cruzada
                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()

                    # Armazena o modelo treinado e seus resultados
                    self.models[name] = model
                    self.results[name] = metrics

                    # Armazena a importância das features se disponível
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = model.feature_importances_

                    print(f"{name}: {self.get_primary_metric(metrics)}")

                except Exception as e:
                    print(f"Erro ao treinar {name}: {str(e)}")

        # Cria o ensemble dos melhores modelos
        self.create_ensemble(X_train, y_train, X_test, y_test)

        # Otimiza os N melhores modelos se optimize_top_n > 0
        if optimize_top_n > 0:
            self.optimize_top_models(optimize_top_n, X_train, y_train, X_test, y_test)

        # Determina o melhor modelo
        self.determine_best_model()

        # Retorna os resultados e o nome do melhor modelo
        return self.results, self.best_model_name

    # Método para obter scores de predição (probabilidades)
    def get_prediction_scores(self, model, X):
        """Obtém scores/probabilidades para métricas como ROC AUC"""
        # Retorna None se não for classificação
        if self.problem_type != 'classification':
            return None

        try:
            # Tenta obter probabilidades de predição
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            # Tenta obter a função de decisão
            elif hasattr(model, 'decision_function'):
                return model.decision_function(X)
            else:
                return None
        except Exception:
            return None

    # Método para calcular métricas de avaliação
    def calculate_metrics(self, y_true, y_pred, y_score=None):
        """Calcula todas as métricas relevantes"""
        metrics = {}

        # Se for classificação
        if self.problem_type == 'classification':
            # Calcula a acurácia do modelo
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            # Calcula a precisão ponderada do modelo
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            # Calcula a revocação ponderada do modelo
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            # Calcula a pontuação F1 ponderada do modelo
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            try:
                # Obtém as classes únicas presentes nos dados verdadeiros
                unique_classes = np.unique(y_true)

                # Verifica se os scores de predição foram fornecidos
                if y_score is not None:
                    # Se houver mais de duas classes (problema de classificação multiclasse)
                    if len(unique_classes) > 2:
                        # Verifica se y_score é um array NumPy com 2 dimensões
                        if isinstance(y_score, np.ndarray) and y_score.ndim == 2:
                            # Calcula a pontuação ROC AUC para classificação multiclasse usando a estratégia 'one-vs-rest' e média ponderada
                            metrics['roc_auc'] = roc_auc_score(
                                y_true,
                                y_score,
                                multi_class='ovr',
                                average='weighted'
                            )
                        else:
                            # Se y_score não for um array 2D, define ROC AUC como NaN
                            metrics['roc_auc'] = np.nan
                    else:
                        # Se houver apenas duas classes (problema de classificação binária)
                        if isinstance(y_score, np.ndarray):
                            # Verifica se y_score é um array NumPy com 2 dimensões e pelo menos 2 colunas
                            if y_score.ndim == 2 and y_score.shape[1] >= 2:
                                # Calcula a pontuação ROC AUC para classificação binária usando a segunda coluna de probabilidades
                                metrics['roc_auc'] = roc_auc_score(y_true, y_score[:, 1])
                            else:
                                # Se y_score não for um array 2D com pelo menos 2 colunas, usa y_score diretamente
                                metrics['roc_auc'] = roc_auc_score(y_true, y_score)
                        else:
                            # Se y_score não for um array NumPy, define ROC AUC como NaN
                            metrics['roc_auc'] = np.nan
                else:
                    # Se y_score for None, define ROC AUC como NaN
                    metrics['roc_auc'] = np.nan

            except Exception:
                # Em caso de qualquer erro durante o cálculo do ROC AUC, define como NaN
                metrics['roc_auc'] = np.nan

            # Calcula a matriz de confusão
            cm = confusion_matrix(y_true, y_pred)
            # Armazena a matriz de confusão nas métricas
            metrics['confusion_matrix'] = cm

        # Se for regressão
        else:
            # Calcula o erro quadrático médio (MSE) entre os valores verdadeiros e as previsões
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            # Calcula a raiz do erro quadrático médio (RMSE)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            # Calcula o erro absoluto médio (MAE) entre os valores verdadeiros e as previsões
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            # Calcula o coeficiente de determinação (R^2)
            metrics['r2'] = r2_score(y_true, y_pred)

            # Garante que os valores verdadeiros sejam positivos para evitar divisão por zero ou valores muito pequenos
            y_true_safe = np.clip(np.abs(y_true), 1e-10, None)
            # Calcula o erro percentual absoluto médio (MAPE)
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

        # Retorna o dicionário de métricas calculadas
        return metrics

    # Método para obter a métrica principal para ranking
    def get_primary_metric(self, metrics):
        """Retorna a métrica principal para ranking"""
        if self.problem_type == 'classification':
            return metrics['f1']
        else:
            return -metrics['rmse']

    # Método para obter o peso do ensemble
    def _get_ensemble_weight(self, metrics):
        """Converte desempenho em peso positivo para o ensemble"""
        if self.problem_type == 'classification':
            # Retorna a pontuação F1 ponderada, garantindo que seja pelo menos um valor pequeno positivo
            return max(metrics.get('f1', 0.0), 1e-6)
        else:
            # Obtém o RMSE dos resultados, retornando None se não estiver presente
            rmse = metrics.get('rmse', None)
            # Se RMSE for None ou não positivo, retorna um valor pequeno positivo para evitar divisão por zero
            if rmse is None or rmse <= 0:
                return 1e-6
            # Retorna o inverso do RMSE (quanto menor o RMSE, maior o peso)
            return 1.0 / (rmse + 1e-6)


    # Método para criar o ensemble
    def create_ensemble(self, X_train, y_train, X_test, y_test):
        """Cria ensemble dos melhores modelos"""
        print("Criando ensemble dos melhores modelos...")
        
        # Ordena os modelos pelo desempenho, pegando os 5 melhores
        sorted_models = sorted(
            # Itera sobre os itens (nome do modelo, métricas) do dicionário de resultados.
            self.results.items(),
            # Define a chave de ordenação como a métrica principal calculada para cada modelo.
            key=lambda x: self.get_primary_metric(x[1]),
            # Ordena em ordem decrescente (do melhor para o pior).
            reverse=True
        # Seleciona os 5 melhores modelos da lista ordenada.
        )[:5]

        ensemble_models = []
        weights = []

        # Seleciona os melhores modelos e seus pesos
        for name, metrics in sorted_models:
            if name in self.models:
                ensemble_models.append((name, self.models[name]))
                weights.append(self._get_ensemble_weight(metrics))

        # Cria o ensemble se houver pelo menos 3 modelos
        if len(ensemble_models) >= 3:
            # Converte a lista de pesos para um array NumPy de ponto flutuante
            weights = np.array(weights, dtype=float)

            # Verifica se a soma dos pesos é menor ou igual a zero. Se for, inicializa todos os pesos como 1.
            if weights.sum() <= 0:
                weights = np.ones(len(weights), dtype=float)

            # Normaliza os pesos para que a soma de todos os pesos seja 1
            weights = weights / weights.sum()

            try:
                # Cria VotingClassifier para classificação
                if self.problem_type == 'classification':
                    # Lista para armazenar modelos compatíveis com predict_proba
                    compatible_models = []
                    # Lista para armazenar os pesos dos modelos compatíveis
                    compatible_weights = []

                    # Itera sobre os modelos selecionados para o ensemble e seus pesos
                    for (name, model), weight in zip(ensemble_models, weights):
                        # Verifica se o modelo possui o método predict_proba (necessário para voting='soft')
                        if hasattr(model, 'predict_proba'):
                            # Adiciona o modelo e seu peso às listas de modelos compatíveis
                            compatible_models.append((name, model))
                            compatible_weights.append(weight)

                    # Verifica se há pelo menos 3 modelos compatíveis para usar voting='soft'
                    if len(compatible_models) >= 3:
                        # Converte os pesos dos modelos compatíveis para um array NumPy
                        compatible_weights = np.array(compatible_weights, dtype=float)
                        # Normaliza os pesos para que a soma seja 1
                        compatible_weights = compatible_weights / compatible_weights.sum()

                        # Cria um VotingClassifier com voting='soft' (baseado em probabilidades)
                        ensemble = VotingClassifier(
                            estimators=compatible_models,
                            voting='soft',
                            weights=compatible_weights.tolist()
                        )
                    # Caso contrário, usa voting='hard' (baseado em previsões diretas)
                    else:
                        ensemble = VotingClassifier(
                            estimators=ensemble_models,
                            voting='hard'
                        )
                # Cria VotingRegressor para regressão
                else:
                    ensemble = VotingRegressor(
                        estimators=ensemble_models,
                        weights=weights.tolist()
                    )

                # Treina o ensemble
                ensemble.fit(X_train, y_train)
                # Faz previsões
                y_pred = ensemble.predict(X_test)

                # Calcula métricas do ensemble
                y_score = self.get_prediction_scores(ensemble, X_test)
                metrics = self.calculate_metrics(y_test, y_pred, y_score=y_score)

                # Calcula pontuações de validação cruzada para o ensemble
                # Define a estratégia de validação cruzada estratificada para classificação, com 5 folds, embaralhamento e semente aleatória fixa.
                if self.problem_type == 'classification':
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                # Define a estratégia de validação cruzada K-Fold para regressão, com 5 folds, embaralhamento e semente aleatória fixa.
                else:
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)

                # Calcula as pontuações da validação cruzada para o modelo de ensemble.
                cv_scores = cross_val_score(
                    ensemble, # O modelo de ensemble a ser avaliado.
                    X_train, # Os dados de treinamento.
                    y_train, # Os rótulos de treinamento.
                    cv=cv, # A estratégia de validação cruzada a ser usada.
                    scoring=self.get_scoring_metric(), # A métrica de pontuação a ser usada.
                    n_jobs=-1 # Usa todos os processadores disponíveis para paralelizar o cálculo.
                )

                # Armazena a média das pontuações da validação cruzada nas métricas do ensemble.
                metrics['cv_mean'] = cv_scores.mean()
                # Armazena o desvio padrão das pontuações da validação cruzada nas métricas do ensemble.
                metrics['cv_std'] = cv_scores.std()

                # Armazena o ensemble e seus resultados
                self.models['Ensemble'] = ensemble
                self.results['Ensemble'] = metrics

                print(f"Ensemble criado com {len(ensemble_models)} modelos")

            except Exception as e:
                print(f"Erro ao criar ensemble: {str(e)}")

    # Método para otimizar os N melhores modelos
    def optimize_top_models(self, n_models, X_train, y_train, X_test, y_test):
        """Otimiza os N melhores modelos"""
        print(f"Otimizando os {n_models} melhores modelos...")

        # Obtém os N melhores modelos com base no desempenho, ordenando-os em ordem decrescente e selecionando os N primeiros.
        sorted_models = sorted(
            # Itera sobre os itens (nome do modelo, métricas) do dicionário de resultados.
            self.results.items(),
            # Define a chave de ordenação como a métrica principal calculada para cada modelo.
            key=lambda x: self.get_primary_metric(x[1]),
            # Ordena em ordem decrescente (do melhor para o pior).
            reverse=True
        # Seleciona os 'n_models' melhores modelos da lista ordenada.
        )[:n_models]

        # Itera sobre os melhores modelos para otimização
        for name, _ in sorted_models:
            # Otimiza apenas XGBoost e Random Forest
            if name in ['XGBoost', 'RandomForestClassifier', 'RandomForestRegressor']:
                try:
                    # Otimiza hiperparâmetros com Optuna
                    best_params = self.optimize_with_optuna(name, X_train, y_train, n_trials=30)

                    model_class = type(self.models[name])

                    # Cria o modelo otimizado
                    # Verifica se o modelo é XGBoost e o problema é de classificação
                    if name == 'XGBoost' and self.problem_type == 'classification':
                        # Cria uma instância do XGBoost Classifier com os melhores parâmetros encontrados pelo Optuna
                        optimized_model = model_class(
                            **best_params, # Desempacota o dicionário de melhores parâmetros
                            random_state=42, # Define a semente aleatória para reprodutibilidade
                            use_label_encoder=False, # Desabilita o uso do LabelEncoder
                            eval_metric='logloss' # Define a métrica de avaliação
                        )
                    # Verifica se o modelo é XGBoost e o problema é de regressão
                    elif name == 'XGBoost' and self.problem_type == 'regression':
                        # Cria uma instância do XGBoost Regressor com os melhores parâmetros encontrados pelo Optuna
                        optimized_model = model_class(
                            **best_params, # Desempacota o dicionário de melhores parâmetros
                            random_state=42 # Define a semente aleatória para reprodutibilidade
                        )
                    # Para outros modelos (como RandomForestClassifier/Regressor)
                    else:
                        # Cria uma instância do modelo com os melhores parâmetros encontrados pelo Optuna
                        optimized_model = model_class(
                            **best_params, # Desempacota o dicionário de melhores parâmetros
                            random_state=42 # Define a semente aleatória para reprodutibilidade
                        )

                    # Treina o modelo otimizado
                    optimized_model.fit(X_train, y_train)
                    # Faz previsões
                    y_pred = optimized_model.predict(X_test)

                    # Calcula métricas do modelo otimizado
                    y_score = self.get_prediction_scores(optimized_model, X_test)
                    metrics = self.calculate_metrics(y_test, y_pred, y_score=y_score)

                    # Define a estratégia de validação cruzada estratificada para classificação, com 5 folds, embaralhamento e semente aleatória fixa.
                    if self.problem_type == 'classification':
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    # Define a estratégia de validação cruzada K-Fold para regressão, com 5 folds, embaralhamento e semente aleatória fixa.
                    else:
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)

                    # Calcula as pontuações da validação cruzada para o modelo otimizado.
                    cv_scores = cross_val_score(
                        optimized_model, # O modelo otimizado a ser avaliado.
                        X_train, # Os dados de treinamento.
                        y_train, # Os rótulos de treinamento.
                        cv=cv, # A estratégia de validação cruzada a ser usada.
                        scoring=self.get_scoring_metric(), # A métrica de pontuação a ser usada.
                        n_jobs=-1 # Usa todos os processadores disponíveis para paralelizar o cálculo.
                    )

                    # Armazena a média das pontuações da validação cruzada nas métricas do modelo otimizado.
                    metrics['cv_mean'] = cv_scores.mean()
                    # Armazena o desvio padrão das pontuações da validação cruzada nas métricas do modelo otimizado.
                    metrics['cv_std'] = cv_scores.std()

                    # Armazena o modelo otimizado e seus resultados
                    self.models[f'{name}_Optimized'] = optimized_model
                    self.results[f'{name}_Optimized'] = metrics

                    # Armazena a importância das features se disponível
                    if hasattr(optimized_model, 'feature_importances_'):
                        # Armazena a importância das features do modelo otimizado se disponível
                        self.feature_importance[f'{name}_Optimized'] = optimized_model.feature_importances_

                    # Imprime a métrica principal do modelo otimizado
                    print(f"{name} otimizado: {self.get_primary_metric(metrics)}")

                # Captura e imprime qualquer exceção que ocorra durante a otimização
                except Exception as e:
                    print(f"Erro ao otimizar {name}: {str(e)}")


    # Método para determinar o melhor modelo
    def determine_best_model(self):
        """Determina o melhor modelo baseado nas métricas"""
        if not self.results:
            return

        # Encontra o nome do melhor modelo
        best_model_name = max(
            self.results.items(),
            key=lambda x: self.get_primary_metric(x[1])
        )[0]

        # Atualiza os atributos best_model_name e best_model
        self.best_model_name = best_model_name
        # Obtém o melhor modelo do dicionário de modelos treinados usando o nome do melhor modelo
        self.best_model = self.models.get(best_model_name)

        # Imprime o nome do melhor modelo encontrado
        print(f"\n MELHOR MODELO: {best_model_name}")
        # Imprime a métrica principal do melhor modelo, formatada com 4 casas decimais
        print(f"Métrica principal: {self.get_primary_metric(self.results[best_model_name]):.4f}")


    # Método para obter os modelos ranqueados
    def get_ranked_models(self):
        """Retorna modelos ordenados do melhor para o pior"""
        # Ordena os resultados pelo desempenho, usando a métrica principal e em ordem decrescente
        ranked = sorted(
            self.results.items(), # Obtém os itens (nome do modelo, métricas) do dicionário de resultados
            key=lambda x: self.get_primary_metric(x[1]), # Define a chave de ordenação como a métrica principal calculada para cada modelo
            reverse=True # Ordena em ordem decrescente (do melhor para o pior)
        )

        # Cria um DataFrame com o ranking dos modelos
        ranking_df = pd.DataFrame([
            {
                'Modelo': name,
                'Métrica Principal': self.get_primary_metric(metrics),
                'Detalhes': metrics
            }
            for name, metrics in ranked
        ])

        return ranking_df

    # Método para salvar os modelos treinados
    def save_models(self, path='models/'):
        """Salva todos os modelos treinados"""
        import os
        # Cria o diretório de modelos se não existir
        os.makedirs(path, exist_ok=True)

        # Salva cada modelo treinado
        for name, model in self.models.items():
            # Salva o modelo usando joblib em um arquivo com extensão .pkl
            joblib.dump(model, f'{path}/{name}.pkl')
            # Imprime uma mensagem confirmando que o modelo foi salvo e o caminho onde foi salvo
            print(f"Modelo {name} salvo em {path}/{name}.pkl")

        # Salva os resultados em um arquivo CSV
        results_df = pd.DataFrame(self.results).T
        # Salva o DataFrame de resultados em um arquivo CSV
        results_df.to_csv(f'{path}/model_results.csv')

        # Imprime uma mensagem confirmando que os resultados foram salvos
        print(f"Resultados salvos em {path}/model_results.csv")