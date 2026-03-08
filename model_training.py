# Este código é responsável por:
# 1. Treinar dezenas de modelos automaticamente
# 2. Avaliar corretamente cada modelo
# 3. Ranqueia os modelos
# 4. Cria ensembles automaticamente
# 5. Otimiza hiperparâmetros com Optuna
# 6. Escolhe o melhor modelo final

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import joblib
import optuna
import xgboost as xgb
import lightgbm as lgb

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import *
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class AdvancedModelTrainer:
    def __init__(self, problem_type):
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = ""
        self.feature_importance = {}

    def get_all_models(self):
        """Retorna mais de 30 modelos de ML"""
        if self.problem_type == 'classification':
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

                'VotingClassifier': None
            }

        else:
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

                'VotingRegressor': None
            }

        return models

    def optimize_with_optuna(self, model_name, X_train, y_train, n_trials=50):
        """Otimização hiperparâmetros com Optuna"""
        print(f"Otimizando {model_name} com Optuna...")

        def objective(trial):
            model = None

            if self.problem_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier

                if model_name == 'XGBoost':
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    }
                    model = xgb.XGBClassifier(
                        **param,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )

                elif model_name == 'RandomForestClassifier':
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    }
                    model = RandomForestClassifier(**param, random_state=42)

            else:
                from sklearn.ensemble import RandomForestRegressor

                if model_name == 'XGBoost':
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    }
                    model = xgb.XGBRegressor(**param, random_state=42)

                elif model_name == 'RandomForestRegressor':
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    }
                    model = RandomForestRegressor(**param, random_state=42)

            if model is None:
                return -np.inf

            if self.problem_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)

            score = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=self.get_scoring_metric(),
                n_jobs=-1
            )
            return score.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    def get_scoring_metric(self):
        """Retorna a métrica de avaliação baseada no tipo de problema"""
        if self.problem_type == 'classification':
            return 'f1_weighted'
        else:
            return 'neg_root_mean_squared_error'

    def train_models(self, X, y, optimize_top_n=5):
        """Treina todos os modelos com validação cruzada"""
        print(f"Iniciando treinamento de modelos ({self.problem_type})...")

        if self.problem_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        all_models = self.get_all_models()

        for name, model in all_models.items():
            if model is not None:
                try:
                    print(f"Treinando {name}...")

                    if self.problem_type == 'classification':
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    else:
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)

                    cv_scores = cross_val_score(
                        model,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=self.get_scoring_metric(),
                        n_jobs=-1
                    )

                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    y_score = self.get_prediction_scores(model, X_test)
                    metrics = self.calculate_metrics(y_test, y_pred, y_score=y_score)

                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()

                    self.models[name] = model
                    self.results[name] = metrics

                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = model.feature_importances_

                    print(f"{name}: {self.get_primary_metric(metrics)}")

                except Exception as e:
                    print(f"Erro ao treinar {name}: {str(e)}")

        self.create_ensemble(X_train, y_train, X_test, y_test)

        if optimize_top_n > 0:
            self.optimize_top_models(optimize_top_n, X_train, y_train, X_test, y_test)

        self.determine_best_model()

        return self.results, self.best_model_name

    def get_prediction_scores(self, model, X):
        """Obtém scores/probabilidades para métricas como ROC AUC"""
        if self.problem_type != 'classification':
            return None

        try:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            elif hasattr(model, 'decision_function'):
                return model.decision_function(X)
            else:
                return None
        except Exception:
            return None

    def calculate_metrics(self, y_true, y_pred, y_score=None):
        """Calcula todas as métricas relevantes"""
        metrics = {}

        if self.problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            try:
                unique_classes = np.unique(y_true)

                if y_score is not None:
                    if len(unique_classes) > 2:
                        if isinstance(y_score, np.ndarray) and y_score.ndim == 2:
                            metrics['roc_auc'] = roc_auc_score(
                                y_true,
                                y_score,
                                multi_class='ovr',
                                average='weighted'
                            )
                        else:
                            metrics['roc_auc'] = np.nan
                    else:
                        if isinstance(y_score, np.ndarray):
                            if y_score.ndim == 2 and y_score.shape[1] >= 2:
                                metrics['roc_auc'] = roc_auc_score(y_true, y_score[:, 1])
                            else:
                                metrics['roc_auc'] = roc_auc_score(y_true, y_score)
                        else:
                            metrics['roc_auc'] = np.nan
                else:
                    metrics['roc_auc'] = np.nan

            except Exception:
                metrics['roc_auc'] = np.nan

            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm

        else:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)

            y_true_safe = np.clip(np.abs(y_true), 1e-10, None)
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

        return metrics

    def get_primary_metric(self, metrics):
        """Retorna a métrica principal para ranking"""
        if self.problem_type == 'classification':
            return metrics['f1']
        else:
            return -metrics['rmse']

    def _get_ensemble_weight(self, metrics):
        """Converte desempenho em peso positivo para o ensemble"""
        if self.problem_type == 'classification':
            return max(metrics.get('f1', 0.0), 1e-6)
        else:
            rmse = metrics.get('rmse', None)
            if rmse is None or rmse <= 0:
                return 1e-6
            return 1.0 / (rmse + 1e-6)

    def create_ensemble(self, X_train, y_train, X_test, y_test):
        """Cria ensemble dos melhores modelos"""
        print("Criando ensemble dos melhores modelos...")

        sorted_models = sorted(
            self.results.items(),
            key=lambda x: self.get_primary_metric(x[1]),
            reverse=True
        )[:5]

        ensemble_models = []
        weights = []

        for name, metrics in sorted_models:
            if name in self.models:
                ensemble_models.append((name, self.models[name]))
                weights.append(self._get_ensemble_weight(metrics))

        if len(ensemble_models) >= 3:
            weights = np.array(weights, dtype=float)

            if weights.sum() <= 0:
                weights = np.ones(len(weights), dtype=float)

            weights = weights / weights.sum()

            try:
                if self.problem_type == 'classification':
                    compatible_models = []
                    compatible_weights = []

                    for (name, model), weight in zip(ensemble_models, weights):
                        if hasattr(model, 'predict_proba'):
                            compatible_models.append((name, model))
                            compatible_weights.append(weight)

                    if len(compatible_models) >= 3:
                        compatible_weights = np.array(compatible_weights, dtype=float)
                        compatible_weights = compatible_weights / compatible_weights.sum()

                        ensemble = VotingClassifier(
                            estimators=compatible_models,
                            voting='soft',
                            weights=compatible_weights.tolist()
                        )
                    else:
                        ensemble = VotingClassifier(
                            estimators=ensemble_models,
                            voting='hard'
                        )
                else:
                    ensemble = VotingRegressor(
                        estimators=ensemble_models,
                        weights=weights.tolist()
                    )

                ensemble.fit(X_train, y_train)
                y_pred = ensemble.predict(X_test)

                y_score = self.get_prediction_scores(ensemble, X_test)
                metrics = self.calculate_metrics(y_test, y_pred, y_score=y_score)

                if self.problem_type == 'classification':
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                else:
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)

                cv_scores = cross_val_score(
                    ensemble,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=self.get_scoring_metric(),
                    n_jobs=-1
                )

                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()

                self.models['Ensemble'] = ensemble
                self.results['Ensemble'] = metrics

                print(f"Ensemble criado com {len(ensemble_models)} modelos")

            except Exception as e:
                print(f"Erro ao criar ensemble: {str(e)}")

    def optimize_top_models(self, n_models, X_train, y_train, X_test, y_test):
        """Otimiza os N melhores modelos"""
        print(f"Otimizando os {n_models} melhores modelos...")

        sorted_models = sorted(
            self.results.items(),
            key=lambda x: self.get_primary_metric(x[1]),
            reverse=True
        )[:n_models]

        for name, _ in sorted_models:
            if name in ['XGBoost', 'RandomForestClassifier', 'RandomForestRegressor']:
                try:
                    best_params = self.optimize_with_optuna(name, X_train, y_train, n_trials=30)

                    model_class = type(self.models[name])

                    if name == 'XGBoost' and self.problem_type == 'classification':
                        optimized_model = model_class(
                            **best_params,
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric='logloss'
                        )
                    elif name == 'XGBoost' and self.problem_type == 'regression':
                        optimized_model = model_class(
                            **best_params,
                            random_state=42
                        )
                    else:
                        optimized_model = model_class(
                            **best_params,
                            random_state=42
                        )

                    optimized_model.fit(X_train, y_train)
                    y_pred = optimized_model.predict(X_test)

                    y_score = self.get_prediction_scores(optimized_model, X_test)
                    metrics = self.calculate_metrics(y_test, y_pred, y_score=y_score)

                    if self.problem_type == 'classification':
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    else:
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)

                    cv_scores = cross_val_score(
                        optimized_model,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=self.get_scoring_metric(),
                        n_jobs=-1
                    )

                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()

                    self.models[f'{name}_Optimized'] = optimized_model
                    self.results[f'{name}_Optimized'] = metrics

                    if hasattr(optimized_model, 'feature_importances_'):
                        self.feature_importance[f'{name}_Optimized'] = optimized_model.feature_importances_

                    print(f"{name} otimizado: {self.get_primary_metric(metrics)}")

                except Exception as e:
                    print(f"Erro ao otimizar {name}: {str(e)}")

    def determine_best_model(self):
        """Determina o melhor modelo baseado nas métricas"""
        if not self.results:
            return

        best_model_name = max(
            self.results.items(),
            key=lambda x: self.get_primary_metric(x[1])
        )[0]

        self.best_model_name = best_model_name
        self.best_model = self.models.get(best_model_name)

        print(f"\n MELHOR MODELO: {best_model_name}")
        print(f"Métrica principal: {self.get_primary_metric(self.results[best_model_name]):.4f}")

    def get_ranked_models(self):
        """Retorna modelos ordenados do melhor para o pior"""
        ranked = sorted(
            self.results.items(),
            key=lambda x: self.get_primary_metric(x[1]),
            reverse=True
        )

        ranking_df = pd.DataFrame([
            {
                'Modelo': name,
                'Métrica Principal': self.get_primary_metric(metrics),
                'Detalhes': metrics
            }
            for name, metrics in ranked
        ])

        return ranking_df

    def save_models(self, path='models/'):
        """Salva todos os modelos treinados"""
        import os
        os.makedirs(path, exist_ok=True)

        for name, model in self.models.items():
            joblib.dump(model, f'{path}/{name}.pkl')
            print(f"Modelo {name} salvo em {path}/{name}.pkl")

        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(f'{path}/model_results.csv')

        print(f"Resultados salvos em {path}/model_results.csv")