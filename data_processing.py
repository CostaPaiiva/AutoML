import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


class AdvancedDataProcessor:
    def __init__(self, target_column=None, problem_type='auto'):
        self.target_column = target_column
        self.problem_type = problem_type
        self.preprocessors = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_selector = None
        self.imputer = None

    def detect_problem_type(self, data):
        """Detecta automaticamente se é classificação ou regressão"""
        if self.target_column and self.target_column in data.columns:
            target = data[self.target_column]

            try:
                target_numeric = pd.to_numeric(target, errors='coerce')
                numeric_ratio = target_numeric.notna().mean()

                if target.dtype == 'object' or str(target.dtype) == 'category':
                    return 'classification'

                if numeric_ratio < 0.8:
                    return 'classification'

                unique_count = target.nunique(dropna=True)
                if unique_count <= 10:
                    return 'classification'

                return 'regression'
            except Exception:
                if target.dtype == 'object' or len(target.unique()) <= 10:
                    return 'classification'
                return 'regression'

        return 'auto'

    def advanced_cleaning(self, data):
        """Limpeza avançada dos dados"""
        print("Realizando limpeza avançada dos dados...")

        data_cleaned = data.copy()

        # 1. Remover duplicatas
        original_len = len(data_cleaned)
        data_cleaned = data_cleaned.drop_duplicates()
        print(f"Duplicatas removidas: {original_len - len(data_cleaned)}")

        # 2. Tratar valores infinitos
        data_cleaned = data_cleaned.replace([np.inf, -np.inf], np.nan)

        # 3. Remover colunas com muitos valores faltantes (>50%)
        missing_percentage = data_cleaned.isnull().mean()
        columns_to_drop = missing_percentage[missing_percentage > 0.5].index.tolist()

        if columns_to_drop:
            data_cleaned = data_cleaned.drop(columns=columns_to_drop)
            print(f"Colunas removidas (alta taxa de missing): {columns_to_drop}")

        # 4. Detectar e tratar outliers de forma menos agressiva
        if len(data_cleaned) < 10000:
            numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns.tolist()

            if self.target_column in numeric_cols:
                numeric_cols.remove(self.target_column)

            # Em vez de remover linhas sucessivamente, faz clipping dos outliers
            for col in numeric_cols:
                series = data_cleaned[col]

                if series.notna().sum() < 5:
                    continue

                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1

                if pd.isna(IQR) or IQR == 0:
                    continue

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (series < lower_bound) | (series > upper_bound)
                outlier_ratio = outlier_mask.mean()

                if 0 < outlier_ratio < 0.1:
                    data_cleaned[col] = series.clip(lower=lower_bound, upper=upper_bound)

        # 5. Engenharia de features básica
        if len(data_cleaned) < 10000:
            data_cleaned = self.feature_engineering(data_cleaned)

        return data_cleaned

    def feature_engineering(self, data):
        """Engenharia de features avançada"""
        print("Aplicando engenharia de features...")

        data = data.copy()

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)

        numeric_cols = numeric_cols[:5]

        if len(numeric_cols) >= 2:
            for i in range(min(len(numeric_cols), 3)):
                for j in range(i + 1, min(len(numeric_cols), 4)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    data[f'{col1}_x_{col2}'] = data[col1] * data[col2]

                    mask = data[col2] != 0
                    data[f'{col1}_div_{col2}'] = np.where(mask, data[col1] / data[col2], 0)

        if len(numeric_cols) > 1:
            data['mean_features'] = data[numeric_cols].mean(axis=1)
            data['std_features'] = data[numeric_cols].std(axis=1)

        return data

    def handle_missing_values(self, data, strategy='simple'):
        """Tratamento de valores faltantes - VERSÃO CORRIGIDA"""
        print("Tratando valores faltantes...")

        data = data.copy()

        if strategy == 'simple':
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

            for col in numeric_cols:
                if data[col].isnull().any():
                    median_value = data[col].median()
                    data[col] = data[col].fillna(median_value)

            for col in categorical_cols:
                if data[col].isnull().any():
                    mode_value = data[col].mode()
                    if not mode_value.empty:
                        data[col] = data[col].fillna(mode_value.iloc[0])
                    else:
                        data[col] = data[col].fillna('')

            return data

        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

            if numeric_cols:
                numeric_data = data[numeric_cols].values
                imputer_num = SimpleImputer(strategy='mean')
                numeric_data_imputed = imputer_num.fit_transform(numeric_data)
                data[numeric_cols] = numeric_data_imputed

            if categorical_cols:
                for col in categorical_cols:
                    mode_value = data[col].mode()
                    if not mode_value.empty:
                        data[col] = data[col].fillna(mode_value.iloc[0])
                    else:
                        data[col] = data[col].fillna('missing')

            return data

    def encode_categorical(self, data):
        """Codificação avançada de variáveis categóricas - VERSÃO SIMPLIFICADA"""
        print("Codificando variáveis categóricas...")

        data = data.copy()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_cols:
            if col != self.target_column:
                nunique = data[col].nunique(dropna=False)

                if nunique <= 10:
                    dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                    data = pd.concat([data.drop(columns=[col]), dummies], axis=1)
                else:
                    le = LabelEncoder()
                    col_data = data[col].fillna('missing').astype(str)
                    data[col] = le.fit_transform(col_data)
                    self.encoders[col] = le

        return data

    def scale_features(self, data):
        """Normalização e padronização das features - VERSÃO SIMPLIFICADA"""
        print("Escalando features...")

        data = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if self.target_column and self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)

        if len(numeric_cols) > 0:
            cols_to_scale = []
            for col in numeric_cols:
                if data[col].std() > 0:
                    cols_to_scale.append(col)

            if cols_to_scale:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data[cols_to_scale])
                data[cols_to_scale] = data_scaled
                self.scalers['standard'] = scaler

        return data

    def feature_selection(self, X, y, k='auto'):
        """Seleção avançada de features - VERSÃO SIMPLIFICADA"""
        print("Selecionando melhores features...")

        if k == 'auto':
            k = min(20, X.shape[1])

        if k >= X.shape[1]:
            print(f"Não há features suficientes para seleção. Usando todas as {X.shape[1]} features.")
            return X

        if self.problem_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            selector = SelectKBest(score_func=f_regression, k=k)

        try:
            X_selected = selector.fit_transform(X, y)
            self.feature_selector = selector

            selected_features = X.columns[selector.get_support()].tolist()
            print(f"Features selecionadas: {len(selected_features)}/{X.shape[1]}")

            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        except Exception:
            print("Erro na seleção de features. Usando todas as features.")
            return X

    def process(self, data_path):
        """Pipeline completo de processamento - VERSÃO ROBUSTA"""
        print("Carregando dados...")

        try:
            if isinstance(data_path, str):
                data = pd.read_csv(data_path)
            else:
                data = pd.read_csv(data_path)

            print(f"Dados carregados: {data.shape[0]} linhas, {data.shape[1]} colunas")

            # Definir target antes de tudo
            if not self.target_column:
                self.target_column = data.columns[-1]

            if self.target_column not in data.columns:
                raise ValueError(f"Coluna target '{self.target_column}' não encontrada nos dados.")

            # Detectar tipo do problema
            if self.problem_type == 'auto':
                self.problem_type = self.detect_problem_type(data)
            print(f"Tipo de problema detectado: {self.problem_type}")

            # Separar X e y antes do processamento pesado
            X = data.drop(columns=[self.target_column]).copy()
            y = data[self.target_column].copy()

            # Processar apenas X
            X = self.advanced_cleaning(X)
            X = self.handle_missing_values(X, strategy='simple')
            X = self.encode_categorical(X)

            # Reindexar y se alguma linha foi removida ou alterada em X
            y = y.loc[X.index]

            # Feature selection
            if X.shape[1] > 10:
                X = self.feature_selection(X, y)

            # Scaling
            X = self.scale_features(X)

            print(f"✅ Processamento concluído. Shape final: {X.shape}")

            return X, y, self.problem_type

        except Exception as e:
            print(f"❌ Erro no processamento: {str(e)}")
            return self.simple_process(data_path)

    def simple_process(self, data_path):
        """Processamento simples de fallback"""
        print("Usando processamento simples de fallback...")

        try:
            if isinstance(data_path, str):
                data = pd.read_csv(data_path)
            else:
                data = pd.read_csv(data_path)

            if self.target_column and self.target_column in data.columns:
                X = data.drop(columns=[self.target_column]).copy()
                y = data[self.target_column].copy()
            else:
                self.target_column = data.columns[-1]
                X = data.iloc[:, :-1].copy()
                y = data.iloc[:, -1].copy()

            if y.dtype == 'object' or str(y.dtype) == 'category' or len(y.unique()) <= 10:
                problem_type = 'classification'
            else:
                problem_type = 'regression'

            # Remover colunas com muitos NaN
            X = X.dropna(axis=1, thresh=int(len(X) * 0.5))

            # Numéricas
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

            # Categóricas
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                X[col] = X[col].fillna('missing')
                X[col] = pd.factorize(X[col])[0]

            print(f"✅ Processamento simples concluído. Shape final: {X.shape}")

            return X, y, problem_type

        except Exception as e:
            print(f"❌ Erro no processamento simples: {str(e)}")
            raise