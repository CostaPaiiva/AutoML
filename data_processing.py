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
        if self.target_column:
            target = data[self.target_column]
            # Se o target for categórico ou tiver poucos valores únicos, é classificação
            if target.dtype == 'object' or len(target.unique()) <= 10:
                return 'classification'
            else:
                return 'regression'
        return 'auto'
    
    def advanced_cleaning(self, data):
        """Limpeza avançada dos dados"""
        print("Realizando limpeza avançada dos dados...")
        
        # 1. Remover duplicatas
        data_cleaned = data.drop_duplicates()
        print(f"Duplicatas removidas: {len(data) - len(data_cleaned)}")
        
        # 2. Tratar valores infinitos
        data_cleaned = data_cleaned.replace([np.inf, -np.inf], np.nan)
        
        # 3. Remover colunas com muitos valores faltantes (>50%)
        missing_percentage = data_cleaned.isnull().mean()
        columns_to_drop = missing_percentage[missing_percentage > 0.5].index.tolist()
        if columns_to_drop:
            data_cleaned = data_cleaned.drop(columns=columns_to_drop)
            print(f"Colunas removidas (alta taxa de missing): {columns_to_drop}")
        
        # 4. Detectar e remover outliers usando IQR (apenas para datasets pequenos)
        if len(data_cleaned) < 10000:  # Apenas para datasets pequenos
            numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
            if self.target_column in numeric_cols:
                numeric_cols = numeric_cols.drop(self.target_column)
            
            for col in numeric_cols:
                if col != self.target_column:
                    Q1 = data_cleaned[col].quantile(0.25)
                    Q3 = data_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = data_cleaned[(data_cleaned[col] < lower_bound) | 
                                           (data_cleaned[col] > upper_bound)]
                    if len(outliers) > 0 and len(outliers) < len(data_cleaned) * 0.1:  # Menos de 10% outliers
                        data_cleaned = data_cleaned[(data_cleaned[col] >= lower_bound) & 
                                                   (data_cleaned[col] <= upper_bound)]
        
        # 5. Engenharia de features básica (apenas para datasets pequenos)
        if len(data_cleaned) < 10000:
            data_cleaned = self.feature_engineering(data_cleaned)
        
        return data_cleaned
    
    def feature_engineering(self, data):
        """Engenharia de features avançada"""
        print("Aplicando engenharia de features...")
        
        # Adicionar features interativas (apenas algumas combinações)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if self.target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(self.target_column)
        
        # Limitar a 5 features numéricas para não explodir o número de colunas
        numeric_cols = numeric_cols[:5]
        
        if len(numeric_cols) >= 2:
            # Apenas algumas combinações
            for i in range(min(len(numeric_cols), 3)):
                for j in range(i+1, min(len(numeric_cols), 4)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                    # Evitar divisão por zero
                    mask = data[col2] != 0
                    data[f'{col1}_div_{col2}'] = np.where(mask, data[col1] / data[col2], 0)
        
        # Adicionar estatísticas básicas
        if len(numeric_cols) > 1:
            data['mean_features'] = data[numeric_cols].mean(axis=1)
            data['std_features'] = data[numeric_cols].std(axis=1)
        
        return data
    
    def handle_missing_values(self, data, strategy='simple'):
        """Tratamento de valores faltantes - VERSÃO CORRIGIDA"""
        print("Tratando valores faltantes...")
        
        if strategy == 'simple':
            # Método simples e robusto
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            # Para colunas numéricas: imputar com mediana (mais robusta que média)
            for col in numeric_cols:
                if data[col].isnull().any():
                    # Usar mediana para evitar influência de outliers
                    median_value = data[col].median()
                    data[col] = data[col].fillna(median_value)
            
            # Para colunas categóricas: imputar com moda
            for col in categorical_cols:
                if data[col].isnull().any():
                    # Encontrar valor mais frequente
                    mode_value = data[col].mode()
                    if not mode_value.empty:
                        data[col] = data[col].fillna(mode_value[0])
                    else:
                        # Se não houver moda, usar string vazia
                        data[col] = data[col].fillna('')
            
            return data
        
        else:
            # Método original (com correção)
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            
            # Para colunas numéricas
            if numeric_cols:
                # Converter para numpy array primeiro
                numeric_data = data[numeric_cols].values
                imputer_num = SimpleImputer(strategy='mean')
                numeric_data_imputed = imputer_num.fit_transform(numeric_data)
                data[numeric_cols] = numeric_data_imputed
            
            # Para colunas categóricas
            if categorical_cols:
                # Para categóricas, usar fillna com moda para evitar problemas
                for col in categorical_cols:
                    mode_value = data[col].mode()
                    if not mode_value.empty:
                        data[col] = data[col].fillna(mode_value[0])
                    else:
                        data[col] = data[col].fillna('missing')
            
            return data
    
    def encode_categorical(self, data):
        """Codificação avançada de variáveis categóricas - VERSÃO SIMPLIFICADA"""
        print("Codificando variáveis categóricas...")
        
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != self.target_column:
                # Se tiver poucas categorias, usar One-Hot
                if data[col].nunique() <= 10:
                    # One-Hot Encoding
                    dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                    data = pd.concat([data.drop(columns=[col]), dummies], axis=1)
                else:
                    # Se tiver muitas categorias, usar Label Encoding
                    le = LabelEncoder()
                    # Lidar com NaN
                    col_data = data[col].fillna('missing').astype(str)
                    data[col] = le.fit_transform(col_data)
                    self.encoders[col] = le
        
        return data
    
    def scale_features(self, data):
        """Normalização e padronização das features - VERSÃO SIMPLIFICADA"""
        print("Escalando features...")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if self.target_column and self.target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(self.target_column)
        
        if len(numeric_cols) > 0:
            # Apenas scale se houver variação
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
        
        # Determinar k automaticamente
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
        except:
            print("Erro na seleção de features. Usando todas as features.")
            return X
    
    def process(self, data_path):
        """Pipeline completo de processamento - VERSÃO ROBUSTA"""
        print(f"Carregando dados...")
        
        try:
            # Carregar dados
            if isinstance(data_path, str):
                data = pd.read_csv(data_path)
            else:
                # Se for um objeto de arquivo
                data = pd.read_csv(data_path)
            
            print(f"Dados carregados: {data.shape[0]} linhas, {data.shape[1]} colunas")
            
            # Detectar tipo de problema
            if self.problem_type == 'auto':
                self.problem_type = self.detect_problem_type(data)
            print(f"Tipo de problema detectado: {self.problem_type}")
            
            # Processamento em etapas com tratamento de erros
            data = self.advanced_cleaning(data)
            data = self.handle_missing_values(data, strategy='simple')
            data = self.encode_categorical(data)
            
            # Separar features e target
            if self.target_column:
                if self.target_column not in data.columns:
                    raise ValueError(f"Coluna target '{self.target_column}' não encontrada nos dados.")
                
                X = data.drop(columns=[self.target_column])
                y = data[self.target_column]
            else:
                # Se não houver target especificado, usar a última coluna
                self.target_column = data.columns[-1]
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]
            
            # Feature selection (apenas se houver muitas features)
            if X.shape[1] > 10:
                X = self.feature_selection(X, y)
            
            # Scaling
            X = self.scale_features(X)
            
            print(f"✅ Processamento concluído. Shape final: {X.shape}")
            
            return X, y, self.problem_type
            
        except Exception as e:
            print(f"❌ Erro no processamento: {str(e)}")
            # Tentar método de fallback
            return self.simple_process(data_path)
    
    def simple_process(self, data_path):
        """Processamento simples de fallback"""
        print("Usando processamento simples de fallback...")
        
        try:
            # Carregar dados
            if isinstance(data_path, str):
                data = pd.read_csv(data_path)
            else:
                data = pd.read_csv(data_path)
            
            # Separar features e target
            if self.target_column and self.target_column in data.columns:
                X = data.drop(columns=[self.target_column])
                y = data[self.target_column]
            else:
                self.target_column = data.columns[-1]
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]
            
            # Detectar tipo de problema
            if y.dtype == 'object' or len(y.unique()) <= 10:
                problem_type = 'classification'
            else:
                problem_type = 'regression'
            
            # Limpeza básica
            # Remover colunas com muitos NaN
            X = X.dropna(axis=1, thresh=len(X) * 0.5)
            
            # Preencher valores faltantes
            X = X.fillna(X.mean(numeric_only=True))
            
            # Codificar categóricas simples
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = pd.factorize(X[col])[0]
            
            print(f"✅ Processamento simples concluído. Shape final: {X.shape}")
            
            return X, y, problem_type
            
        except Exception as e:
            print(f"❌ Erro no processamento simples: {str(e)}")
            raise