# Esse código é um pipeline automatizado de pré-processamento que:

# Limpa dados (duplicatas, outliers, missing).

# Cria novas features.

# Codifica variáveis categóricas de forma inteligente.

# Normaliza e seleciona as melhores features.

# Detecta automaticamente se o problema é classificação ou regressão.

# Importa a biblioteca pandas para manipulação de dados tabulares
import pandas as pd

# Importa a biblioteca numpy para operações numéricas e vetoriais
import numpy as np

# Importa codificadores e escaladores do scikit-learn
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# Importa imputadores para tratamento de valores ausentes
from sklearn.impute import SimpleImputer, KNNImputer

# Importa métodos de seleção de features e funções estatísticas
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression

# Importa o módulo de warnings
import warnings

# Suprime avisos para evitar poluição no log
warnings.filterwarnings('ignore')


# Define a classe principal de processamento avançado de dados
class AdvancedDataProcessor:

    # Método construtor da classe
    def __init__(self, target_column=None, problem_type='auto'):

        # Define a coluna alvo (target)
        self.target_column = target_column

        # Define o tipo do problema (classificação, regressão ou automático)
        self.problem_type = problem_type

        # Dicionário para armazenar preprocessadores
        self.preprocessors = {}

        # Dicionário para armazenar encoders categóricos
        self.encoders = {}

        # Dicionário para armazenar escaladores
        self.scalers = {}

        # Inicializa o seletor de features
        self.feature_selector = None

        # Inicializa o imputador
        self.imputer = None
        

    # Método para detectar automaticamente o tipo do problema
    def detect_problem_type(self, data):
        """Detecta automaticamente se é classificação ou regressão"""

        # Verifica se a coluna alvo foi definida
        if self.target_column:

            # Extrai o target do dataframe
            target = data[self.target_column]

            # Se o target for categórico ou tiver poucos valores únicos
            if target.dtype == 'object' or len(target.unique()) <= 10:

                # Define o problema como classificação
                return 'classification'

            # Caso contrário
            else:

                # Define o problema como regressão
                return 'regression'

        # Se não for possível detectar, retorna automático
        return 'auto'
    

    # Método para limpeza avançada dos dados
    def advanced_cleaning(self, data):
        """Limpeza avançada dos dados"""

        # Log informativo
        print("Realizando limpeza avançada dos dados...")
        
        # Remove linhas duplicadas
        data_cleaned = data.drop_duplicates()

        # Log da quantidade de duplicatas removidas
        print(f"Duplicatas removidas: {len(data) - len(data_cleaned)}")
        
        # Substitui valores infinitos por NaN
        data_cleaned = data_cleaned.replace([np.inf, -np.inf], np.nan)
        
        # Calcula a porcentagem de valores ausentes por coluna
        missing_percentage = data_cleaned.isnull().mean()

        # Identifica colunas com mais de 50% de valores ausentes
        columns_to_drop = missing_percentage[missing_percentage > 0.5].index.tolist()

        # Remove as colunas com alta taxa de missing
        data_cleaned = data_cleaned.drop(columns=columns_to_drop)

        # Log das colunas removidas
        print(f"Colunas removidas (alta taxa de missing): {columns_to_drop}")
        
        # Seleciona colunas numéricas
        numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns

        # Loop por cada coluna numérica
        for col in numeric_cols:

            # Ignora a coluna target
            if col != self.target_column:

                # Calcula o primeiro quartil
                Q1 = data_cleaned[col].quantile(0.25)

                # Calcula o terceiro quartil
                Q3 = data_cleaned[col].quantile(0.75)

                # Calcula o intervalo interquartil
                IQR = Q3 - Q1

                # Define o limite inferior
                lower_bound = Q1 - 1.5 * IQR

                # Define o limite superior
                upper_bound = Q3 + 1.5 * IQR

                # Identifica outliers
                outliers = data_cleaned[
                    (data_cleaned[col] < lower_bound) | 
                    (data_cleaned[col] > upper_bound)
                ]

                # Se houver outliers
                if len(outliers) > 0:

                    # Remove os outliers do dataset
                    data_cleaned = data_cleaned[
                        (data_cleaned[col] >= lower_bound) & 
                        (data_cleaned[col] <= upper_bound)
                    ]
        
        # Aplica engenharia de features
        data_cleaned = self.feature_engineering(data_cleaned)
        
        # Retorna o dataset limpo
        return data_cleaned
    

    # Método para engenharia de features
    def feature_engineering(self, data):
        """Engenharia de features avançada"""

        # Log informativo
        print("Aplicando engenharia de features...")
        
        # Seleciona colunas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Remove a coluna target da lista
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        # Verifica se há pelo menos duas colunas numéricas
        if len(numeric_cols) >= 2:

            # Loop pelas combinações de colunas
            for i in range(len(numeric_cols)):

                # Loop limitado para evitar explosão de features
                for j in range(i+1, min(len(numeric_cols), 5)):

                    # Define as duas colunas
                    col1, col2 = numeric_cols[i], numeric_cols[j]

                    # Cria feature de multiplicação
                    data[f'{col1}_x_{col2}'] = data[col1] * data[col2]

                    # Cria feature de divisão protegida contra zero
                    data[f'{col1}_div_{col2}'] = np.where(
                        data[col2] != 0, 
                        data[col1] / data[col2], 
                        0
                    )
        
        # Se houver mais de uma feature numérica
        if len(numeric_cols) > 1:

            # Calcula a média das features numéricas
            data['mean_features'] = data[numeric_cols].mean(axis=1)

            # Calcula o desvio padrão das features numéricas
            data['std_features'] = data[numeric_cols].std(axis=1)
        
        # Retorna o dataset com novas features
        return data
    

    # Método para tratamento de valores ausentes
    def handle_missing_values(self, data, strategy='auto'):
        """Tratamento sofisticado de valores faltantes"""

        # Log informativo
        print("Tratando valores faltantes...")
        
        # Estratégia automática
        if strategy == 'auto':

            # Identifica colunas numéricas
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            # Identifica colunas categóricas
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            # Se houver colunas numéricas
            if len(numeric_cols) > 0:

                # Inicializa o imputador KNN
                imputer_num = KNNImputer(n_neighbors=5)

                # Aplica imputação nos dados numéricos
                data_numeric = pd.DataFrame(
                    imputer_num.fit_transform(data[numeric_cols]),
                    columns=numeric_cols,
                    index=data.index
                )

                # Atualiza o dataframe original
                data[numeric_cols] = data_numeric
            
            # Se houver colunas categóricas
            if len(categorical_cols) > 0:

                # Inicializa o imputador por moda
                imputer_cat = SimpleImputer(strategy='most_frequent')

                # Aplica imputação nos dados categóricos
                data_cat = pd.DataFrame(
                    imputer_cat.fit_transform(data[categorical_cols]),
                    columns=categorical_cols,
                    index=data.index
                )

                # Atualiza o dataframe original
                data[categorical_cols] = data_cat
        
        # Retorna os dados tratados
        return data
    

    # Método para codificação de variáveis categóricas
    def encode_categorical(self, data):
        """Codificação avançada de variáveis categóricas"""

        # Log informativo
        print("Codificando variáveis categóricas...")
        
        # Identifica colunas categóricas
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Loop pelas colunas categóricas
        for col in categorical_cols:

            # Ignora a coluna target
            if col != self.target_column:

                # Se a cardinalidade for alta e houver target
                if len(data[col].unique()) > 10 and self.target_column:

                    # Target encoding para classificação
                    if self.problem_type == 'classification':

                        # Calcula a média do target por categoria
                        target_means = data.groupby(col)[self.target_column].mean()

                        # Cria a feature codificada
                        data[f'{col}_target_encoded'] = data[col].map(target_means)

                        # Remove a coluna original
                        data = data.drop(columns=[col])

                    # Target encoding para regressão
                    else:

                        # Calcula a média do target por categoria
                        target_means = data.groupby(col)[self.target_column].mean()

                        # Cria a feature codificada
                        data[f'{col}_target_encoded'] = data[col].map(target_means)

                        # Remove a coluna original
                        data = data.drop(columns=[col])

                # Se a cardinalidade for baixa
                else:

                    # One-Hot Encoding
                    if len(data[col].unique()) <= 10:

                        # Gera variáveis dummy
                        dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)

                        # Concatena ao dataframe
                        data = pd.concat([data.drop(columns=[col]), dummies], axis=1)

                    # Caso contrário, usa Label Encoding
                    else:

                        # Inicializa o LabelEncoder
                        le = LabelEncoder()

                        # Aplica a codificação
                        data[col] = le.fit_transform(data[col].astype(str))

                        # Armazena o encoder
                        self.encoders[col] = le
        
        # Retorna os dados codificados
        return data
    

    # Método para escalonamento de features
    def scale_features(self, data):
        """Normalização e padronização das features"""

        # Log informativo
        print("Escalando features...")
        
        # Seleciona colunas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Remove a coluna target do escalonamento
        if self.target_column and self.target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(self.target_column)
        
        # Se houver colunas para escalar
        if len(numeric_cols) > 0:

            # Inicializa o StandardScaler
            scaler = StandardScaler()

            # Aplica o escalonamento
            data_scaled = scaler.fit_transform(data[numeric_cols])

            # Atualiza o dataframe
            data[numeric_cols] = data_scaled

            # Armazena o scaler
            self.scalers['standard'] = scaler
        
        # Retorna os dados escalados
        return data
    

    # Método para seleção de features
    def feature_selection(self, X, y, k=20):
        """Seleção avançada de features"""

        # Log informativo
        print(f"Selecionando as {k} melhores features...")
        
        # Seleção baseada no tipo de problema
        if self.problem_type == 'classification':

            # Usa ANOVA F-test
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))

        else:

            # Usa F-test para regressão
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        
        # Aplica a seleção de features
        X_selected = selector.fit_transform(X, y)

        # Armazena o seletor
        self.feature_selector = selector
        
        # Obtém os nomes das features selecionadas
        selected_features = X.columns[selector.get_support()].tolist()

        # Log das features selecionadas
        print(f"Features selecionadas: {selected_features}")
        
        # Retorna dataframe com as features selecionadas
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    

    # Método principal do pipeline
    def process(self, data_path):
        """Pipeline completo de processamento"""

        # Log informativo
        print(f"Carregando dados de {data_path}...")

        # Carrega o CSV
        data = pd.read_csv(data_path)
        
        # Detecta automaticamente o tipo do problema
        if self.problem_type == 'auto':
            self.problem_type = self.detect_problem_type(data)

        # Log do tipo de problema
        print(f"Tipo de problema detectado: {self.problem_type}")
        
        # Aplica limpeza avançada
        data = self.advanced_cleaning(data)

        # Trata valores faltantes
        data = self.handle_missing_values(data)

        # Codifica variáveis categóricas
        data = self.encode_categorical(data)
        
        # Separa features e target
        if self.target_column:

            # Features
            X = data.drop(columns=[self.target_column])

            # Target
            y = data[self.target_column]

        else:

            # Assume a última coluna como target
            self.target_column = data.columns[-1]

            # Features
            X = data.iloc[:, :-1]

            # Target
            y = data.iloc[:, -1]
        
        # Aplica seleção de features
        X = self.feature_selection(X, y)
        
        # Aplica escalonamento
        X_scaled = self.scale_features(X)
        
        # Log final
        print(f"Processamento concluído. Shape final: {X_scaled.shape}")
        
        # Retorna dados processados, target e tipo do problema
        return X_scaled, y, self.problem_type
