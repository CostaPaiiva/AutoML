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
        # Define um método para normalizar e padronizar as features.
        """Normalização e padronização das features - VERSÃO SIMPLIFICADA"""
        # Imprime uma mensagem indicando o início do processo de escalonamento.
        print("Escalando features...")

        # Cria uma cópia do DataFrame de entrada para evitar modificar o original.
        data = data.copy()
        # Seleciona todas as colunas numéricas no DataFrame e as lista.
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Verifica se a coluna alvo foi definida e se ela está entre as colunas numéricas.
        if self.target_column and self.target_column in numeric_cols:
            # Remove a coluna alvo da lista de colunas numéricas para não ser escalada.
            numeric_cols.remove(self.target_column)

        # Verifica se há colunas numéricas restantes para serem escaladas.
        if len(numeric_cols) > 0:
            # Inicializa uma lista para armazenar as colunas que realmente serão escaladas.
            cols_to_scale = []
            # Itera sobre cada coluna numérica.
            for col in numeric_cols:
                # Verifica se o desvio padrão da coluna é maior que zero (para evitar divisão por zero no scaler).
                if data[col].std() > 0:
                    # Adiciona a coluna à lista de colunas a serem escaladas.
                    cols_to_scale.append(col)

            # Verifica se há colunas válidas para serem escaladas.
            if cols_to_scale:
                # Instancia um objeto StandardScaler.
                scaler = StandardScaler()
                # Ajusta o scaler aos dados das colunas selecionadas e os transforma.
                data_scaled = scaler.fit_transform(data[cols_to_scale])
                # Atribui os dados escalados de volta às colunas originais no DataFrame.
                data[cols_to_scale] = data_scaled
                # Armazena o scaler ajustado no dicionário 'scalers' da instância, usando a chave 'standard'.
                self.scalers['standard'] = scaler

        # Retorna o DataFrame com as features numéricas escaladas.
        return data

    def feature_selection(self, X, y, k='auto'):
        """Seleção avançada de features - VERSÃO SIMPLIFICADA"""
        # Imprime uma mensagem indicando o início do processo de seleção de features.
        print("Selecionando melhores features...")

        # Verifica se o parâmetro 'k' para o número de features a selecionar está definido como 'auto'.
        if k == 'auto':
            # Se 'k' for 'auto', define 'k' como o mínimo entre 20 e o número total de colunas (features) em X.
            k = min(20, X.shape[1])

        # Verifica se o número de features a selecionar ('k') é maior ou igual ao número total de features em X.
        if k >= X.shape[1]:
            # Imprime uma mensagem informando que não há features suficientes para seleção e todas as features serão usadas.
            print(f"Não há features suficientes para seleção. Usando todas as {X.shape[1]} features.")
            # Retorna o DataFrame X original sem alteração.
            return X

        # Verifica se o tipo de problema detectado é 'classification'.
        if self.problem_type == 'classification':
            # Se for classificação, inicializa um seletor SelectKBest usando f_classif como função de pontuação.
            selector = SelectKBest(score_func=f_classif, k=k)
        # Caso contrário (se for regressão).
        else:
            # Inicializa um seletor SelectKBest usando f_regression como função de pontuação.
            selector = SelectKBest(score_func=f_regression, k=k)

        # Inicia um bloco try para lidar com possíveis erros durante a seleção de features.
        try:
            # Aplica o seletor aos dados X e y para ajustar o modelo e transformar os dados, selecionando as melhores features.
            X_selected = selector.fit_transform(X, y)
            # Armazena o seletor ajustado na instância da classe para uso posterior.
            self.feature_selector = selector

            # Obtém os nomes das colunas (features) que foram selecionadas.
            selected_features = X.columns[selector.get_support()].tolist()
            # Imprime o número de features selecionadas em relação ao total.
            print(f"Features selecionadas: {len(selected_features)}/{X.shape[1]}")

            # Retorna um novo DataFrame contendo apenas as features selecionadas, mantendo os nomes das colunas e o índice original.
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        # Captura qualquer exceção que ocorra durante o processo de seleção de features.
        except Exception:
            # Imprime uma mensagem de erro indicando que a seleção de features falhou.
            print("Erro na seleção de features. Usando todas as features.")
            # Retorna o DataFrame X original sem alteração em caso de erro.
            return X

    def process(self, data_path):
        """Pipeline completo de processamento - VERSÃO ROBUSTA"""
        print("Carregando dados...") # Imprime uma mensagem indicando o início do carregamento dos dados.

        try:
            if isinstance(data_path, str): # Verifica se o 'data_path' é uma string (caminho de arquivo).
                data = pd.read_csv(data_path) # Carrega os dados de um arquivo CSV usando o caminho.
            else:
                data = pd.read_csv(data_path) # Caso contrário, assume que é um objeto de arquivo e carrega o CSV.

            print(f"Dados carregados: {data.shape[0]} linhas, {data.shape[1]} colunas") # Imprime as dimensões dos dados carregados.

            if not self.target_column: # Verifica se a coluna alvo não foi definida.
                self.target_column = data.columns[-1] # Se não, define a última coluna como a coluna alvo.

            if self.target_column not in data.columns: # Verifica se a coluna alvo especificada existe nos dados.
                raise ValueError(f"Coluna target '{self.target_column}' não encontrada nos dados.") # Levanta um erro se a coluna alvo não for encontrada.

            if self.problem_type == 'auto': # Verifica se o tipo de problema deve ser detectado automaticamente.
                self.problem_type = self.detect_problem_type(data) # Detecta o tipo de problema (classificação ou regressão).
            print(f"Tipo de problema detectado: {self.problem_type}") # Imprime o tipo de problema detectado.

            X = data.drop(columns=[self.target_column]).copy() # Cria um DataFrame X (features) removendo a coluna alvo.
            y = data[self.target_column].copy() # Cria uma Série y (alvo) com a coluna alvo.

            X = self.advanced_cleaning(X) # Aplica a função de limpeza avançada aos dados X.
            X = self.handle_missing_values(X, strategy='simple') # Trata os valores faltantes nos dados X usando a estratégia 'simple'.
            X = self.encode_categorical(X) # Codifica as variáveis categóricas nos dados X.

            y = y.loc[X.index] # Alinha o índice de y com o índice de X, garantindo que as linhas correspondam após as operações em X.

            if X.shape[1] > 10: # Verifica se há mais de 10 features para considerar a seleção.
                X = self.feature_selection(X, y) # Realiza a seleção de features em X usando y como alvo.

            X = self.scale_features(X) # Escala (normaliza/padroniza) as features em X.

            print(f"✅ Processamento concluído. Shape final: {X.shape}") # Imprime uma mensagem de sucesso com as dimensões finais de X.

            return X, y, self.problem_type # Retorna as features processadas, o alvo e o tipo de problema.

        except Exception as e: # Captura qualquer exceção que ocorra durante o processamento.
            print(f"❌ Erro no processamento: {str(e)}") # Imprime uma mensagem de erro.
            return self.simple_process(data_path) # Em caso de erro, chama a função de processamento simples como fallback.

    def simple_process(self, data_path):
        """Processamento simples de fallback"""
        # Imprime uma mensagem indicando que o processamento simples de fallback está sendo usado
        print("Usando processamento simples de fallback...")

        try:
            # Verifica se o data_path é uma string (caminho do arquivo)
            if isinstance(data_path, str):
                # Carrega o CSV do caminho especificado em um DataFrame pandas
                data = pd.read_csv(data_path)
            else:
                # Se não for uma string, assume que é um objeto de arquivo e carrega o CSV
                data = pd.read_csv(data_path)

            # Verifica se a coluna alvo (target_column) está definida e presente nos dados
            if self.target_column and self.target_column in data.columns:
                # Separa as features (X) removendo a coluna alvo
                X = data.drop(columns=[self.target_column]).copy()
                # Atribui a coluna alvo (y)
                y = data[self.target_column].copy()
            else:
                # Se a coluna alvo não estiver definida ou não for encontrada, assume a última coluna como alvo
                self.target_column = data.columns[-1]
                # Separa as features (X) como todas as colunas exceto a última
                X = data.iloc[:, :-1].copy()
                # Atribui a última coluna como alvo (y)
                y = data.iloc[:, -1].copy()

            # Detecta o tipo de problema (classificação ou regressão) com base na coluna alvo
            if y.dtype == 'object' or str(y.dtype) == 'category' or len(y.unique()) <= 10:
                # Se o tipo for objeto/categoria ou tiver poucas classes únicas, é classificação
                problem_type = 'classification'
            else:
                # Caso contrário, é regressão
                problem_type = 'regression'

            # Remove colunas de X que tenham mais de 50% de valores ausentes
            X = X.dropna(axis=1, thresh=int(len(X) * 0.5))

            # Seleciona as colunas numéricas no DataFrame X
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            # Verifica se existem colunas numéricas para processar
            if numeric_cols:
                # Preenche os valores nulos nas colunas numéricas com a média de cada coluna
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

            # Categóricas
            # Seleciona as colunas categóricas restantes no DataFrame X
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            # Itera sobre cada coluna categórica
            for col in categorical_cols:
                # Preenche quaisquer valores nulos na coluna com a string 'missing'
                X[col] = X[col].fillna('missing')
                # Converte a coluna categórica em valores numéricos usando factorize (Label Encoding)
                X[col] = pd.factorize(X[col])[0]

            # Imprime uma mensagem de sucesso após o processamento simples, mostrando a forma final dos dados
            print(f"✅ Processamento simples concluído. Shape final: {X.shape}")

            # Retorna as features processadas (X), o target (y) e o tipo de problema detectado
            return X, y, problem_type

        # Captura qualquer exceção que ocorra durante o processamento simples
        except Exception as e:
            # Imprime uma mensagem de erro se o processamento simples falhar
            print(f"❌ Erro no processamento simples: {str(e)}")
            # Relança a exceção para que o chamador possa lidar com ela
            raise

            # Seleciona as colunas numéricas no DataFrame X
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            # Verifica se existem colunas numéricas para processar
            if numeric_cols:
                # Preenche os valores nulos nas colunas numéricas com a média de cada coluna
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

            # Categóricas
            # Seleciona as colunas categóricas restantes no DataFrame X
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            # Itera sobre cada coluna categórica
            for col in categorical_cols:
                # Preenche quaisquer valores nulos na coluna com a string 'missing'
                X[col] = X[col].fillna('missing')
                # Converte a coluna categórica em valores numéricos usando factorize (Label Encoding)
                X[col] = pd.factorize(X[col])[0]

            # Imprime uma mensagem de sucesso após o processamento simples, mostrando a forma final dos dados
            print(f"✅ Processamento simples concluído. Shape final: {X.shape}")

            # Retorna as features processadas (X), o target (y) e o tipo de problema detectado
            return X, y, problem_type

        # Captura qualquer exceção que ocorra durante o processamento simples
        except Exception as e:
            # Imprime uma mensagem de erro se o processamento simples falhar
            print(f"❌ Erro no processamento simples: {str(e)}")
            # Relança a exceção para que o chamador possa lidar com ela
            raise