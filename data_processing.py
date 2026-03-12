# Classe AdvancedDataProcessor: pipeline de pré-processamento de dados.
# 1. Detecta automaticamente se o problema é classificação ou regressão.
# 2. Faz limpeza avançada: remove duplicatas, trata infinitos e colunas com muitos NaN.
# 3. Aplica tratamento de outliers em colunas numéricas (clipping).
# 4. Cria novas features (produto, divisão, média, desvio padrão).
# 5. Trata valores faltantes com mediana/moda ou imputação.
# 6. Codifica variáveis categóricas (One-Hot para poucas categorias, LabelEncoder para muitas).
# 7. Escala variáveis numéricas com StandardScaler.
# 8. Seleciona melhores features com SelectKBest.
# 9. Executa todo o pipeline no método process(), retornando X, y e tipo de problema.


import pandas as pd  # Importa a biblioteca pandas, fundamental para manipulação e análise de dados tabulares.
import numpy as np  # Importa a biblioteca numpy, usada para operações numéricas e arrays.
# Importa classes de pré-processamento do scikit-learn:
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder  # LabelEncoder para codificação de rótulos, StandardScaler para padronização, OneHotEncoder para codificação one-hot.
# Importa classes de imputação do scikit-learn para tratamento de valores ausentes:
from sklearn.impute import SimpleImputer, KNNImputer  # SimpleImputer para estratégias básicas (média, mediana, moda), KNNImputer para imputação baseada em vizinhos.
# Importa classes e funções para seleção de features do scikit-learn:
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression  # SelectKBest para selecionar as K melhores features, e diferentes funções de pontuação para classificação (f_classif, mutual_info_classif) e regressão (f_regression, mutual_info_regression).
import warnings  # Importa o módulo warnings para gerenciar avisos.
warnings.filterwarnings('ignore')  # Configura para ignorar todos os avisos durante a execução do código.


class AdvancedDataProcessor:  # Define uma classe chamada AdvancedDataProcessor para encapsular as funcionalidades de processamento de dados.
    def __init__(self, target_column=None, problem_type='auto'):  # Define o método construtor da classe, que é chamado ao criar uma nova instância.
        self.target_column = target_column  # Inicializa o atributo target_column com o nome da coluna alvo (variável dependente), pode ser None se não for especificado.
        self.problem_type = problem_type  # Inicializa o atributo problem_type (tipo de problema: 'classification', 'regression' ou 'auto' para detecção automática).
        self.preprocessors = {}  # Inicializa um dicionário vazio para armazenar pré-processadores ajustados (não usados explicitamente no código fornecido, mas útil para expansão).
        self.encoders = {}  # Inicializa um dicionário vazio para armazenar LabelEncoders ajustados para colunas categóricas.
        self.scalers = {}  # Inicializa um dicionário vazio para armazenar Scalers ajustados para colunas numéricas.
        self.feature_selector = None  # Inicializa o atributo feature_selector como None, que armazenará o seletor de features ajustado.
        self.imputer = None  # Inicializa o atributo imputer como None, que armazenará o imputer de valores ausentes ajustado (não usado explicitamente, mas útil para expansão).

    def detect_problem_type(self, data):
        """Detecta automaticamente se é classificação ou regressão"""
        # Verifica se a coluna alvo foi definida e se ela existe no DataFrame.
        if self.target_column and self.target_column in data.columns:
            # Seleciona a coluna alvo do DataFrame.
            target = data[self.target_column]

            # Inicia um bloco try-except para lidar com possíveis erros durante a detecção do tipo de problema.
            try:
                # Tenta converter a coluna alvo para um tipo numérico, transformando erros em NaN.
                target_numeric = pd.to_numeric(target, errors='coerce')
                # Calcula a proporção de valores não nulos após a tentativa de conversão numérica.
                numeric_ratio = target_numeric.notna().mean()

                # Se o tipo de dado original da coluna alvo for 'object' (geralmente string) ou 'category', é classificação.
                if target.dtype == 'object' or str(target.dtype) == 'category':
                    # Retorna 'classification' se for uma coluna categórica.
                    return 'classification'

                # Se a proporção de valores numéricos válidos for menor que 80%, assume-se que é classificação.
                if numeric_ratio < 0.8:
                    # Retorna 'classification' se muitos valores não puderam ser convertidos para numérico.
                    return 'classification'

                # Conta o número de valores únicos na coluna alvo, ignorando NaN.
                unique_count = target.nunique(dropna=True)
                # Se o número de valores únicos for menor ou igual a 10, é classificação (critério comum para classes discretas).
                if unique_count <= 10:
                    # Retorna 'classification' se houver poucas classes únicas.
                    return 'classification'

                # Se nenhuma das condições acima for atendida, assume-se que é um problema de regressão.
                return 'regression'
            # Captura qualquer exceção que possa ocorrer no bloco try.
            except Exception:
                # Em caso de erro, verifica o tipo de dado original ou o número de únicos como fallback.
                if target.dtype == 'object' or len(target.unique()) <= 10:
                    # Retorna 'classification' se for objeto ou tiver poucas classes únicas.
                    return 'classification'
                # Caso contrário, retorna 'regression'.
                return 'regression'

        # Se a coluna alvo não foi definida ou não encontrada, retorna 'auto' para indicar que não foi possível detectar.
        return 'auto'

    def advanced_cleaning(self, data):
        # Define um método para realizar a limpeza avançada dos dados.
        """Limpeza avançada dos dados"""
        # Imprime uma mensagem indicando o início do processo de limpeza avançada.
        print("Realizando limpeza avançada dos dados...")

        # Cria uma cópia do DataFrame de entrada para evitar modificar o original.
        data_cleaned = data.copy()

        # Armazena o número original de linhas do DataFrame antes da remoção de duplicatas.
        original_len = len(data_cleaned)
        # Remove linhas duplicadas do DataFrame, mantendo apenas a primeira ocorrência.
        data_cleaned = data_cleaned.drop_duplicates()
        # Imprime o número de duplicatas que foram removidas.
        print(f"Duplicatas removidas: {original_len - len(data_cleaned)}")

        # Substitui todos os valores infinitos positivos e negativos por NaN (Not a Number).
        data_cleaned = data_cleaned.replace([np.inf, -np.inf], np.nan)

        # Calcula a porcentagem de valores ausentes para cada coluna.
        missing_percentage = data_cleaned.isnull().mean()
        # Identifica as colunas que têm mais de 50% de valores ausentes.
        columns_to_drop = missing_percentage[missing_percentage > 0.5].index.tolist()

        # Verifica se há colunas a serem removidas.
        if columns_to_drop:
            # Remove as colunas identificadas com alta taxa de valores ausentes.
            data_cleaned = data_cleaned.drop(columns=columns_to_drop)
            # Imprime os nomes das colunas que foram removidas.
            print(f"Colunas removidas (alta taxa de missing): {columns_to_drop}")

        # Verifica se o número de linhas no DataFrame limpo é menor que 10000.
        if len(data_cleaned) < 10000:
            # Seleciona todas as colunas numéricas no DataFrame.
            numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns.tolist()

            # Verifica se a coluna alvo foi definida e se ela está entre as colunas numéricas.
            if self.target_column in numeric_cols:
                # Remove a coluna alvo da lista de colunas numéricas para não ser incluída nas operações subsequentes (como tratamento de outliers).
                numeric_cols.remove(self.target_column)

            # Itera sobre cada coluna numérica identificada.
            for col in numeric_cols:
                # Extrai a série de dados da coluna atual para facilitar o acesso.
                series = data_cleaned[col]

                # Verifica se há pelo menos 5 valores não nulos na série para calcular quartis de forma significativa.
                if series.notna().sum() < 5:
                    # Se não houver valores suficientes, pula para a próxima coluna.
                    continue

                # Calcula o primeiro quartil (Q1) da série.
                Q1 = series.quantile(0.25)
                # Calcula o terceiro quartil (Q3) da série.
                Q3 = series.quantile(0.75)
                # Calcula o Intervalo Interquartil (IQR), que é a diferença entre Q3 e Q1.
                IQR = Q3 - Q1

                # Verifica se o IQR é NaN (indica que não há variabilidade ou dados suficientes) ou zero.
                if pd.isna(IQR) or IQR == 0:
                    # Se for, pula para a próxima coluna, pois não é possível calcular limites de outliers.
                    continue

                # Calcula o limite inferior para detecção de outliers.
                lower_bound = Q1 - 1.5 * IQR
                # Calcula o limite superior para detecção de outliers.
                upper_bound = Q3 + 1.5 * IQR

                # Cria uma máscara booleana para identificar os valores que são considerados outliers (abaixo do limite inferior ou acima do superior).
                outlier_mask = (series < lower_bound) | (series > upper_bound)
                # Calcula a proporção de outliers na coluna.
                outlier_ratio = outlier_mask.mean()

                # Verifica se a proporção de outliers está entre 0 (não há outliers) e 0.1 (até 10% de outliers).
                if 0 < outlier_ratio < 0.1:
                    # Se a condição for atendida, aplica o clipping: valores abaixo do lower_bound são substituídos por lower_bound,
                    # e valores acima do upper_bound são substituídos por upper_bound.
                    data_cleaned[col] = series.clip(lower=lower_bound, upper=upper_bound)

        # Verifica novamente se o número de linhas no DataFrame limpo é menor que 10000.
        if len(data_cleaned) < 10000:
            # Se for, aplica a engenharia de features ao DataFrame.
            data_cleaned = self.feature_engineering(data_cleaned)

        # Retorna o DataFrame com a limpeza avançada aplicada.
        return data_cleaned

    def feature_engineering(self, data):
        # Define um método para realizar engenharia de features no DataFrame.
        """Engenharia de features avançada"""
        # Imprime uma mensagem indicando o início do processo de engenharia de features.
        print("Aplicando engenharia de features...")

        # Cria uma cópia do DataFrame de entrada para evitar modificar o original.
        data = data.copy()

        # Seleciona todas as colunas numéricas no DataFrame.
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        # Verifica se a coluna alvo foi definida e se ela está entre as colunas numéricas.
        if self.target_column in numeric_cols:
            # Remove a coluna alvo da lista de colunas numéricas para não ser usada na engenharia de features.
            numeric_cols.remove(self.target_column)

        # Limita o número de colunas numéricas a serem consideradas para engenharia de features a um máximo de 5.
        numeric_cols = numeric_cols[:5]

        # Verifica se há pelo menos duas colunas numéricas para criar features de interação.
        if len(numeric_cols) >= 2:
            # Itera sobre um subconjunto das colunas numéricas (até 3 primeiras).
            for i in range(min(len(numeric_cols), 3)):
                # Itera sobre as colunas restantes para formar pares (até a 4ª coluna).
                for j in range(i + 1, min(len(numeric_cols), 4)):
                    # Obtém os nomes das duas colunas para interação.
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    # Cria uma nova feature que é o produto das duas colunas.
                    data[f'{col1}_x_{col2}'] = data[col1] * data[col2]

                    # Cria uma máscara para identificar onde a segunda coluna não é zero, para evitar divisão por zero.
                    mask = data[col2] != 0
                    # Cria uma nova feature que é a divisão da primeira pela segunda coluna, tratando a divisão por zero.
                    data[f'{col1}_div_{col2}'] = np.where(mask, data[col1] / data[col2], 0)

        # Verifica se há mais de uma coluna numérica para calcular estatísticas agregadas.
        if len(numeric_cols) > 1:
            # Cria uma nova feature que é a média de todas as colunas numéricas para cada linha.
            data['mean_features'] = data[numeric_cols].mean(axis=1)
            # Cria uma nova feature que é o desvio padrão de todas as colunas numéricas para cada linha.
            data['std_features'] = data[numeric_cols].std(axis=1)

        # Retorna o DataFrame com as novas features criadas.
        return data

    def handle_missing_values(self, data, strategy='simple'):
        # Define um método para tratar valores faltantes no DataFrame.
        """Tratamento de valores faltantes - VERSÃO CORRIGIDA"""
        # Imprime uma mensagem indicando o início do processo de tratamento de valores faltantes.
        print("Tratando valores faltantes...")

        # Cria uma cópia do DataFrame de entrada para evitar modificar o original.
        data = data.copy()

        # Verifica se a estratégia de tratamento de valores faltantes é 'simple'.
        if strategy == 'simple':
            # Seleciona as colunas numéricas no DataFrame.
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            # Seleciona as colunas categóricas (tipo 'object' ou 'category') no DataFrame.
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

            # Itera sobre cada coluna numérica.
            for col in numeric_cols:
                # Verifica se a coluna contém algum valor nulo.
                if data[col].isnull().any():
                    # Calcula a mediana dos valores na coluna.
                    median_value = data[col].median()
                    # Preenche os valores nulos na coluna com a mediana calculada.
                    data[col] = data[col].fillna(median_value)

            # Itera sobre cada coluna categórica.
            for col in categorical_cols:
                # Verifica se a coluna contém algum valor nulo.
                if data[col].isnull().any():
                    # Calcula o valor mais frequente (moda) na coluna.
                    mode_value = data[col].mode()
                    # Verifica se a moda não está vazia (ou seja, se há um valor mais frequente).
                    if not mode_value.empty:
                        # Preenche os valores nulos na coluna com o primeiro valor da moda.
                        data[col] = data[col].fillna(mode_value.iloc[0])
                    # Caso a moda esteja vazia (ex: todos NaN ou múltiplos modos sem um único dominante).
                    else:
                        # Preenche os valores nulos com uma string vazia.
                        data[col] = data[col].fillna('')

            # Retorna o DataFrame com os valores faltantes tratados pela estratégia 'simple'.
            return data

        # Se a estratégia não for 'simple' (código existente assume alguma outra estratégia, como 'advanced' ou padrão).
        else:
            # Seleciona as colunas numéricas no DataFrame.
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            # Seleciona as colunas categóricas (tipo 'object' ou 'category') no DataFrame.
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

            # Verifica se existem colunas numéricas para tratar.
            if numeric_cols:
                # Extrai os valores das colunas numéricas em um array NumPy.
                numeric_data = data[numeric_cols].values
                # Instancia um SimpleImputer para preencher valores nulos com a média.
                imputer_num = SimpleImputer(strategy='mean')
                # Ajusta o imputer aos dados numéricos e os transforma, preenchendo os nulos.
                numeric_data_imputed = imputer_num.fit_transform(numeric_data)
                # Atribui os dados numéricos imputados de volta às colunas originais no DataFrame.
                data[numeric_cols] = numeric_data_imputed

            # Verifica se existem colunas categóricas para tratar.
            if categorical_cols:
                # Itera sobre cada coluna categórica.
                for col in categorical_cols:
                    # Calcula o valor mais frequente (moda) na coluna.
                    mode_value = data[col].mode()
                    # Verifica se a moda não está vazia.
                    if not mode_value.empty:
                        # Preenche os valores nulos na coluna com o primeiro valor da moda.
                        data[col] = data[col].fillna(mode_value.iloc[0])
                    # Caso a moda esteja vazia.
                    else:
                        # Preenche os valores nulos com a string 'missing'.
                        data[col] = data[col].fillna('missing')

            # Retorna o DataFrame com os valores faltantes tratados pela estratégia 'else'.
            return data

    def encode_categorical(self, data):
        # Imprime uma mensagem indicando o início do processo de codificação.
        """Codificação avançada de variáveis categóricas - VERSÃO SIMPLIFICADA"""
        # Imprime uma mensagem para o console, informando que a codificação de variáveis categóricas está começando.
        print("Codificando variáveis categóricas...")

        # Cria uma cópia do DataFrame de entrada para evitar modificar o original.
        data = data.copy()
        # Identifica todas as colunas no DataFrame que são do tipo 'object' (geralmente strings) ou 'category'.
        # Essas são as colunas categóricas que precisarão ser codificadas.
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Itera sobre cada coluna identificada como categórica.
        for col in categorical_cols:
            # Verifica se a coluna atual não é a coluna alvo (target_column), pois a coluna alvo não deve ser codificada desta forma.
            if col != self.target_column:
                # Calcula o número de valores únicos na coluna atual, incluindo valores NaN (valores faltantes).
                nunique = data[col].nunique(dropna=False)

                # Verifica se o número de valores únicos é menor ou igual a 10.
                if nunique <= 10:
                    # Se houver 10 ou menos valores únicos, aplica One-Hot Encoding.
                    # Cria variáveis dummy para a coluna, convertendo categorias em colunas binárias.
                    # 'prefix=col' adiciona o nome da coluna original como prefixo às novas colunas dummy.
                    # 'drop_first=True' evita a multicolinearidade, removendo a primeira categoria de cada grupo.
                    dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                    # Concatena o DataFrame original (sem a coluna categórica original) com as novas colunas dummy.
                    data = pd.concat([data.drop(columns=[col]), dummies], axis=1)
                # Se o número de valores únicos for maior que 10.
                else:
                    # Aplica Label Encoding para colunas com muitas categorias únicas (alta cardinalidade).
                    # Instancia um objeto LabelEncoder.
                    le = LabelEncoder()
                    # Preenche os valores NaN na coluna com a string 'missing' e converte a coluna para o tipo string.
                    # Isso garante que o LabelEncoder possa processar todos os valores, incluindo os que eram NaN.
                    col_data = data[col].fillna('missing').astype(str)
                    # Ajusta o LabelEncoder aos dados da coluna e os transforma em valores numéricos inteiros.
                    data[col] = le.fit_transform(col_data)
                    # Armazena o LabelEncoder ajustado no dicionário 'encoders' da instância, usando o nome da coluna como chave.
                    # Isso permite que o mesmo encoder seja usado para transformar novos dados posteriormente.
                    self.encoders[col] = le

        # Retorna o DataFrame com as variáveis categóricas codificadas.
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