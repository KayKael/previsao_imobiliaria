import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt

# Carregar dados dos arquivos CSV
data_quarterly = pd.read_csv('C:\\Users\\kaell\\Desktop\\projeto_imobiliario\\prc_hpi_q_linear.csv')
data_annual = pd.read_csv('C:\\Users\\kaell\\Desktop\\projeto_imobiliario\\prc_hpi_a_linear.csv')

# Verificar e corrigir os nomes das colunas
data_quarterly.columns = data_quarterly.columns.str.strip()
data_annual.columns = data_annual.columns.str.strip()

# Selecionar a coluna de interesse que contém os valores do índice de preços das casas
column_name = 'OBS_VALUE'

# Remover linhas com valores NaN ou infinitos para garantir a qualidade dos dados
data_quarterly = data_quarterly.replace([np.inf, -np.inf], np.nan)
data_quarterly = data_quarterly.dropna(subset=[column_name])
data_annual = data_annual.replace([np.inf, -np.inf], np.nan)
data_annual = data_annual.dropna(subset=[column_name])

# Filtrar dados para o intervalo de anos 2010 a 2023
data_quarterly = data_quarterly[(data_quarterly['TIME_PERIOD'] >= '2010-Q1') & (data_quarterly['TIME_PERIOD'] <= '2023-Q4')]
data_annual = data_annual[(data_annual['TIME_PERIOD'] >= 2010) & (data_annual['TIME_PERIOD'] <= 2023)]

# Função para preparar os dados para o modelo, criando as variáveis de entrada (X) e saída (y)
def prepare_data(data, n_lags):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Definir o número de lags (passos anteriores) a serem usados como entrada para o modelo
n_lags = 4

# Lista de países para análise (códigos dos países da UE)
countries_of_interest = [
    'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 
    'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 
    'NO', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK', 'TR'
]

# Função para treinar e avaliar o modelo
def train_and_evaluate(data, country, freq):
    # Filtrar dados pelo país
    country_data = data[data['geo'] == country]
    
    # Preparar os dados
    X, y = prepare_data(country_data[column_name].values, n_lags)
    
    # Dividir os dados em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizar os dados para melhorar a performance do modelo
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Callback personalizado para monitorar o R²
    class R2EarlyStopping(Callback):
        def __init__(self, validation_data, patience=10):
            super(R2EarlyStopping, self).__init__()
            self.patience = patience
            self.best_weights = None
            self.best_r2 = -np.inf
            self.wait = 0
            self.validation_data = validation_data
        
        def on_epoch_end(self, epoch, logs=None):
            X_val, y_val = self.validation_data
            y_pred = self.model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            if r2 > self.best_r2:
                self.best_r2 = r2
                self.best_weights = self.model.get_weights()
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)
                    print(f'\nEarly stopping at epoch {epoch+1} with best R²: {self.best_r2:.4f}')
    
    # Função para construir e treinar o modelo
    def train_model(X_train, y_train, X_val, y_val):
        model = Sequential([
            Input(shape=(X_train.shape[1],)),  # Camada de entrada
            Dense(64, activation='relu'),  # Primeira camada densa com 64 neurônios e ativação ReLU
            Dropout(0.3),  # Dropout para evitar overfitting
            Dense(32, activation='relu'),  # Segunda camada densa com 32 neurônios e ativação ReLU
            Dropout(0.3),  # Dropout adicional
            Dense(16, activation='relu'),  # Terceira camada densa com 16 neurônios e ativação ReLU
            Dense(1)  # Camada de saída com um neurônio para prever o índice de preços
        ])
        
        # Compilar o modelo
        model.compile(optimizer='adam', loss='mean_squared_error')  # Define o otimizador e a função de perda

        # Adicionar callbacks para R²EarlyStopping e ModelCheckpoint
        early_stopping = R2EarlyStopping(validation_data=(X_val, y_val), patience=20)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

        # Treinar o modelo
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32,
                            callbacks=[early_stopping, model_checkpoint])  # Treina o modelo por 100 épocas
        
        return model, history
    
    # Treinar o modelo
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    # Avaliar o modelo usando o conjunto de teste
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, mae, r2
    
    mse, mae, r2 = evaluate_model(model, X_test, y_test)
    
    # Visualizar o desempenho dos modelos durante o treinamento
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'Dados {freq} - {country}')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Salvar o modelo treinado
    model.save(f'house_price_model_{freq}_{country}.keras')
    
    # Função para carregar o modelo salvo e fazer previsões futuras
    def predict_future_prices(model_path, recent_data, n_lags, scaler):
        model = load_model(model_path)
        recent_data_scaled = scaler.transform(recent_data[-n_lags:].reshape(1, -1))
        return model.predict(recent_data_scaled)
    
    # Previsão para os próximos 5 anos com base no crescimento dos preços das casas
    def forecast_growth(data, model_path, n_periods, n_lags, scaler, frequency='annual'):
        recent_data = data[-n_lags:]
        future_prices = []
        for i in range(n_periods):
            future_price = predict_future_prices(model_path, recent_data, n_lags, scaler)
            future_prices.append(future_price[0][0])
            recent_data = np.append(recent_data, future_price)[-n_lags:]
        return future_prices
    
    # Definir os valores de referência
    price_reference_euros = 200000  # Exemplo de preço médio de referência em euros
    hpi_reference_value = 100  # Exemplo de valor do HPI de referência
    
    # Previsões futuras para os próximos 5 anos
    n_periods = 20 if freq == 'quarterly' else 5
    future_prices = forecast_growth(country_data[column_name].values, f'house_price_model_{freq}_{country}.keras', n_periods, n_lags, scaler, frequency=freq)
    
    # Converter as previsões do HPI para valores reais em euros
    future_prices_euros = [price / hpi_reference_value * price_reference_euros for price in future_prices]
    
    # Mostrar previsões futuras de forma clara
    print(f'\nPrevisão dos valores reais das casas para os próximos 5 anos ({freq} - {country}):')
    if freq == 'quarterly':
        for i, price in enumerate(future_prices_euros, start=1):
            print(f'Trimestre {i}: {price:.2f} euros')
    else:
        for i, price in enumerate(future_prices_euros, start=1):
            print(f'Ano {i}: {price:.2f} euros')
    
    return mse, mae, r2, future_prices

# Função para permitir que o usuário selecione um país e exiba as previsões
def user_selection(countries_of_interest, data_quarterly, data_annual):
    # Exibir a lista de países disponíveis
    print("Selecione um país:")
    for idx, country in enumerate(countries_of_interest, 1):
        print(f"{idx}. {country}")
    
    # Solicitar que o usuário selecione um país
    selected_index = int(input("\nDigite o número correspondente ao país desejado: ")) - 1
    selected_country = countries_of_interest[selected_index]
    
    # Treinar e avaliar o modelo para o país selecionado (dados trimestrais)
    print(f'\n\nProcessando país: {selected_country}')
    print('\nDados Trimestrais:')
    mse_q, mae_q, r2_q, future_prices_q = train_and_evaluate(data_quarterly, selected_country, 'quarterly')
    
    # Treinar e avaliar o modelo para o país selecionado (dados anuais)
    print('\nDados Anuais:')
    mse_a, mae_a, r2_a, future_prices_a = train_and_evaluate(data_annual, selected_country, 'annual')
    
    # Treinar e avaliar o modelo geral para a União Europeia (média dos países)
    print('\n\nProcessando União Europeia:')
    mse_q, mae_q, r2_q, future_prices_q = train_and_evaluate(data_quarterly, 'EU27_2020', 'quarterly')
    mse_a, mae_a, r2_a, future_prices_a = train_and_evaluate(data_annual, 'EU27_2020', 'annual')

# Permitir que o usuário selecione um país e exiba as previsões
user_selection(countries_of_interest, data_quarterly, data_annual)
