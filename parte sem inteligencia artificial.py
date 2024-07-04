import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Carregar dados trimestrais e anuais
data_quarterly = pd.read_csv('house_price_index_quarterly.csv')
data_annual = pd.read_csv('house_price_index_annual.csv')

# Assumir valor base em euros para o ano de 2015
valor_base_euros = 200000  # Pode ser ajustado conforme necessário

# Converter índices para valores em euros
data_quarterly['VALUE_EUROS'] = data_quarterly['OBS_VALUE'] * (valor_base_euros / 100)
data_annual['VALUE_EUROS'] = data_annual['OBS_VALUE'] * (valor_base_euros / 100)

# Preparar dados para o modelo de previsão
def prepare_supervised_data(data, column_name):
    data_shifted = data[[column_name]].shift(1).dropna()
    X = data_shifted.values
    y = data[column_name][1:].values
    return X, y

X_q, y_q = prepare_supervised_data(data_quarterly, 'VALUE_EUROS')
X_a, y_a = prepare_supervised_data(data_annual, 'VALUE_EUROS')

# Treinar modelos
model_q = RandomForestRegressor(n_estimators=100, random_state=42)
model_q.fit(X_q, y_q)

model_a = RandomForestRegressor(n_estimators=100, random_state=42)
model_a.fit(X_a, y_a)

# Avaliar modelos
y_pred_q = model_q.predict(X_q)
y_pred_a = model_a.predict(X_a)
rmse_q = np.sqrt(mean_squared_error(y_q, y_pred_q))
rmse_a = np.sqrt(mean_squared_error(y_a, y_pred_a))
print(f'RMSE Trimestral: {rmse_q:.2f} €')
print(f'RMSE Anual: {rmse_a:.2f} €')

# Previsões futuras
future_steps_q = 20  # Próximos 5 anos trimestralmente
future_steps_a = 5   # Próximos 5 anos anualmente
future_prices_q = [y_q[-1]]
future_prices_a = [y_a[-1]]

for _ in range(future_steps_q):
    next_q = model_q.predict(np.array(future_prices_q[-1]).reshape(1, -1))[0]
    future_prices_q.append(next_q)

for _ in range(future_steps_a):
    next_a = model_a.predict(np.array(future_prices_a[-1]).reshape(1, -1))[0]
    future_prices_a.append(next_a)

future_prices_q = future_prices_q[1:]
future_prices_a = future_prices_a[1:]

# Visualizar previsões futuras com dados históricos
dates_quarterly = pd.date_range(start=data_quarterly['TIME_PERIOD'].values[-1], periods=future_steps_q+1, freq='Q')[1:]
dates_annual = pd.date_range(start=data_annual['TIME_PERIOD'].values[-1], periods=future_steps_a+1, freq='A')[1:]

plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(data_quarterly['TIME_PERIOD'], data_quarterly['VALUE_EUROS'], label='Histórico (Trimestral)')
plt.plot(dates_quarterly, future_prices_q, label='Previsão (Trimestral)', linestyle='--')
plt.xlabel('Data')
plt.ylabel('Valor (€)')
plt.title('Previsão dos Valores das Casas (Trimestral)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data_annual['TIME_PERIOD'], data_annual['VALUE_EUROS'], label='Histórico (Anual)')
plt.plot(dates_annual, future_prices_a, label='Previsão (Anual)', linestyle='--')
plt.xlabel('Data')
plt.ylabel('Valor (€)')
plt.title('Previsão dos Valores das Casas (Anual)')
plt.legend()

plt.tight_layout()
plt.show()
