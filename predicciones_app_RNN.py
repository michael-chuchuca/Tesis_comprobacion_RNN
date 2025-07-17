import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# -----------------------
# Cargar y preparar datos
# -----------------------
df = pd.read_excel("Items_Morante.xlsx")
df['FECHA_VENTA'] = pd.to_datetime(df['FECHA_VENTA'])
df = df.sort_values('FECHA_VENTA')

item = "CODIGO_EJEMPLO"  # Reemplaza con un código real
df_item = df[df['ITEM'] == item].copy()

# Agrupar semanal
df_item = df_item.groupby(pd.Grouper(key='FECHA_VENTA', freq='W')).agg({'CANTIDAD_VENDIDA': 'sum'}).reset_index()
df_item = df_item.rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})

# Escalar
scaler = MinMaxScaler()
df_item['y_scaled'] = scaler.fit_transform(df_item[['y']])

# Crear secuencias
def crear_secuencias(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 4  # Usamos 4 semanas para predecir 1
X, y = crear_secuencias(df_item['y_scaled'].values, n_steps)

# Reshape para LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# -----------------------
# Modelo LSTM
# -----------------------
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# -----------------------
# Predicción
# -----------------------
X_input = df_item['y_scaled'].values[-n_steps:]
predicciones = []
for _ in range(6):  # Predecimos 6 semanas = 42 días aprox
    x_input = X_input[-n_steps:].reshape((1, n_steps, 1))
    yhat = model.predict(x_input, verbose=0)
    predicciones.append(yhat[0, 0])
    X_input = np.append(X_input, yhat[0, 0])

# Inversión de escala
predicciones = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1)).flatten()

# -----------------------
# Visualización
# -----------------------
fechas_futuras = pd.date_range(start=df_item['ds'].iloc[-1] + pd.Timedelta(days=7), periods=6, freq='W')
plt.figure(figsize=(10,5))
plt.plot(df_item['ds'], df_item['y'], label="Real")
plt.plot(fechas_futuras, predicciones, label="LSTM", color="orange")
plt.legend()
plt.title("Pronóstico con RNN (LSTM)")
plt.xlabel("Fecha")
plt.ylabel("Cantidad vendida")
plt.grid()
plt.show()
