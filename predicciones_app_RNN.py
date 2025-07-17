import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet_lite import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# -----------------------
# CONFIGURACIÓN GENERAL
# -----------------------
st.set_page_config(page_title="Predicción de Inventario", layout="wide")
st.markdown("<h1 style='text-align: center;'>Predicción de Demanda Semanal</h1>", unsafe_allow_html=True)

# -----------------------
# FUNCIONES
# -----------------------
@st.cache_data
def cargar_datos(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    df['FECHA_VENTA'] = pd.to_datetime(df['FECHA_VENTA'])
    df = df.sort_values(by='FECHA_VENTA')
    df['ITEM_DESC'] = df['ITEM'].astype(str) + " - " + df['DESCRIPCION'].astype(str)
    return df

def preparar_serie_semanal(df_item_raw):
    df_agg = df_item_raw.groupby(pd.Grouper(key='FECHA_VENTA', freq='W')).agg({
        'CANTIDAD_VENDIDA': 'sum'
    }).reset_index()
    df_agg = df_agg.rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})
    return df_agg

def entrenar_prophet(df, periodo_semanas):
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periodo_semanas, freq='W')
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))
    return forecast[['ds', 'yhat']]

def entrenar_lstm(df_semanal, periodo_semanas):
    df_scaled = df_semanal.copy()
    scaler = MinMaxScaler()
    df_scaled['y'] = scaler.fit_transform(df_scaled[['y']])

    def crear_secuencias(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i+n_steps])
            y.append(data[i+n_steps])
        return np.array(X), np.array(y)

    n_steps = 4
    X, y_train = crear_secuencias(df_scaled['y'].values, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y_train, epochs=100, verbose=0)

    entrada = df_scaled['y'].values[-n_steps:]
    pred_scaled = []
    for _ in range(periodo_semanas):
        x_input = entrada[-n_steps:].reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        pred_scaled.append(yhat[0][0])
        entrada = np.append(entrada, yhat[0][0])

    pred = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()
    fechas_futuras = pd.date_range(start=df_semanal['ds'].iloc[-1] + pd.Timedelta(days=7), periods=periodo_semanas, freq='W')
    df_pred = pd.DataFrame({'ds': fechas_futuras, 'yhat': pred})
    return df_pred

# -----------------------
# CARGA Y SELECCIÓN DE DATOS
# -----------------------
excel_path = "Items_Morante.xlsx"
df = cargar_datos(excel_path)

item_opciones = df[['ITEM', 'ITEM_DESC']].drop_duplicates().set_index('ITEM_DESC')
item_seleccionado_desc = st.selectbox("Selecciona un ítem:", item_opciones.index)
item_seleccionado = item_opciones.loc[item_seleccionado_desc]['ITEM']

df_item_raw = df[df['ITEM'] == item_seleccionado].copy()
descripcion = df_item_raw['DESCRIPCION'].iloc[0]
st.write(f"**Descripción:** {descripcion}")

periodo_dias = st.slider("Selecciona el número de días a predecir:", 7, 90, 45, step=7)
periodo_semanas = int(np.ceil(periodo_dias / 7))

# -----------------------
# PREDICCIÓN Y GRÁFICAS
# -----------------------
df_semanal = preparar_serie_semanal(df_item_raw)
forecast_prophet = entrenar_prophet(df_semanal, periodo_semanas)
forecast_lstm = entrenar_lstm(df_semanal, periodo_semanas)

fecha_corte = df_semanal['ds'].max()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_semanal['ds'], df_semanal['y'], 'b-', label='Real', linewidth=2)
ax.plot(forecast_prophet['ds'], forecast_prophet['yhat'], 'r--', label='Prophet', linewidth=2)
ax.plot(forecast_lstm['ds'], forecast_lstm['yhat'], 'g-.', label='LSTM', linewidth=2)
ax.axvline(fecha_corte, color='gray', linestyle=':', alpha=0.7)
ax.set_title("Predicción de Demanda Semanal", fontsize=16)
ax.set_xlabel("Fecha")
ax.set_ylabel("Cantidad Vendida")
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)

# -----------------------
# MÉTRICAS DE EVALUACIÓN
# -----------------------
df_eval = pd.merge(df_semanal, forecast_prophet, on='ds', how='inner').dropna()
if len(df_eval) > 0:
    y_true = df_eval['y']
    y_pred = df_eval['yhat']
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    st.subheader("Evaluación del modelo Prophet:")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")
else:
    st.warning("No hay suficientes datos para evaluar el modelo.")
