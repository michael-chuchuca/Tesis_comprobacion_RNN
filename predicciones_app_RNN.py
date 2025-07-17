import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

# -----------------------
# CONFIGURACIÓN
# -----------------------
st.set_page_config(page_title="Predicción LSTM", layout="wide")
st.markdown("<h1 style='text-align: center;'>Predicción de Inventarios con LSTM</h1>", unsafe_allow_html=True)

# -----------------------
# FUNCIONES
# -----------------------
@st.cache_data
def cargar_datos(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df['FECHA_VENTA'] = pd.to_datetime(df['FECHA_VENTA'])
    df['ITEM_DESC'] = df['ITEM'].astype(str) + " - " + df['DESCRIPCION'].astype(str)
    df = df.sort_values('FECHA_VENTA')
    return df

def preparar_semanal(df_raw):
    df = df_raw.groupby(pd.Grouper(key='FECHA_VENTA', freq='W')).agg({
        'CANTIDAD_VENDIDA': 'sum'
    }).reset_index()
    df = df.rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})
    return df

def crear_secuencias(serie, n_steps):
    X, y = [], []
    for i in range(len(serie) - n_steps):
        X.append(serie[i:i+n_steps])
        y.append(serie[i+n_steps])
    return np.array(X), np.array(y)

def entrenar_y_predecir_lstm(df, periodo=6):
    df = df.copy()
    scaler = MinMaxScaler()
    df['y_scaled'] = scaler.fit_transform(df[['y']])

    n_steps = 4
    X, y = crear_secuencias(df['y_scaled'].values, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    entrada = df['y_scaled'].values[-n_steps:]
    pred_scaled = []

    for _ in range(periodo):
        x_input = entrada[-n_steps:].reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        pred_scaled.append(yhat[0][0])
        entrada = np.append(entrada, yhat[0][0])

    pred = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()
    fechas = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(weeks=1), periods=periodo, freq='W')
    return pd.DataFrame({'ds': fechas, 'yhat': pred})

# -----------------------
# APP
# -----------------------
excel_path = "Items_Morante.xlsx"
df = cargar_datos(excel_path)

item_opciones = df[['ITEM', 'ITEM_DESC']].drop_duplicates().set_index('ITEM_DESC')
item_seleccionado_desc = st.selectbox("Selecciona un ítem:", item_opciones.index)
item_codigo = item_opciones.loc[item_seleccionado_desc]['ITEM']

df_item = df[df['ITEM'] == item_codigo].copy()
descripcion = df_item['DESCRIPCION'].iloc[0]
st.write(f"**Descripción del ítem:** {descripcion}")

dias_prediccion = st.slider("Número de días a predecir", 7, 90, 45, step=7)
semanas_prediccion = int(np.ceil(dias_prediccion / 7))

df_semanal = preparar_semanal(df_item)
pred_df = entrenar_y_predecir_lstm(df_semanal, semanas_prediccion)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_semanal['ds'], df_semanal['y'], label="Real", linewidth=2)
ax.plot(pred_df['ds'], pred_df['yhat'], label="LSTM", color="green", linestyle="--", linewidth=2)
ax.set_title("Demanda semanal real vs predicción LSTM")
ax.set_xlabel("Fecha")
ax.set_ylabel("Cantidad Vendida")
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)

# -----------------------
# MÉTRICAS
# -----------------------
df_eval = df_semanal.copy().tail(len(pred_df))
if len(df_eval) == len(pred_df):
    y_true = df_eval['y'].values
    y_pred = pred_df['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    st.subheader("Evaluación del modelo LSTM:")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")
else:
    st.warning("No hay datos reales suficientes para calcular las métricas.")
