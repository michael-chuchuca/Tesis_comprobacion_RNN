import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------
# Funciones
# -----------------------

@st.cache_data
def cargar_datos(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    df['FECHA_VENTA'] = pd.to_datetime(df['FECHA_VENTA'])
    df = df.sort_values(by='FECHA_VENTA')
    return df

def preparar_serie_semanal(df_item_raw):
    df_agg = df_item_raw.groupby(pd.Grouper(key='FECHA_VENTA', freq='W')).agg({
        'CANTIDAD_VENDIDA': 'sum',
        'DESCRIPCION': 'first',
        'ITEM': 'first'
    }).reset_index()

    df_agg = df_agg.rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})

    # Rellenar semanas faltantes con 0
    fecha_inicio = df_agg['ds'].min()
    fecha_fin = df_agg['ds'].max()
    todas_fechas = pd.date_range(fecha_inicio, fecha_fin, freq='W')
    df_agg = df_agg.set_index('ds').reindex(todas_fechas).fillna({'y': 0}).reset_index()
    df_agg = df_agg.rename(columns={'index': 'ds'})

    # Eliminación de outliers extremos
    umbral_extremo = df_agg['y'].quantile(0.98)
    df_agg['y'] = np.where(df_agg['y'] > umbral_extremo, umbral_extremo, df_agg['y'])

    # Suavizado doble
    df_agg['y'] = df_agg['y'].rolling(window=3, min_periods=1).mean()
    df_agg['y'] = df_agg['y'].clip(upper=df_agg['y'].quantile(0.95))
    df_agg['y'] = df_agg['y'].rolling(window=2, min_periods=1).mean()

    return df_agg

def crear_secuencias(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def entrenar_lstm(df_semanal, periodo_semanas):
    serie = df_semanal['y'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    serie_scaled = scaler.fit_transform(serie)

    window_size = 5
    X, y = crear_secuencias(serie_scaled, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    input_seq = serie_scaled[-window_size:]
    predictions = []

    for _ in range(periodo_semanas):
        input_seq_reshaped = input_seq.reshape((1, window_size, 1))
        pred = model.predict(input_seq_reshaped, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[1:], [[pred]], axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    fechas_futuras = pd.date_range(start=df_semanal['ds'].max() + pd.Timedelta(weeks=1), periods=periodo_semanas, freq='W')
    
    forecast = pd.DataFrame({'ds': fechas_futuras, 'yhat': predictions})
    return forecast

# -----------------------
# Interfaz Streamlit
# -----------------------

st.markdown("<h1 style='text-align: center;'>Predicción de Demanda Semanal con LSTM</h1>", unsafe_allow_html=True)

excel_path = "Items_Morante.xlsx"
df = cargar_datos(excel_path)

df['ITEM_DESC'] = df['ITEM'].astype(str) + " - " + df['DESCRIPCION'].astype(str)
item_opciones = df[['ITEM', 'ITEM_DESC']].drop_duplicates().set_index('ITEM_DESC')
item_seleccionado_desc = st.selectbox("Selecciona un ítem:", item_opciones.index)
item_seleccionado = item_opciones.loc[item_seleccionado_desc]['ITEM']

df_item_raw = df[df['ITEM'] == item_seleccionado].copy()

periodo_dias = st.slider("Selecciona el número de días a predecir:", min_value=7, max_value=90, value=45)
periodo_semanas = int(np.ceil(periodo_dias / 7))

df_semanal = preparar_serie_semanal(df_item_raw)
forecast = entrenar_lstm(df_semanal, periodo_semanas)

df_real = df_semanal.copy()
df_comparacion = pd.merge(df_real, forecast, on='ds', how='left')
fecha_corte = df_real['ds'].max()

fig, ax = plt.subplots(figsize=(14, 6))

# Línea de la serie original
serie_original = df_item_raw.groupby(pd.Grouper(key='FECHA_VENTA', freq='W'))['CANTIDAD_VENDIDA'].sum().reset_index()
ax.plot(serie_original['FECHA_VENTA'], serie_original['CANTIDAD_VENDIDA'], 'c--', label='Serie Original', linewidth=1.2)

# Línea suavizada y pronóstico
ax.plot(df_comparacion['ds'], df_comparacion['y'], 'b--', label='Cantidad Real Suavizada', linewidth=2)
ax.plot(forecast['ds'], forecast['yhat'], 'r--', label='Cantidad Pronosticada', linewidth=2)

ax.axvline(fecha_corte, color='gray', linestyle=':', alpha=0.7)
ax.annotate('Inicio de Predicción', xy=(fecha_corte, ax.get_ylim()[1]*0.9),
            xytext=(10, 0), textcoords='offset points', fontsize=10, color='gray')
ax.set_title("Pronóstico Semanal de Ventas con LSTM Optimizado", fontsize=15)
ax.set_xlabel("Fecha", fontsize=12)
ax.set_ylabel("Cantidad Vendida (semanal)", fontsize=12)
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)


forecast_futuro = forecast[forecast['ds'] > fecha_corte].copy()
total_predicho = forecast_futuro['yhat'].sum() if not forecast_futuro.empty else 0
total_diario_estimado = total_predicho * (periodo_dias / (periodo_semanas * 7))

st.subheader(f"Total estimado para los próximos {periodo_dias} días:")
st.write(f"**{total_diario_estimado:.0f} unidades estimadas** para importar en {periodo_dias} días.")
