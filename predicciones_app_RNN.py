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
    
    # Eliminar outliers extremos
    umbral_extremo = df_agg['y'].quantile(0.98)
    df_agg['y'] = np.where(df_agg['y'] > umbral_extremo, umbral_extremo, df_agg['y'])

    # Suavizado doble
    df_agg['y'] = df_agg['y'].rolling(window=3, min_periods=1).mean()
    df_agg['y'] = df_agg['y'].clip(upper=df_agg['y'].quantile(0.95))
    df_agg['y'] = df_agg['y'].rolling(window=2, min_periods=1).mean()

    # Restaurar columnas perdidas
    df_agg['DESCRIPCION'] = df_item_raw['DESCRIPCION'].iloc[0]
    df_agg['ITEM'] = df_item_raw['ITEM'].iloc[0]

    return df_agg
