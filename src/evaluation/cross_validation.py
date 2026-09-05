# -*- coding: utf-8 -*-
"""
Modulo de division espacial por poligonos e introduccion de StratifiedGroupKFold.
Garantiza el 100% de aislamiento espacial entre entrenamiento, validacion interna y holdout final.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


def organizar_poligonos_por_tamano(archivos_burned, archivos_not_burned):
    """
    Ordena los archivos CSV de poligonos de mayor a menor segun su recuento de muestras (pixeles).
    - Los 2 mas grandes de cada clase van a Train/Val (StratifiedGroupKFold).
    - El mas pequeno de cada clase va a Holdout Espacial Final.
    """
    counts_b = [(f, len(pd.read_csv(f))) for f in archivos_burned]
    counts_nb = [(f, len(pd.read_csv(f))) for f in archivos_not_burned]

    counts_b.sort(key=lambda x: x[1], reverse=True)
    counts_nb.sort(key=lambda x: x[1], reverse=True)

    poligonos_tv = {
        'Q1_Mayor': counts_b[0][0],
        'Q2_Medio': counts_b[1][0],
        'NQ1_Mayor': counts_nb[0][0],
        'NQ2_Medio': counts_nb[1][0],
    }

    poligonos_h = {
        'Q3_Menor': counts_b[2][0],
        'NQ3_Menor': counts_nb[2][0],
    }

    print("\n========================================================")
    print("   ASIGNACION AUTOMATICA DE POLIGONOS POR TAMANO")
    print("========================================================")
    print("POLIGONOS PARA TRAIN / VAL (STRATIFIED GROUPKFOLD):")
    print(f"  - Q1 (Mayor Burned):   {os.path.basename(counts_b[0][0])} ({counts_b[0][1]} px)")
    print(f"  - Q2 (Medio Burned):   {os.path.basename(counts_b[1][0])} ({counts_b[1][1]} px)")
    print(f"  - NQ1 (Mayor NotBurn): {os.path.basename(counts_nb[0][0])} ({counts_nb[0][1]} px)")
    print(f"  - NQ2 (Medio NotBurn): {os.path.basename(counts_nb[1][0])} ({counts_nb[1][1]} px)")

    print("\nPOLIGONOS PARA HOLDOUT ESPACIAL FINAL (PRUEBA):")
    print(f"  - Q3 (Menor Burned):   {os.path.basename(counts_b[2][0])} ({counts_b[2][1]} px)")
    print(f"  - NQ3 (Menor NotBurn): {os.path.basename(counts_nb[2][0])} ({counts_nb[2][1]} px)")
    print("========================================================\n")

    return poligonos_tv, poligonos_h


def cargar_y_etiquetar_poligonos(diccionario_poligonos):
    """
    Carga cada archivo CSV de poligono y le asigna una columna 'grupo_poligono'
    con su identificador regional para permitir la division con StratifiedGroupKFold.
    """
    dfs = []
    for group_id, filepath in diccionario_poligonos.items():
        print(f"[INFO] Cargando poligono {group_id} desde: {filepath}")
        df_poly = pd.read_csv(filepath)
        df_poly['grupo_poligono'] = group_id
        dfs.append(df_poly)
    df_resultado = pd.concat(dfs, ignore_index=True)
    return df_resultado


def preparar_features(df, indices_seleccionados, target_column="Burn_Classification"):
    """
    Extrae la matriz X y el vector y filtrando las caracteristicas seleccionadas.
    """
    y = df[target_column]
    X = df.drop([target_column, 'grupo_poligono'], axis=1, errors='ignore')

    if indices_seleccionados:
        valid_indices = [idx for idx in indices_seleccionados if idx in X.columns]
        X = X[valid_indices]

    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y


def dividir_y_escalar_dataset(df_train_val, df_holdout, indices_seleccionados, target_column="Burn_Classification", n_splits=2):
    """
    Realiza la particion espacial con StratifiedGroupKFold y aplica escalado StandardScaler
    calculado unicamente sobre el conjunto de entrenamiento para evitar fuga de informacion.
    """
    X_tv_raw, y_train_val = preparar_features(df_train_val, indices_seleccionados, target_column)
    X_h_raw, y_holdout = preparar_features(df_holdout, indices_seleccionados, target_column)
    feature_names = X_tv_raw.columns.tolist()

    grupos_tv = df_train_val['grupo_poligono']

    print("\n=== DIVISION ESPACIAL CON STRATIFIED GROUPKFOLD ===")
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    train_idx, test_idx = next(sgkf.split(X_tv_raw, y_train_val, groups=grupos_tv))

    X_train_raw = X_tv_raw.iloc[train_idx]
    y_train = y_train_val.iloc[train_idx]
    X_test_raw = X_tv_raw.iloc[test_idx]
    y_test = y_train_val.iloc[test_idx]

    print(f"[INFO] Poligono(s) asignados a Train: {grupos_tv.iloc[train_idx].unique()} -> {X_train_raw.shape[0]} muestras")
    print(f"[INFO] Poligono(s) asignados a Test interno: {grupos_tv.iloc[test_idx].unique()} -> {X_test_raw.shape[0]} muestras")
    print(f"[INFO] Poligono(s) asignados a Holdout Final: {df_holdout['grupo_poligono'].unique()} -> {X_h_raw.shape[0]} muestras")

    # Escalado sin fuga espacial
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_holdout = scaler.transform(X_h_raw)

    return X_train, X_test, X_holdout, y_train, y_test, y_holdout, scaler, feature_names


def guardar_dataset_dividido(X_train, X_test, X_holdout, y_train, y_test, y_holdout, scaler, carpeta_destino):
    """
    Guarda los arreglos de datos divididos y el scaler en formato PKL.
    """
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino, exist_ok=True)

    with open(os.path.join(carpeta_destino, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(carpeta_destino, "datos_entrenamiento.pkl"), 'wb') as f:
        pickle.dump({'X_train': X_train, 'y_train': y_train}, f)

    with open(os.path.join(carpeta_destino, "datos_test.pkl"), 'wb') as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)

    with open(os.path.join(carpeta_destino, "datos_holdout.pkl"), 'wb') as f:
        pickle.dump({'X_holdout': X_holdout, 'y_holdout': y_holdout}, f)

    print(f"[INFO] Dataset dividido guardado en: {carpeta_destino}")
