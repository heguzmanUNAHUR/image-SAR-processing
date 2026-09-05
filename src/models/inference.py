# -*- coding: utf-8 -*-
"""
Modulo de inferencia masiva y prediccion en batch sobre escenas satelitales completas.
Aplica el escalado StandardScaler y la inferencia de modelos entrenados .pkl sin fuga de informacion.
"""

import os
import ast
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def convertir_features_string_a_lista(features_input):
    """
    Convierte la lista de caracteristicas desde string/lista a formato de lista pura de Python.
    """
    if isinstance(features_input, list):
        return features_input

    if isinstance(features_input, str):
        try:
            res = ast.literal_eval(features_input)
            if isinstance(res, list):
                return res
        except Exception:
            pass

        clean_str = features_input.strip("[]")
        return [f.strip().strip("'\"") for f in clean_str.split(',') if f.strip()]

    return [str(features_input)]


def cargar_modelos_desde_carpeta(carpeta_modelos):
    """
    Carga todos los modelos .pkl presentes en una carpeta dada.
    """
    modelos_cargados = {}
    if not os.path.exists(carpeta_modelos):
        print(f"[ERROR] La carpeta de modelos no existe: {carpeta_modelos}")
        return modelos_cargados

    for archivo in os.listdir(carpeta_modelos):
        if archivo.endswith('_modelo.pkl'):
            nombre_modelo = archivo.replace('_modelo.pkl', '').replace('_', ' ')
            modelo_path = os.path.join(carpeta_modelos, archivo)
            try:
                with open(modelo_path, 'rb') as f:
                    modelo = pickle.load(f)
                modelos_cargados[nombre_modelo] = modelo
                print(f"[INFO] Modelo {nombre_modelo} cargado exitosamente")
            except Exception as e:
                print(f"[ERROR] Error al cargar modelo {archivo}: {e}")

    return modelos_cargados


def cargar_scaler_desde_carpeta(carpeta_modelos):
    """
    Carga el objeto StandardScaler guardado en la carpeta de modelos o carpeta padre.
    """
    scaler_path = os.path.join(carpeta_modelos, "scaler.pkl")

    if not os.path.exists(scaler_path):
        # Probar en carpeta padre (dataset_dividido o raiz de prueba)
        padre = os.path.dirname(carpeta_modelos)
        scaler_path_padre = os.path.join(padre, "dataset_dividido", "scaler.pkl")
        if os.path.exists(scaler_path_padre):
            scaler_path = scaler_path_padre

    if not os.path.exists(scaler_path):
        print(f"[WARNING] No se encontro el scaler.pkl en {carpeta_modelos}")
        return None

    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[INFO] Scaler cargado desde: {scaler_path}")
        return scaler
    except Exception as e:
        print(f"[ERROR] Error al cargar scaler: {e}")
        return None


def filtrar_y_preparar_dataset(dataset_path, features_list, target_column=None):
    """
    Carga el dataset CSV completo (complete_image.csv) y extrae las columnas seleccionadas.
    """
    print(f"[INFO] Cargando dataset de escena completa desde: {dataset_path}")
    features_list = convertir_features_string_a_lista(features_list)

    df = pd.read_csv(dataset_path)
    features_disponibles = df.columns.tolist()
    features_filtradas = [f for f in features_list if f in features_disponibles]

    if not features_filtradas:
        print("[ERROR] Ninguna caracteristica especificada fue encontrada en el CSV")
        return None, None, None

    X = df[features_filtradas].copy()

    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    y = df[target_column] if target_column and target_column in df.columns else None
    return X, y, df


def generar_predicciones_con_modelos(dataset_path, features_list, carpeta_modelos, carpeta_salida, target_column=None):
    """
    Genera predicciones en batch para cada modelo cargado desde la carpeta dada.
    """
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida, exist_ok=True)

    X, y, df_original = filtrar_y_preparar_dataset(dataset_path, features_list, target_column)
    if X is None:
        return None

    modelos = cargar_modelos_desde_carpeta(carpeta_modelos)
    if not modelos:
        return None

    scaler = cargar_scaler_desde_carpeta(carpeta_modelos)

    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    predicciones_por_modelo = {}

    for nombre_modelo, modelo in modelos.items():
        try:
            y_pred = modelo.predict(X_scaled)
            y_prob = None

            if hasattr(modelo, 'predict_proba'):
                try:
                    y_prob = modelo.predict_proba(X_scaled)
                except Exception:
                    y_prob = None

            df_res = df_original.copy()
            nombre_safe = nombre_modelo.replace(' ', '_')
            df_res[f'Prediccion_{nombre_safe}'] = y_pred

            if y_prob is not None:
                for i in range(y_prob.shape[1]):
                    df_res[f'Probabilidad_Clase_{i}_{nombre_safe}'] = y_prob[:, i]

            if y is not None:
                df_res['Target_Real'] = y

            archivo_out = os.path.join(carpeta_salida, f"predicciones_{nombre_safe}.csv")
            df_res.to_csv(archivo_out, index=False)
            print(f"[INFO] Predicciones guardadas en: {archivo_out}")

            predicciones_por_modelo[nombre_modelo] = df_res

        except Exception as e:
            print(f"[ERROR] Error generando predicciones con {nombre_modelo}: {e}")

    return predicciones_por_modelo
