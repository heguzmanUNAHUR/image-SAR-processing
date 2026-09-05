# -*- coding: utf-8 -*-
"""
Modulo de calculo y guardado de metricas de evaluacion cuantitativa.
Calcula Precision, Recall, Accuracy, F1-Score, AUC y Specificity.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def calcular_metricas_completas(y_test, y_pred, y_prob=None):
    """
    Calcula todas las metricas cuantitativas de clasificacion.
    
    Args:
        y_test: Etiquetas reales
        y_pred: Predicciones del modelo
        y_prob: Probabilidades predichas (opcional, para AUC)
        
    Returns:
        dict: Diccionario con Precision, Recall, Accuracy, F1_Score, AUC, Specificity
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    if len(np.unique(y_test)) == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        auc = None
        if y_prob is not None:
            try:
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc = None
    else:
        specificities = []
        for i in range(len(cm)):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificities.append(spec)
        specificity = float(np.mean(specificities)) if specificities else 0.0
        auc = None

    return {
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'F1_Score': f1,
        'AUC': auc,
        'Specificity': specificity
    }


def guardar_metricas_modelo(resultado, carpeta_destino):
    """
    Calcula y almacena las metricas de un modelo entrenado en un CSV acumulativo.
    """
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino, exist_ok=True)

    metricas_path = os.path.join(carpeta_destino, "metricas_prueba.csv")
    df_existente = pd.DataFrame()

    if os.path.exists(metricas_path):
        try:
            df_existente = pd.read_csv(metricas_path)
        except Exception as e:
            print(f"[WARNING] Error al leer archivo existente de metricas: {e}")

    nombre_modelo = resultado.get('nombre', 'Modelo')
    y_test = resultado['y_test']
    y_pred = resultado['y_pred']
    y_prob = resultado.get('y_prob', None)
    time_train = resultado.get('time_train', 0.0)

    metricas = calcular_metricas_completas(y_test, y_pred, y_prob)
    metricas['Modelo'] = nombre_modelo
    metricas['Tiempo_Entrenamiento_Segundos'] = time_train

    df_nueva = pd.DataFrame([metricas])
    columnas_ordenadas = ['Modelo', 'Precision', 'Recall', 'Accuracy', 'F1_Score', 'AUC', 'Specificity', 'Tiempo_Entrenamiento_Segundos']
    df_nueva = df_nueva[columnas_ordenadas]

    if not df_existente.empty and list(df_existente.columns) == columnas_ordenadas:
        df_final = pd.concat([df_existente, df_nueva], ignore_index=True)
    else:
        df_final = df_nueva

    df_final.to_csv(metricas_path, index=False)
    print(f"[INFO] Metricas de {nombre_modelo} guardadas en: {metricas_path}")
    return df_final


def cargar_modelos_entrenados(carpeta_modelos):
    """
    Carga los modelos entrenados (.pkl) desde una carpeta dada.
    """
    modelos_cargados = {}
    if not os.path.exists(carpeta_modelos):
        return modelos_cargados

    for archivo in os.listdir(carpeta_modelos):
        if archivo.endswith('_modelo.pkl'):
            nombre_modelo = archivo.replace('_modelo.pkl', '').replace('_', ' ')
            modelo_path = os.path.join(carpeta_modelos, archivo)
            try:
                with open(modelo_path, 'rb') as f:
                    modelo = pickle.load(f)
                modelos_cargados[nombre_modelo] = modelo
            except Exception as e:
                print(f"[ERROR] Error al cargar modelo {archivo}: {e}")

    return modelos_cargados


def evaluar_todos_los_modelos_de_prueba(nombre_prueba, X_holdout, y_holdout, features, carpeta_base='results/models/'):
    """
    Evalua todos los modelos entrenados en una prueba contra el dataset Holdout espacial aislado.
    Genera y guarda 'metricas_holdout_todos_los_modelos.csv'.
    """
    print(f"\n=== EVALUACION FINAL EN HOLDOUT ESPACIAL (PRUEBA: {nombre_prueba}) ===")
    carpeta_prueba = os.path.join(carpeta_base, nombre_prueba)

    if not os.path.exists(carpeta_prueba):
        print(f"[ERROR] La carpeta de prueba no existe: {carpeta_prueba}")
        return None

    subcarpetas = [
        item for item in os.listdir(carpeta_prueba)
        if os.path.isdir(os.path.join(carpeta_prueba, item)) and item != 'dataset_dividido'
    ]

    todas_las_metricas = []

    for subcarpeta in subcarpetas:
        carpeta_modelo_path = os.path.join(carpeta_prueba, subcarpeta)
        modelos = cargar_modelos_entrenados(carpeta_modelo_path)

        for nombre_modelo, modelo in modelos.items():
            try:
                y_pred = modelo.predict(X_holdout)
                y_prob = None
                if hasattr(modelo, 'predict_proba') and len(np.unique(y_holdout)) == 2:
                    try:
                        y_prob = modelo.predict_proba(X_holdout)[:, 1]
                    except Exception:
                        y_prob = None

                metricas = calcular_metricas_completas(y_holdout, y_pred, y_prob)
                metricas['Modelo'] = nombre_modelo
                metricas['Carpeta'] = subcarpeta
                metricas['Dataset'] = 'Holdout'
                metricas['Features'] = str(features)

                todas_las_metricas.append(metricas)

            except Exception as e:
                print(f"[ERROR] Error evaluando {nombre_modelo}: {e}")

    if todas_las_metricas:
        df_todas = pd.DataFrame(todas_las_metricas)
        columnas_ordenadas = ['Carpeta', 'Modelo', 'Dataset', 'Features', 'Precision', 'Recall', 'Accuracy', 'F1_Score', 'AUC', 'Specificity']
        df_todas = df_todas[columnas_ordenadas]

        resultados_path = os.path.join(carpeta_prueba, "metricas_holdout_todos_los_modelos.csv")
        df_todas.to_csv(resultados_path, index=False)

        print(f"[INFO] Resultados de Holdout guardados en: {resultados_path}")
        print("\n=== RESUMEN DE EVALUACION EN HOLDOUT ESPACIAL ===")
        print(df_todas.to_string(index=False, float_format='%.4f'))

        mejor = df_todas.loc[df_todas['Accuracy'].idxmax()]
        print(f"\n[INFO] Mejor modelo en holdout: {mejor['Modelo']} (Accuracy: {mejor['Accuracy']:.4f})")
        return df_todas
    else:
        print("[WARNING] No se evaluaron modelos en holdout")
        return None
