# -*- coding: utf-8 -*-
"""
Script Orquestador - Step 4: Entrenamiento y Evaluación Multi-Incendio Flexible (N -> M)
---------------------------------------------------------------------------------------
Permite la selección interactiva o parametrizada por CLI de N incendios para entrenamiento
y M incendios para evaluación (Cross-Fire / Zero-Shot y Modelos Globales).

Aislamiento Espacial:
- El polígono de menor tamaño por clase de cada incendio se reserva para Prueba Espacial.
- Todos los demás polígonos (mayores) se asignan a Entrenamiento/Validación.
- StandardScaler se ajusta (fit) EXCLUSIVAMENTE con el conjunto de entrenamiento.

Uso Interactivo:
    python scripts/04_multi_fire_training.py

Uso CLI / Headless:
    python scripts/04_multi_fire_training.py --train-incendios PONTON SAN_JUAN --test-incendios LOS_ALERCES --prueba "ZeroShot_LosAlerces" --headless
"""

import matplotlib
matplotlib.use('Agg')
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Agregar raíz al path de módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.multi_fire_selector import (
    escanear_incendios_disponibles,
    validar_y_organizar_poligonos,
    crear_interfaz_seleccion_incendios_multiples
)
from src.models.feature_selection import crear_interfaz_seleccion_indices
from src.models.model_factory import (
    crear_interfaz_seleccion_modelos,
    entrenar_modelo,
    guardar_modelo_y_artefactos
)
from src.evaluation.metrics import calcular_metricas_completas


def cargar_y_etiquetar_poligonos(diccionario_poligonos):
    """
    Carga cada CSV de polígono y le asigna una columna 'grupo_poligono' para GroupKFold.
    """
    dfs = []
    for group_id, filepath in diccionario_poligonos.items():
        df_poly = pd.read_csv(filepath)
        df_poly['grupo_poligono'] = group_id
        dfs.append(df_poly)
    return pd.concat(dfs, ignore_index=True)


def preparar_features(df, indices_seleccionados, target_column="Burn_Classification"):
    """
    Filtra características y codifica categóricas si existieran.
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


def main():
    parser = argparse.ArgumentParser(description="Step 4: Entrenamiento y Evaluación Multi-Incendio Flexible (N -> M)")
    parser.add_argument("--train-incendios", nargs="+", help="Lista de identificadores de incendios para entrenamiento (ej: PONTON SAN_JUAN)")
    parser.add_argument("--test-incendios", nargs="+", help="Lista de identificadores de incendios para evaluación (ej: LOS_ALERCES)")
    parser.add_argument("--prueba", type=str, default=None, help="Nombre de la prueba para guardar los resultados")
    parser.add_argument("--base-dir", type=str, default="data/raw", help="Directorio base conteniendo los incendios")
    parser.add_argument("--headless", action="store_true", help="Ejecutar en modo no interactivo sin GUIs")
    
    args = parser.parse_args()

    # 1. Escanear incendios disponibles
    incendios_disponibles = escanear_incendios_disponibles(args.base_dir)
    if not incendios_disponibles:
        print("Error: No se encontraron datasets de incendios válidos en 'data/raw/'.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("   ENTRENAMIENTO Y EVALUACIÓN MULTI-INCENDIO FLEXIBLE (STEP 4)")
    print("=" * 70)
    print(f"  Incendios detectados en el sistema ({len(incendios_disponibles)}): {list(incendios_disponibles.keys())}")

    nombre_prueba_gui = None
    if args.train_incendios and args.test_incendios:
        train_selected = [i.upper() for i in args.train_incendios if i.upper() in incendios_disponibles]
        test_selected = [i.upper() for i in args.test_incendios if i.upper() in incendios_disponibles]
    elif args.headless:
        train_selected = list(incendios_disponibles.keys())
        test_selected = list(incendios_disponibles.keys())
    else:
        train_selected, test_selected, nombre_prueba_gui = crear_interfaz_seleccion_incendios_multiples(incendios_disponibles)

    if not train_selected or not test_selected:
        print("Error: Se requiere seleccionar al menos 1 incendio para entrenamiento y 1 para evaluación.")
        sys.exit(1)

    nombre_prueba = args.prueba or nombre_prueba_gui
    if not nombre_prueba:
        str_tr = "_".join(train_selected)
        str_te = "_".join(test_selected)
        nombre_prueba = f"Prueba_MultiIncendio_Train_{str_tr}_Test_{str_te}"

    print(f"\n  Nombre del Experimento: {nombre_prueba}")
    print(f"  Incendios de Entrenamiento ({len(train_selected)}): {train_selected}")
    print(f"  Incendios de Evaluación    ({len(test_selected)}):  {test_selected}\n")

    # 3. Validación y Organización de Polígonos por Incendio
    poligonos_tv_all = {}
    poligonos_h_all = {}
    
    incendios_involucrados = list(dict.fromkeys(train_selected + test_selected))
    print("=== ASIGNACION DE POLIGONOS POR TAMANO (1 menor a Test, resto a Train/Val) ===")
    for inc_key in incendios_involucrados:
        inc_info = incendios_disponibles[inc_key]
        pol_tv, pol_h, res = validar_y_organizar_poligonos(
            inc_info['archivos_burned'], 
            inc_info['archivos_not_burned'], 
            inc_key
        )
        poligonos_tv_all.update(pol_tv)
        poligonos_h_all.update(pol_h)
        print(f"  - {inc_info['nombre_lindo']}: {res['n_poligonos_tv']} poligonos a Train/Val ({res['total_px_tv']} px) | {res['n_poligonos_h']} poligonos a Test ({res['total_px_h']} px)")

    # 4. Selección de Índices y Modelos
    dataset_ref = incendios_disponibles[train_selected[0]]['dataset_ref']
    if not dataset_ref:
        for k in train_selected:
            if incendios_disponibles[k]['dataset_ref']:
                dataset_ref = incendios_disponibles[k]['dataset_ref']
                break

    if args.headless:
        indices_seleccionados = []
        modelos_seleccionados = [
            "SVM_Linear", "SVM_RBF", "Random Forest", "Extra Trees", 
            "Gradient Boosting Balanced", "Logistics Regression", "NN-ReLU-2L(200,100)-r2"
        ]
    else:
        indices_seleccionados = crear_interfaz_seleccion_indices(dataset_ref) if dataset_ref else []
        modelos_seleccionados = crear_interfaz_seleccion_modelos()

    target_column = "Burn_Classification"

    # 5. Partición Espacial (StratifiedGroupKFold) en los Incendios de Entrenamiento
    all_X_train = []
    all_y_train = []
    all_X_test = []
    all_y_test = []
    feature_names = None

    print("\n=== PARTICIÓN ESPACIAL CON STRATIFIED GROUPKFOLD (INCENDIOS DE ENTRENAMIENTO) ===")

    for inc_key in train_selected:
        inc_info = incendios_disponibles[inc_key]
        pol_tv_sitio = {k: v for k, v in poligonos_tv_all.items() if k.startswith(inc_key)}
        
        df_tv = cargar_y_etiquetar_poligonos(pol_tv_sitio)
        X_tv, y_tv = preparar_features(df_tv, indices_seleccionados, target_column)
        
        if feature_names is None:
            feature_names = X_tv.columns.tolist()

        grupos = df_tv['grupo_poligono']
        sgkf = StratifiedGroupKFold(n_splits=2)
        train_idx, test_idx = next(sgkf.split(X_tv, y_tv, groups=grupos))

        all_X_train.append(X_tv.iloc[train_idx])
        all_y_train.append(y_tv.iloc[train_idx])
        all_X_test.append(X_tv.iloc[test_idx])
        all_y_test.append(y_tv.iloc[test_idx])

        print(f"  - {inc_info['nombre_lindo']}: Train = {len(train_idx)} px | Test Interno = {len(test_idx)} px")

    X_train_raw = pd.concat(all_X_train, ignore_index=True)
    y_train = pd.concat(all_y_train, ignore_index=True)
    X_test_raw = pd.concat(all_X_test, ignore_index=True)
    y_test = pd.concat(all_y_test, ignore_index=True)

    # 6. Preparación de Datasets de Evaluación (Holdouts)
    holdouts_dict = {}
    all_X_h_list = []
    all_y_h_list = []

    print("\n=== PREPARACIÓN DE CONJUNTOS DE PRUEBA ESPACIAL DE EVALUACIÓN ===")
    for inc_key in test_selected:
        inc_info = incendios_disponibles[inc_key]
        pol_h_sitio = {k: v for k, v in poligonos_h_all.items() if k.startswith(inc_key)}
        
        df_h = cargar_y_etiquetar_poligonos(pol_h_sitio)
        X_h, y_h = preparar_features(df_h, indices_seleccionados, target_column)

        es_zero_shot = (inc_key not in train_selected)
        tipo_tag = "[ZERO-SHOT (Bioma Inedito No Visto)]" if es_zero_shot else "[Bioma Conocido en Entrenamiento]"

        holdouts_dict[inc_key] = {
            'X_raw': X_h,
            'y': y_h,
            'nombre_lindo': inc_info['nombre_lindo'],
            'es_zero_shot': es_zero_shot,
            'n_muestras': len(y_h)
        }
        all_X_h_list.append(X_h)
        all_y_h_list.append(y_h)
        print(f"  - {inc_info['nombre_lindo']} [{tipo_tag}]: {len(y_h)} px")

    X_h_combined_raw = pd.concat(all_X_h_list, ignore_index=True)
    y_holdout_combined = pd.concat(all_y_h_list, ignore_index=True)

    # 7. Escalado Estricto con StandardScaler (fit SOLO en Train)
    print("\n=== ESCALADO CON STANDARDSCALER (fit EXCLUSIVAMENTE en Train) ===")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_holdout_combined = scaler.transform(X_h_combined_raw)

    for inc_key, h_info in holdouts_dict.items():
        h_info['X_scaled'] = scaler.transform(h_info['X_raw'])

    # 8. Persistencia de Datos y Modelo
    output_dir = os.path.join("results", "models", nombre_prueba)
    dataset_dir = os.path.join(output_dir, "dataset_dividido")
    os.makedirs(dataset_dir, exist_ok=True)

    with open(os.path.join(dataset_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(dataset_dir, "datos_entrenamiento.pkl"), "wb") as f:
        pickle.dump({'X_train': X_train, 'y_train': y_train}, f)
    with open(os.path.join(dataset_dir, "datos_test.pkl"), "wb") as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)
    with open(os.path.join(dataset_dir, "datos_holdout.pkl"), "wb") as f:
        pickle.dump({'X_holdout': X_holdout_combined, 'y_holdout': y_holdout_combined}, f)

    # 9. Entrenamiento de Modelos Seleccionados
    print("\n=== ENTRENAMIENTO DE CLASIFICADORES SELECCIONADOS ===")
    for nombre_modelo in modelos_seleccionados:
        subcarpeta_modelo = os.path.join(output_dir, nombre_modelo)
        res_fit = entrenar_modelo(X_train, X_test, y_train, y_test, nombre_modelo)
        if res_fit is not None:
            guardar_modelo_y_artefactos(res_fit, scaler, subcarpeta_modelo, feature_names=feature_names)

    # 10. Evaluación Global Combinada
    print("\n=== EVALUACIÓN EN CONJUNTO DE PRUEBA COMBINADO GLOBAL ===")
    metricas_globales = []
    
    for nombre_modelo in modelos_seleccionados:
        nombre_safe = nombre_modelo.replace(' ', '_').replace('/', '_').replace('\\', '_')
        modelo_path = os.path.join(output_dir, nombre_modelo, f"{nombre_safe}_modelo.pkl")
        if os.path.exists(modelo_path):
            with open(modelo_path, "rb") as f:
                mod = pickle.load(f)
            
            y_pred = mod.predict(X_holdout_combined)
            y_prob = mod.predict_proba(X_holdout_combined)[:, 1] if hasattr(mod, "predict_proba") else None
            
            m = calcular_metricas_completas(y_holdout_combined, y_pred, y_prob)
            m["Modelo"] = nombre_modelo
            metricas_globales.append(m)

    df_global = pd.DataFrame(metricas_globales).sort_values("Accuracy", ascending=False)
    csv_global = os.path.join(output_dir, "metricas_holdout_todos_los_modelos.csv")
    df_global.to_csv(csv_global, index=False)
    
    print(f"\n  {'Modelo':<32} {'Accuracy':>9} {'F1-Score':>9} {'Precision':>9} {'Recall':>8} {'AUC':>7}")
    print(f"  {'-'*32} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*7}")
    for _, r in df_global.iterrows():
        auc_str = f"{r['AUC']:.4f}" if r['AUC'] is not None else "  N/A "
        print(f"  {r['Modelo']:<32} {r['Accuracy']:>9.4f} {r['F1_Score']:>9.4f} {r['Precision']:>9.4f} {r['Recall']:>8.4f} {auc_str:>7}")

    # 11. Evaluación Desglosada por Incendio
    print("\n" + "=" * 70)
    print("   EVALUACIÓN DESGLOSADA POR INCENDIO DE PRUEBA")
    print("=" * 70)

    for inc_key, h_info in holdouts_dict.items():
        tag = "[ZERO-SHOT]" if h_info['es_zero_shot'] else "[BIOMA CONOCIDO]"
        print(f"\n  - INCENDIO: {h_info['nombre_lindo']} {tag} - {h_info['n_muestras']} px")
        
        metricas_sitio = []
        for nombre_modelo in modelos_seleccionados:
            nombre_safe = nombre_modelo.replace(' ', '_').replace('/', '_').replace('\\', '_')
            modelo_path = os.path.join(output_dir, nombre_modelo, f"{nombre_safe}_modelo.pkl")
            if os.path.exists(modelo_path):
                with open(modelo_path, "rb") as f:
                    mod = pickle.load(f)
                
                y_pred = mod.predict(h_info['X_scaled'])
                y_prob = mod.predict_proba(h_info['X_scaled'])[:, 1] if hasattr(mod, "predict_proba") else None
                
                m = calcular_metricas_completas(h_info['y'], y_pred, y_prob)
                m["Modelo"] = nombre_modelo
                m["Incendio"] = h_info['nombre_lindo']
                m["Tipo"] = "Zero-Shot Inedito" if h_info['es_zero_shot'] else "Bioma Conocido"
                metricas_sitio.append(m)

        df_sitio = pd.DataFrame(metricas_sitio).sort_values("Accuracy", ascending=False)
        csv_sitio = os.path.join(output_dir, f"metricas_holdout_{inc_key}.csv")
        df_sitio.to_csv(csv_sitio, index=False)

        for _, r in df_sitio.iterrows():
            auc_str = f"{r['AUC']:.4f}" if r['AUC'] is not None else "  N/A "
            print(f"     {r['Modelo']:<30} Acc: {r['Accuracy']:.4f} | F1: {r['F1_Score']:.4f} | AUC: {auc_str}")

    print("\n" + "=" * 70)
    print("   STEP 4 COMPLETADO EXITOSAMENTE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
