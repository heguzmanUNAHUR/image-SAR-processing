# -*- coding: utf-8 -*-
"""
Script orquestador ejecutable para el entrenamiento de modelos ML y evaluacion en holdout espacial.
Preserva el 100% de la logica academica de la tesis (StratifiedGroupKFold, escalado sin fuga, asignacion automatica de poligonos).
"""

import os
import sys
import argparse
import pandas as pd

# Matplotlib backend sin GUI para ejecutables
import matplotlib
matplotlib.use('Agg')

# Incorporar directorio raiz al path para importaciones puras
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.feature_selection import crear_interfaz_seleccion_indices
from src.models.model_factory import (
    crear_interfaz_seleccion_modelos,
    entrenar_modelo,
    guardar_modelo_y_artefactos,
    obtener_modelos_disponibles
)
from src.evaluation.cross_validation import (
    organizar_poligonos_por_tamano,
    cargar_y_etiquetar_poligonos,
    dividir_y_escalar_dataset,
    guardar_dataset_dividido
)
from src.evaluation.metrics import (
    guardar_metricas_modelo,
    evaluar_todos_los_modelos_de_prueba
)


def construir_argumentos():
    parser = argparse.ArgumentParser(
        description="Entrenamiento y evaluacion espacial de modelos ML para cicatrices de incendios SAR."
    )
    parser.add_argument(
        "--incendio",
        type=str,
        default="PONTON",
        choices=["PONTON", "LOS_ALERCES", "SAN_JUAN"],
        help="Nombre del incendio a evaluar ('PONTON', 'LOS_ALERCES' o 'SAN_JUAN')."
    )
    parser.add_argument(
        "--proyecto",
        type=str,
        default=None,
        help="Ruta al directorio del proyecto que contiene los CSVs (sobrescribe --incendio)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Ejecutar en modo no interactivo (sin GUI Tkinter)."
    )
    parser.add_argument(
        "--prueba",
        type=str,
        default=None,
        help="Nombre identificador de la prueba para trazabilidad y carpeta de resultados."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "models"),
        help="Carpeta base donde se guardaran los modelos y metricas."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=111,
        help="Semilla para reproducibilidad (por defecto 111)."
    )
    return parser.parse_args()


def obtener_configuracion_incendio(incendio, proyecto_cli, prueba_cli, root_dir):
    """
    Resuelve las rutas de datasets y vectores para el incendio seleccionado.
    """
    if prueba_cli:
        nombre_prueba = prueba_cli
    elif proyecto_cli:
        nombre_prueba = f"Prueba_{os.path.basename(proyecto_cli)}_GKF_Spatial_Holdout"
    elif incendio == "PONTON":
        nombre_prueba = "Prueba Ponton GKF Spatial Holdout (Tesis)"
    elif incendio == "LOS_ALERCES":
        nombre_prueba = "Prueba Los Alerces GKF Spatial Holdout (Tesis)"
    elif incendio == "SAN_JUAN":
        nombre_prueba = "Prueba San Juan (Misiones) GKF Spatial Holdout (Tesis)"
    else:
        raise ValueError(f"Incendio '{incendio}' no reconocido.")

    if proyecto_cli:
        carpeta_csvs = os.path.join(proyecto_cli, "csvs")
        if not os.path.exists(carpeta_csvs):
            carpeta_csvs = proyecto_cli
    elif incendio == "PONTON":
        carpeta_csvs = os.path.join(root_dir, "data", "raw", "Ponton", "csvs")
    elif incendio == "LOS_ALERCES":
        carpeta_csvs = os.path.join(root_dir, "data", "raw", "Los_Alerces", "csvs")
    elif incendio == "SAN_JUAN":
        carpeta_csvs = os.path.join(root_dir, "data", "raw", "San_Juan", "csvs")
    else:
        raise ValueError(f"Incendio '{incendio}' no reconocido.")

    if not os.path.exists(carpeta_csvs):
        raise FileNotFoundError(f"No se encontro la carpeta de CSVs en: {carpeta_csvs}")

    archivos_in_folder = [os.path.join(carpeta_csvs, f) for f in os.listdir(carpeta_csvs) if f.endswith(".csv")]
    archivos_burned = [f for f in archivos_in_folder if 'Burned.csv' in f and 'Not_Burned' not in f]
    archivos_not_burned = [f for f in archivos_in_folder if 'Not_Burned.csv' in f]

    dataset_total = [f for f in archivos_in_folder if 'datasetTotal' in f]
    ruta_dataset_ref = dataset_total[0] if dataset_total else (archivos_in_folder[0] if archivos_in_folder else "")

    if len(archivos_burned) < 3 or len(archivos_not_burned) < 3:
        raise ValueError(
            f"Se requieren al menos 3 poligonos Burned y 3 Not_Burned en {carpeta_csvs}. "
            f"Encontrados: Burned={len(archivos_burned)}, Not_Burned={len(archivos_not_burned)}"
        )

    return nombre_prueba, ruta_dataset_ref, archivos_burned, archivos_not_burned


def main():
    args = construir_argumentos()

    print("=================================================================")
    print("   ENTRENAMIENTO Y EVALUACION EN HOLDOUT ESPACIAL (PHASE 2)")
    print("=================================================================")

    nombre_prueba, ruta_dataset_ref, archivos_b, archivos_nb = obtener_configuracion_incendio(
        args.incendio, args.proyecto, args.prueba, ROOT_DIR
    )

    if args.prueba is None and not args.headless:
        try:
            input_prueba = input(f"Ingrese el nombre identificador de la prueba para trazabilidad (por defecto '{nombre_prueba}'): ").strip()
            if input_prueba:
                nombre_prueba = input_prueba
        except Exception:
            pass

    print(f"[INFO] Prueba: {nombre_prueba}")
    print(f"[INFO] Dataset referencia: {ruta_dataset_ref}")

    # 1. Asignacion automatica de poligonos por tamano (2 Train/Val + 1 Holdout por clase)
    poligonos_tv, poligonos_holdout = organizar_poligonos_por_tamano(archivos_b, archivos_nb)

    # 2. Carga y etiquetado con identificadores regionales
    print("\n=== CARGANDO POLIGONOS ESPACIALES ===")
    df_train_val = cargar_y_etiquetar_poligonos(poligonos_tv)
    df_holdout = cargar_y_etiquetar_poligonos(poligonos_holdout)

    # 3. Seleccion de caracteristicas (GUI o headless)
    print("\n=== SELECCION DE CARACTERISTICAS ===")
    if args.headless:
        # Modo sin GUI: utilizar todos los indices espectrales (excluyendo objetivo)
        cols = pd.read_csv(ruta_dataset_ref, nrows=0).columns.tolist()
        indices_seleccionados = [c for c in cols if 'Burn_Classification' not in c and 'Classification' not in c]
    else:
        indices_seleccionados = crear_interfaz_seleccion_indices(ruta_dataset_ref)

    if indices_seleccionados:
        print(f"[INFO] Caracteristicas seleccionadas ({len(indices_seleccionados)}): {indices_seleccionados}")
    else:
        print("[INFO] No se filtraron caracteristicas. Se utilizaran todas las disponibles.")

    # 4. Seleccion de modelos (GUI o headless)
    print("\n=== SELECCION DE MODELOS ML ===")
    if args.headless:
        # Seleccion de modelos representativos en headless
        modelos_seleccionados = ['SVM_RBF', 'Logistics Regression', 'Random Forest', 'Decision Tree', 'Extra Trees', 'Gradient Boosting Balanced', 'NN-ReLU-2L(200,100)-r2']
    else:
        modelos_seleccionados = crear_interfaz_seleccion_modelos()

    if not modelos_seleccionados:
        print("[WARNING] No se selecciono ningun modelo. Abortando entrenamiento.")
        return

    print(f"[INFO] Modelos a entrenar ({len(modelos_seleccionados)}): {modelos_seleccionados}")

    # 5. Division espacial con StratifiedGroupKFold y escalado sin fuga
    X_train, X_test, X_holdout, y_train, y_test, y_holdout, scaler, feature_names = dividir_y_escalar_dataset(
        df_train_val, df_holdout, indices_seleccionados, target_column="Burn_Classification"
    )

    # 6. Persistencia de dataset dividido
    carpeta_prueba_dir = os.path.join(args.output_dir, nombre_prueba)
    carpeta_dataset_dividido = os.path.join(carpeta_prueba_dir, "dataset_dividido")
    guardar_dataset_dividido(X_train, X_test, X_holdout, y_train, y_test, y_holdout, scaler, carpeta_dataset_dividido)

    # 7. Entrenamiento de modelos y guardado de resultados
    print("\n=== ENTRENAMIENTO DE MODELOS ===")
    for modelo_nombre in modelos_seleccionados:
        res = entrenar_modelo(X_train, X_test, y_train, y_test, modelo_nombre, random_state=args.random_state)
        if res:
            carpeta_modelo_out = os.path.join(carpeta_prueba_dir, modelo_nombre.replace(' ', '_'))
            guardar_modelo_y_artefactos(res, scaler, carpeta_modelo_out, feature_names=feature_names)
            guardar_metricas_modelo(res, carpeta_modelo_out)

    # 8. Evaluacion final cuantitativa en Holdout Espacial
    df_holdout_res = evaluar_todos_los_modelos_de_prueba(
        nombre_prueba=nombre_prueba,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        features=feature_names,
        carpeta_base=args.output_dir
    )

    print("\n[OK] Fase 2 completada exitosamente.")


if __name__ == "__main__":
    main()
