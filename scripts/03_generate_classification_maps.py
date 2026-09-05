# -*- coding: utf-8 -*-
"""
Script orquestador ejecutable para la Fase 3: Inferencia espacial masiva sobre escenas completas
y generacion de mapas cartograficos georreferenciados de clasificacion e intensidad.
"""

import os
import sys
import argparse
import glob
import pandas as pd

# Matplotlib backend sin GUI para ejecutables
import matplotlib
matplotlib.use('Agg')

# Incorporar directorio raiz al path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.model_factory import crear_interfaz_seleccion_modelos_metricas
from src.models.inference import generar_predicciones_con_modelos
from src.visualization.map_generator import (
    generar_imagen_clasificacion_geo,
    generar_imagen_clasificacion_con_intensidad_geo
)


def construir_argumentos():
    parser = argparse.ArgumentParser(
        description="Generacion de mapas cartograficos de clasificacion e inferencia espacial masiva."
    )
    parser.add_argument(
        "--incendio",
        type=str,
        default="PONTON",
        choices=["PONTON", "LOS_ALERCES", "SAN_JUAN"],
        help="Nombre del incendio a evaluar."
    )
    parser.add_argument(
        "--prueba",
        type=str,
        default=None,
        help="Nombre de la prueba entrenada en la Fase 2 desde la cual evaluar modelos."
    )
    parser.add_argument(
        "--proyecto",
        type=str,
        default=None,
        help="Ruta alternativa al directorio del incendio."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Ejecutar en modo no interactivo (sin GUI Tkinter)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "models"),
        help="Carpeta base donde residen las pruebas entrenadas."
    )
    return parser.parse_args()


def resolver_rutas_escena(incendio, proyecto_cli, root_dir):
    """
    Resuelve la ruta al CSV completo de la escena (complete_image.csv) y la imagen TIFF original.
    """
    if proyecto_cli:
        incendio_dir = proyecto_cli
    elif incendio == "PONTON":
        incendio_dir = os.path.join(root_dir, "data", "raw", "Ponton")
    elif incendio == "LOS_ALERCES":
        incendio_dir = os.path.join(root_dir, "data", "raw", "Los_Alerces")
    elif incendio == "SAN_JUAN":
        incendio_dir = os.path.join(root_dir, "data", "raw", "San_Juan")
    else:
        raise ValueError(f"Incendio '{incendio}' no reconocido.")

    # Buscar complete_image.csv
    posibles_csvs = [
        os.path.join(incendio_dir, "csvs", "complete_image.csv"),
        os.path.join(incendio_dir, "complete_image.csv"),
    ]
    path_complete_csv = None
    for p in posibles_csvs:
        if os.path.exists(p):
            path_complete_csv = p
            break

    if not path_complete_csv:
        # Buscar en subcarpetas Pruebas
        matches = glob.glob(os.path.join(incendio_dir, "**", "complete_image.csv"), recursive=True)
        if matches:
            path_complete_csv = matches[0]

    # Buscar imagen TIFF post incendio para geocoordenadas
    tiff_matches = glob.glob(os.path.join(incendio_dir, "**", "*VH_Post_Incendio*.tif"), recursive=True)
    if not tiff_matches:
        tiff_matches = glob.glob(os.path.join(incendio_dir, "**", "*.tif"), recursive=True)

    ruta_tiff = tiff_matches[0] if tiff_matches else None
    return path_complete_csv, ruta_tiff, incendio_dir


def main():
    args = construir_argumentos()

    print("=================================================================")
    print("   GENERACION DE MAPAS CARTOGRAFICOS E INFERENCIA (FASE 3)")
    print("=================================================================")

    # Resolver nombre de prueba
    nombre_prueba = args.prueba

    if not nombre_prueba and not args.headless:
        # Escanear pruebas disponibles en output_dir
        pruebas_disponibles = []
        if os.path.exists(args.output_dir):
            pruebas_disponibles = [
                d for d in os.listdir(args.output_dir)
                if os.path.isdir(os.path.join(args.output_dir, d))
            ]

        print("\n=== PRUEBAS ENTRENADAS DISPONIBLES EN SISTEMA ===")
        if pruebas_disponibles:
            for idx, p in enumerate(pruebas_disponibles, 1):
                print(f"  {idx}. {p}")
            default_p = pruebas_disponibles[0]
        else:
            default_p = "Prueba Ponton GKF Spatial Holdout (Tesis)"

        try:
            val_in = input(f"\nIngrese el nombre de la prueba desde la que desea evaluar los modelos (por defecto '{default_p}'): ").strip()
            if val_in.isdigit() and 1 <= int(val_in) <= len(pruebas_disponibles):
                nombre_prueba = pruebas_disponibles[int(val_in) - 1]
            elif val_in:
                nombre_prueba = val_in
            else:
                nombre_prueba = default_p
        except Exception:
            nombre_prueba = default_p

    if not nombre_prueba:
        nombre_prueba = "Prueba Ponton GKF Spatial Holdout (Tesis)"

    print(f"\n[INFO] Evaluando modelos de la prueba: {nombre_prueba}")

    # Carpeta base de la prueba
    carpeta_prueba = os.path.join(args.output_dir, nombre_prueba)
    if not os.path.exists(carpeta_prueba):
        print(f"[ERROR] La carpeta de la prueba no existe: {carpeta_prueba}")
        return

    # Buscar archivo CSV de metricas
    metricas_path = os.path.join(carpeta_prueba, "metricas_holdout_todos_los_modelos.csv")
    if not os.path.exists(metricas_path):
        metricas_path = os.path.join(carpeta_prueba, "metricas_prueba_todos_los_modelos.csv")

    if not os.path.exists(metricas_path):
        print(f"[ERROR] No se encontro el archivo de metricas en: {carpeta_prueba}")
        return

    # Resolver dataset completo y TIFF
    path_complete_csv, ruta_tiff, _ = resolver_rutas_escena(args.incendio, args.proyecto, ROOT_DIR)

    if not path_complete_csv or not os.path.exists(path_complete_csv):
        print(f"[ERROR] No se encontro el archivo complete_image.csv para el incendio {args.incendio}.")
        return

    print(f"[INFO] Escena completa (CSV): {path_complete_csv}")
    print(f"[INFO] GeoTIFF de referencia: {ruta_tiff}")

    # Seleccionar modelos a evaluar
    print("\n=== SELECCION DE MODELOS PARA GENERACION DE MAPAS ===")
    modelos_seleccionados = crear_interfaz_seleccion_modelos_metricas(metricas_path, headless=args.headless)

    if not modelos_seleccionados:
        print("[WARNING] No se seleccionaron modelos para evaluar. Abortando.")
        return

    print(f"[INFO] Modelos seleccionados ({len(modelos_seleccionados)}): {[m['Modelo'] for m in modelos_seleccionados]}")

    colores = {0: 'green', 1: 'red'}

    # Procesar cada modelo seleccionado
    print("\n=== INFERENCIA Y GENERACION DE MAPAS PNG ===")
    for modelo_info in modelos_seleccionados:
        nombre_modelo = modelo_info['Modelo']
        carpeta_sub = modelo_info.get('Carpeta', nombre_modelo.replace(' ', '_'))
        features = modelo_info['Features']

        path_modelo_dir = os.path.join(carpeta_prueba, carpeta_sub)

        print(f"\n-----------------------------------------------------------------")
        print(f"Procesando inferencia para modelo: {nombre_modelo}")
        print(f"-----------------------------------------------------------------")

        # Inferencia masiva sobre complete_image.csv
        generar_predicciones_con_modelos(
            dataset_path=path_complete_csv,
            features_list=features,
            carpeta_modelos=path_modelo_dir,
            carpeta_salida=path_modelo_dir
        )

        nombre_safe = nombre_modelo.replace(' ', '_')
        csv_pred = os.path.join(path_modelo_dir, f"predicciones_{nombre_safe}.csv")

        if os.path.exists(csv_pred):
            # 1. Mapa de clasificacion binaria georreferenciado
            out_png_clasif = os.path.join(carpeta_prueba, f"clasificacion_{nombre_safe}.png")
            generar_imagen_clasificacion_geo(
                ruta_csv=csv_pred,
                columna_prediccion=f"Prediccion_{nombre_safe}",
                ruta_salida=out_png_clasif,
                colores=colores,
                ruta_imagen_original=ruta_tiff
            )

            # 2. Mapa de intensidad de probabilidad continua georreferenciado
            out_png_intensidad = os.path.join(carpeta_prueba, f"clasificacion_con_intensidad_{nombre_safe}.png")
            col_prob = f"Probabilidad_Clase_1_{nombre_safe}"

            # Verificar que exista la columna de probabilidad en el CSV de predicciones
            df_check = pd.read_csv(csv_pred, nrows=0)
            if col_prob in df_check.columns:
                generar_imagen_clasificacion_con_intensidad_geo(
                    ruta_csv=csv_pred,
                    columna_probabilidad_clase_1=col_prob,
                    ruta_salida=out_png_intensidad,
                    ruta_imagen_original=ruta_tiff
                )
        else:
            print(f"[ERROR] No se pudo encontrar el CSV de prediccion: {csv_pred}")

    print("\n[OK] Fase 3 completada exitosamente. Todos los mapas PNG han sido generados.")


if __name__ == "__main__":
    main()
