# -*- coding: utf-8 -*-
"""
Script orquestador principal para la Fase 1: Alineacion de imagenes GeoTIFF,
conversion a matrices, calculo de indices SAR y generacion del dataset CSV.

Preserva el 100% de la secuencia de ejecucion original.
"""

import sys
import os
import glob
import argparse

# Asegurar que la raiz del proyecto este en sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importaciones modulares desde src.data
from src.data.alignment import align_images_to_common_size, validate_images_aligned
from src.data.process_matrix import process_images, validate_all_matrix_dimensions
from src.data.sar_indices import compute_all_sar_indices
from src.data.sampling import (
    selected_zone_image,
    combine_all_csv_files,
    almacenar_prueba,
    save_image_with_zones
)


def run_pipeline_generate_dataset(proyecto, matrices_proyecto, cantidad_muestras,
                                 imagen_capturas, carpeta_pruebas, nombre_prueba, carpeta_imagenes):
    """
    Ejecuta la secuencia completa de procesamiento de imagenes y generacion del dataset.
    """
    # PASO 0: ALINEACION DE IMAGENES GEORREFERENCIADAS
    print("\n" + "="*80)
    print("PASO 0: ALINEACION DE IMAGENES GEORREFERENCIADAS")
    print("="*80)
    folder_path_pre = os.path.join(proyecto, 'Images', 'Pre')
    folder_path_post = os.path.join(proyecto, 'Images', 'Post')

    if not align_images_to_common_size(proyecto, folder_path_pre, folder_path_post):
        print("[ERROR] Error al alinear las imagenes")
        sys.exit(1)

    if not validate_images_aligned(proyecto, folder_path_pre, folder_path_post):
        print("[ERROR] Error: Las imagenes no estan correctamente alineadas")
        sys.exit(1)

    # PASO 1: LIMPIEZA Y CONVERSION DE IMAGENES TIFF A MATRICES
    print("\n" + "="*80)
    print("PASO 1: LIMPIEZA Y CONVERSION DE IMAGENES TIFF A MATRICES")
    print("="*80)

    if os.path.exists(matrices_proyecto):
        matrix_files = glob.glob(os.path.join(matrices_proyecto, '*.npz'))
        if matrix_files:
            print(f"\n[INFO] Eliminando {len(matrix_files)} matrices antiguas...")
            for matrix_file in matrix_files:
                os.remove(matrix_file)
            print("   [OK] Matrices antiguas eliminadas")

    process_images(proyecto, folder_path_pre, 'Pre')
    print("[OK] Matrices de imagenes pre incendio generadas")
    process_images(proyecto, folder_path_post, 'Post')
    print("[OK] Matrices de imagenes post incendio generadas")

    # PASO 2: VALIDAR QUE LAS MATRICES TENGAN EL MISMO TAMANO
    print("\n" + "="*80)
    print("PASO 2: VALIDACION DE DIMENSIONES DE MATRICES")
    print("="*80)
    if not validate_all_matrix_dimensions(matrices_proyecto):
        print("[ERROR] Error: Las matrices no tienen el mismo tamano")
        sys.exit(1)

    # PASO 3: GENERACION DE INDICES SAR
    compute_all_sar_indices(matrices_proyecto)

    # PASO 4: SELECCIONAR LAS ZONAS DE LA IMAGEN PARA GENERAR EL DATASET
    print("\n" + "="*80)
    print(f"PASO 4: SELECCION DE {cantidad_muestras} ZONAS")
    print("="*80)
    for i in range(cantidad_muestras):
        print(f"\n--- Seleccionando zona {i+1}/{cantidad_muestras} ---")
        selected_zone_image(imagen_capturas, matrices_proyecto, carpeta_pruebas, nombre_prueba)
    
    print("\n[OK] DATASET SAR GENERADO CORRECTAMENTE!!!")

    combine_all_csv_files(carpeta_pruebas, nombre_prueba)
    almacenar_prueba(carpeta_imagenes, matrices_proyecto, carpeta_pruebas, nombre_prueba)

    # PASO 5: GENERACION DE IMAGEN FINAL CON ZONAS SELECCIONADAS
    print("\n" + "="*80)
    print("PASO 5: GENERACION DE IMAGEN FINAL CON ZONAS SELECCIONADAS")
    print("="*80)
    imagen_final = save_image_with_zones(imagen_capturas, carpeta_pruebas, nombre_prueba)
    if imagen_final:
        print("[OK] Imagen final generada exitosamente")
    else:
        print("[WARNING] No se pudo generar la imagen final")

    print("\n" + "="*80)
    print("[OK] PRUEBA ALMACENADA CORRECTAMENTE!!!")
    print("="*80)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_proyecto = os.path.join(base_dir, 'data', 'raw', 'Ponton')

    parser = argparse.ArgumentParser(description="Generacion de Dataset SAR para un Incendio Especifico")
    parser.add_argument('--proyecto', type=str, default=default_proyecto, help="Ruta a la carpeta del incendio")
    parser.add_argument('--prueba', type=str, default="Prueba_Muestra", help="Nombre de la prueba")
    parser.add_argument('--muestras', type=int, default=None, help="Cantidad de zonas/poligonos a muestrear")

    args = parser.parse_args()

    proyecto_path = os.path.normpath(args.proyecto)
    matrices_path = os.path.join(proyecto_path, 'Matrixs')
    
    if args.prueba == "Prueba_Muestra":
        prueba_input = input("Ingrese el nombre identificador de la prueba para trazabilidad (por defecto 'Prueba_Muestra'): ").strip()
        prueba_name = prueba_input if prueba_input else "Prueba_Muestra"
    else:
        prueba_name = args.prueba

    if args.muestras is None:
        try:
            val_input = input("Ingrese la cantidad de poligonos/zonas a seleccionar (por defecto 1): ").strip()
            muestras = int(val_input) if val_input else 1
        except Exception:
            muestras = 1
    else:
        muestras = args.muestras

    capturas_img = os.path.join(proyecto_path, 'Images', 'Post', 'Band_Sigma0_VH_Post_Incendio.tif')
    pruebas_folder = os.path.join(proyecto_path, 'Pruebas', '')
    imagenes_folder = os.path.join(proyecto_path, 'Images', '')

    run_pipeline_generate_dataset(
        proyecto=proyecto_path,
        matrices_proyecto=matrices_path,
        cantidad_muestras=muestras,
        imagen_capturas=capturas_img,
        carpeta_pruebas=pruebas_folder,
        nombre_prueba=prueba_name,
        carpeta_imagenes=imagenes_folder
    )
