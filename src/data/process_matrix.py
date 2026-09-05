# -*- coding: utf-8 -*-
"""
Modulo de conversion de imagenes GeoTIFF a matrices NumPy comprimidas (.npz)
y validacion de dimensiones.
Preserva el 100% de la logica core.
"""

import os
import numpy as np
import rasterio


def calculate_matrix(proyecto, image_path, type_name, origin):
    """
    Convierte una imagen TIFF georreferenciada a matriz usando rasterio.
    Lee correctamente las imagenes de radar (monocromaticas, un solo canal).
    Guarda la matriz comprimida en <proyecto>/Matrixs/<type_name>_<origin>_matrix.npz
    """
    with rasterio.open(image_path) as src:
        img = src.read()
        
        if len(img.shape) == 3:
            img = img[0]
        else:
            img = img
        
        img = img.astype(np.float64)

    matrix_dir = os.path.join(proyecto, 'Matrixs')
    os.makedirs(matrix_dir, exist_ok=True)
    
    output_path = os.path.join(matrix_dir, f'{type_name}_{origin}_matrix.npz')
    np.savez_compressed(output_path, band=img)
    
    print(f"   {type_name}_{origin}: {img.shape} px")


def process_images(proyecto, folder_path, origin):
    """
    Procesa las imagenes GeoTIFF de un directorio de origen ('Pre' o 'Post')
    filtrando por las polarizaciones VV, VH, HV, HH y generando las matrices .npz.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    vv_files = [f for f in files if 'VV' in f]
    vh_files = [f for f in files if 'VH' in f]
    hv_files = [f for f in files if 'HV' in f]
    hh_files = [f for f in files if 'HH' in f]

    print(f"\n  [INFO] Procesando {len(vv_files + vh_files + hv_files + hh_files)} imagenes {origin}...")

    for file in vv_files:
        img_folder = os.path.join(folder_path, file)
        calculate_matrix(proyecto, img_folder, 'VV', origin)

    for file in vh_files:
        img_folder = os.path.join(folder_path, file)
        calculate_matrix(proyecto, img_folder, 'VH', origin)

    for file in hv_files:
        img_folder = os.path.join(folder_path, file)
        calculate_matrix(proyecto, img_folder, 'HV', origin)

    for file in hh_files:
        img_folder = os.path.join(folder_path, file)
        calculate_matrix(proyecto, img_folder, 'HH', origin)


def validate_all_matrix_dimensions(folder_path):
    """
    Valida que todas las matrices Pre y Post en la carpeta Matrixs tengan exactamente
    las mismas dimensiones.
    
    Args:
        folder_path (str): Ruta a la carpeta Matrixs
    
    Returns:
        bool: True si todas las dimensiones coinciden, False en caso contrario
    """
    files = [f for f in os.listdir(folder_path)]
    
    pre_files = [f for f in files if 'Pre' in f]
    post_files = [f for f in files if 'Post' in f]
    
    if not pre_files or not post_files:
        print("[ERROR] No se encontraron archivos Pre o Post")
        return False
    
    print(f"[INFO] Validando dimensiones de {len(pre_files)} archivos Pre y {len(post_files)} archivos Post...")
    
    first_pre_path = os.path.join(folder_path, pre_files[0])
    first_pre_data = np.load(first_pre_path)
    keys_pre = list(first_pre_data.keys())
    reference_shape = first_pre_data[keys_pre[0]].shape
    
    print(f"[INFO] Dimensiones de referencia (Pre): {reference_shape}")
    
    for pre_file in pre_files:
        pre_path = os.path.join(folder_path, pre_file)
        pre_data = np.load(pre_path)
        keys = list(pre_data.keys())
        pre_matrix = pre_data[keys[0]]
        
        if pre_matrix.shape != reference_shape:
            print(f"  [ERROR] {pre_file}: {pre_matrix.shape} (deberia ser {reference_shape})")
            return False
        else:
            print(f"  [OK] {pre_file}: {pre_matrix.shape}")
    
    for post_file in post_files:
        post_path = os.path.join(folder_path, post_file)
        post_data = np.load(post_path)
        keys = list(post_data.keys())
        post_matrix = post_data[keys[0]]
        
        if post_matrix.shape != reference_shape:
            print(f"  [ERROR] {post_file}: {post_matrix.shape} (deberia ser {reference_shape})")
            return False
        else:
            print(f"  [OK] {post_file}: {post_matrix.shape}")
    
    print("[OK] Todas las matrices tienen las mismas dimensiones")
    return True
