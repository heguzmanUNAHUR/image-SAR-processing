# -*- coding: utf-8 -*-
"""
Modulo de calculo de indices espectrales SAR (IRV, NDPI, NDBI, RBR y Normalized).
Preserva el 100% de las formulas matematicas, acotaciones y claves de guardado .npz.
"""

import os
import numpy as np


def process_irv(folder_path, origin, epsilon=1e-10):
    """
    Calcula el Indice Radar de Vegetacion (IRV):
    IRV = 8 * HV / (HH + VV + 2 * HV + epsilon)
    Clave de guardado: 'irv'
    """
    files = [f for f in os.listdir(folder_path)]
    vv_files = [f for f in files if 'VV' in f and origin in f]
    hv_files = [f for f in files if 'HV' in f and origin in f]
    hh_files = [f for f in files if 'HH' in f and origin in f]
    
    if not vv_files or not hv_files or not hh_files:
        print(f"[WARNING] No se encontraron archivos completos (VV, HV, HH) para IRV {origin}")
        return None

    matrix_hv_path = os.path.join(folder_path, hv_files[0])
    matrix_hh_path = os.path.join(folder_path, hh_files[0])
    matrix_vv_path = os.path.join(folder_path, vv_files[0])
    
    output_path = os.path.join(folder_path, f'irv_{origin}_matrix.npz')

    matrix_hv_data = np.load(matrix_hv_path)
    matrix_hh_data = np.load(matrix_hh_path)
    matrix_vv_data = np.load(matrix_vv_path)

    matrix_hv = matrix_hv_data['band']
    matrix_hh = matrix_hh_data['band']
    matrix_vv = matrix_vv_data['band']

    irv = np.divide(8 * matrix_hv, matrix_hh + matrix_vv + 2 * matrix_hv + epsilon)
    irv = np.clip(irv, 0, 20)

    np.savez_compressed(output_path, irv=irv)
    return irv


def process_ndpi(folder_path, origin, epsilon=1e-10):
    """
    Calcula el Normalized Difference Polarization Index (NDPI):
    NDPI = (VV - HV) / (VV + HV + epsilon)
    Clave de guardado: 'ndpi'
    """
    files = [f for f in os.listdir(folder_path)]
    vv_files = [f for f in files if 'VV' in f and origin in f]
    hv_files = [f for f in files if 'HV' in f and origin in f]
    
    if not vv_files or not hv_files:
        print(f"[WARNING] No se encontraron archivos VV o HV para NDPI {origin}")
        return None
    
    matrix_vv_path = os.path.join(folder_path, vv_files[0])
    matrix_hv_path = os.path.join(folder_path, hv_files[0])
    output_path = os.path.join(folder_path, f'ndpi_{origin}_matrix.npz')

    matrix_vv_data = np.load(matrix_vv_path)
    matrix_hv_data = np.load(matrix_hv_path)
    
    matrix_vv = matrix_vv_data['band']
    matrix_hv = matrix_hv_data['band']
    
    ndpi = np.divide(matrix_vv - matrix_hv, matrix_vv + matrix_hv + epsilon)
    ndpi = np.clip(ndpi, -1, 1)
    
    np.savez_compressed(output_path, ndpi=ndpi)
    return ndpi


def process_ndbi(folder_path, origin, epsilon=1e-10):
    """
    Calcula el Normalized Difference Built/Soil Index (NDBI):
    NDBI = (VV - VH) / (VV + VH + epsilon)
    Clave de guardado: 'ndbi'
    """
    files = [f for f in os.listdir(folder_path)]
    vv_files = [f for f in files if 'VV' in f and origin in f]
    vh_files = [f for f in files if 'VH' in f and origin in f]
    
    if not vv_files or not vh_files:
        print(f"[WARNING] No se encontraron archivos VV o VH para NDBI {origin}")
        return None
    
    matrix_vv_path = os.path.join(folder_path, vv_files[0])
    matrix_vh_path = os.path.join(folder_path, vh_files[0])
    output_path = os.path.join(folder_path, f'ndbi_{origin}_matrix.npz')

    matrix_vv_data = np.load(matrix_vv_path)
    matrix_vh_data = np.load(matrix_vh_path)

    matrix_vv = matrix_vv_data['band']
    matrix_vh = matrix_vh_data['band']

    ndbi = np.divide(matrix_vv - matrix_vh, matrix_vv + matrix_vh + epsilon)
    ndbi = np.clip(ndbi, -1, 1)

    np.savez_compressed(output_path, ndbi=ndbi)
    return ndbi


def process_rbr(folder_path, band_type, epsilon=1e-10):
    """
    Calcula Relativized Burn Ratio (RBR):
    RBR = Post / (Pre + epsilon)
    Clave de guardado: 'rbr'
    """
    files = [f for f in os.listdir(folder_path)]
    pre_files = [f for f in files if band_type in f and 'Pre' in f]
    post_files = [f for f in files if band_type in f and 'Post' in f]
    
    if not pre_files or not post_files:
        print(f"[WARNING] No se encontraron archivos Pre o Post para RBR {band_type}")
        return None
    
    matrix_pre_path = os.path.join(folder_path, pre_files[0])
    matrix_post_path = os.path.join(folder_path, post_files[0])
    output_path = os.path.join(folder_path, f'rbr_{band_type}_matrix.npz')

    matrix1_data = np.load(matrix_post_path)
    matrix2_data = np.load(matrix_pre_path)

    keys1 = list(matrix1_data.keys())
    keys2 = list(matrix2_data.keys())
    
    matrix_post = matrix1_data[keys1[0]]
    matrix_pre = matrix2_data[keys2[0]]

    rbr = np.divide(matrix_post, matrix_pre + epsilon)
    rbr = np.clip(rbr, 0, 10)

    np.savez_compressed(output_path, rbr=rbr)
    return rbr


def process_normalized(folder_path, band_type, epsilon=1e-10):
    """
    Calcula la variacion normalizada entre Post y Pre:
    Normalized = (Post - Pre) / (Post + Pre + epsilon)
    Clave de guardado: 'normalized'
    """
    files = [f for f in os.listdir(folder_path)]
    pre_files = [f for f in files if band_type in f and 'Pre' in f]
    post_files = [f for f in files if band_type in f and 'Post' in f]
    
    if not pre_files or not post_files:
        print(f"[WARNING] No se encontraron archivos Pre o Post para Normalized {band_type}")
        return None
    
    matrix_pre_path = os.path.join(folder_path, pre_files[0])
    matrix_post_path = os.path.join(folder_path, post_files[0])
    output_path = os.path.join(folder_path, f'normalized_{band_type}_matrix.npz')

    matrix_post_data = np.load(matrix_post_path)
    matrix_pre_data = np.load(matrix_pre_path)

    keys1 = list(matrix_post_data.keys())
    keys2 = list(matrix_pre_data.keys())
    
    matrix_post = matrix_post_data[keys1[0]]
    matrix_pre = matrix_pre_data[keys2[0]]

    normalized = np.divide(matrix_post - matrix_pre, matrix_post + matrix_pre + epsilon)
    normalized = np.clip(normalized, -1, 1)

    np.savez_compressed(output_path, normalized=normalized)
    return normalized


def compute_all_sar_indices(matrix_dir):
    """
    Ejecuta exactamente la secuencia completa de calculo de indices SAR
    (PASO 3 del pipeline original).
    """
    print("\n" + "="*80)
    print("PASO 3: GENERACION DE INDICES SAR")
    print("="*80)

    # 1. IRV (Post y Pre)
    process_irv(matrix_dir, 'Post')
    print("  [OK] Matriz IRV Post Incendio generada")
    process_irv(matrix_dir, 'Pre')
    print("  [OK] Matriz IRV Pre Incendio generada")

    # 2. NDPI (Pre y Post)
    process_ndpi(matrix_dir, 'Pre')
    print("  [OK] Matriz NDPI Pre incendio generada")
    process_ndpi(matrix_dir, 'Post')
    print("  [OK] Matriz NDPI Post incendio generada")

    # 3. NDBI (Pre y Post)
    process_ndbi(matrix_dir, 'Pre')
    print("  [OK] Matriz NDBI Pre incendio generada")
    process_ndbi(matrix_dir, 'Post')
    print("  [OK] Matriz NDBI Post incendio generada")

    # 4. RBR (VV, VH, NDBI)
    process_rbr(matrix_dir, 'VV')
    print("  [OK] Matriz RBR banda VV generada")
    process_rbr(matrix_dir, 'VH')
    print("  [OK] Matriz RBR banda VH generada")
    process_rbr(matrix_dir, 'ndbi')
    print("  [OK] Matriz RBR banda NDBI generada")

    # 5. Normalized (VV, VH, NDBI)
    process_normalized(matrix_dir, 'VV')
    print("  [OK] Matriz Normalized banda VV generada")
    process_normalized(matrix_dir, 'VH')
    print("  [OK] Matriz Normalized banda VH generada")
    process_normalized(matrix_dir, 'ndbi')
    print("  [OK] Matriz Normalized banda NDBI generada")

    print("\n  [OK] Todos los indices SAR han sido calculados correctamente.")
