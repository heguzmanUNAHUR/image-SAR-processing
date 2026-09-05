# -*- coding: utf-8 -*-
"""
Modulo de alineacion de imagenes georreferenciadas GeoTIFF.
Preserva el 100% de la logica core de recorte por interseccion geografica y tamano comun.
"""

import os
import rasterio
from rasterio.transform import xy
import numpy as np
import shutil
from datetime import datetime


def get_image_info(image_path):
    """
    Obtiene informacion de una imagen georreferenciada.
    
    Args:
        image_path (str): Ruta a la imagen TIFF
        
    Returns:
        dict: Diccionario con informacion de la imagen (shape, transform, bounds, crs)
    """
    with rasterio.open(image_path) as src:
        return {
            'shape': (src.height, src.width),
            'transform': src.transform,
            'bounds': src.bounds,
            'crs': src.crs,
            'dtype': src.dtypes[0],
            'count': src.count,
            'nodata': src.nodata
        }


def get_geographic_intersection(all_images_info):
    """
    Calcula la interseccion geografica de todas las imagenes.
    
    Args:
        all_images_info (list): Lista de diccionarios con informacion de cada imagen
        
    Returns:
        tuple: (left, bottom, right, top) de la interseccion geografica
    """
    first_bounds = all_images_info[0]['bounds']
    left, bottom, right, top = first_bounds
    
    for img_info in all_images_info[1:]:
        bounds = img_info['bounds']
        left = max(left, bounds.left)
        bottom = max(bottom, bounds.bottom)
        right = min(right, bounds.right)
        top = min(top, bounds.top)
    
    return (left, bottom, right, top)


def calculate_common_pixel_bounds(all_images_info, common_bounds):
    """
    Calcula los limites en pixeles comunes para todas las imagenes
    basandose en la interseccion geografica.
    
    Args:
        all_images_info (list): Lista de diccionarios con informacion de cada imagen
        common_bounds (tuple): (left, bottom, right, top) de la interseccion geografica
        
    Returns:
        tuple: (min_width, min_height) en pixeles
    """
    left, bottom, right, top = common_bounds
    pixel_sizes = []
    
    for img_info in all_images_info:
        transform = img_info['transform']
        row_ul, col_ul = ~transform * (left, top)
        row_lr, col_lr = ~transform * (right, bottom)
        
        width = int(abs(col_lr - col_ul))
        height = int(abs(row_ul - row_lr))
        
        pixel_sizes.append((width, height))
    
    min_width = min(size[0] for size in pixel_sizes)
    min_height = min(size[1] for size in pixel_sizes)
    
    return (min_width, min_height)


def align_images_to_common_size(proyecto_folder, images_pre_folder, images_post_folder):
    """
    Alinea todas las imagenes georreferenciadas a un tamano comun.
    
    Args:
        proyecto_folder (str): Ruta a la carpeta del proyecto
        images_pre_folder (str): Ruta a la carpeta de imagenes Pre
        images_post_folder (str): Ruta a la carpeta de imagenes Post
        
    Returns:
        bool: True si el proceso fue exitoso, False en caso contrario
    """
    print("=" * 80)
    print("ALINEACION DE IMAGENES GEORREFERENCIADAS")
    print("=" * 80)
    
    pre_images = [f for f in os.listdir(images_pre_folder) if f.endswith('.tif')]
    post_images = [f for f in os.listdir(images_post_folder) if f.endswith('.tif')]
    
    all_images = []
    for img_file in pre_images:
        all_images.append({
            'path': os.path.join(images_pre_folder, img_file),
            'name': img_file,
            'type': 'Pre'
        })
    for img_file in post_images:
        all_images.append({
            'path': os.path.join(images_post_folder, img_file),
            'name': img_file,
            'type': 'Post'
        })
    
    if len(all_images) == 0:
        print("[ERROR] No se encontraron imagenes TIFF")
        return False
    
    print(f"\n[INFO] Se encontraron {len(all_images)} imagenes para alinear:")
    for img in all_images:
        print(f"   - {img['name']} ({img['type']})")
    
    print("\n[INFO] Analizando imagenes...")
    all_images_info = []
    
    for img in all_images:
        try:
            info = get_image_info(img['path'])
            info['path'] = img['path']
            info['name'] = img['name']
            info['type'] = img['type']
            all_images_info.append(info)
            
            print(f"   [OK] {img['name']}: {info['shape'][1]}x{info['shape'][0]} px")
            print(f"      Geografia: ({info['bounds'].left:.6f}, {info['bounds'].bottom:.6f}) a "
                  f"({info['bounds'].right:.6f}, {info['bounds'].top:.6f})")
        except Exception as e:
            print(f"   [ERROR] Error al leer {img['name']}: {e}")
            return False
    
    first_shape = all_images_info[0]['shape']
    same_size = all(info['shape'] == first_shape for info in all_images_info)
    
    if same_size:
        print(f"\n[OK] Todas las imagenes tienen el mismo tamano: {first_shape[1]}x{first_shape[0]} px")
        first_bounds = all_images_info[0]['bounds']
        same_geography = all(info['bounds'] == first_bounds for info in all_images_info)
        
        if same_geography:
            print("[OK] Todas las imagenes tienen la misma geografia")
            print("   No es necesario alinear las imagenes")
            return True
        else:
            print("[WARNING] Las imagenes tienen diferentes geografias")
            print("   Se procedera a alinear...")
    else:
        print("\n[WARNING] Las imagenes tienen diferentes tamanos:")
        for info in all_images_info:
            print(f"   - {info['name']}: {info['shape'][1]}x{info['shape'][0]} px")
    
    print("\n[INFO] Calculando interseccion geografica...")
    common_bounds = get_geographic_intersection(all_images_info)
    print(f"   Interseccion: ({common_bounds[0]:.6f}, {common_bounds[1]:.6f}) a "
          f"({common_bounds[2]:.6f}, {common_bounds[3]:.6f})")
    
    min_width, min_height = calculate_common_pixel_bounds(all_images_info, common_bounds)
    print(f"\n[INFO] Tamano comun: {min_width}x{min_height} px")
    
    temp_folder_pre = os.path.join(images_pre_folder, 'aligned_temp')
    temp_folder_post = os.path.join(images_post_folder, 'aligned_temp')
    
    os.makedirs(temp_folder_pre, exist_ok=True)
    os.makedirs(temp_folder_post, exist_ok=True)
    
    print("\n[INFO] Recortando imagenes al tamano comun...")
    
    reference_info = None
    for img_info in all_images_info:
        if img_info['type'] == 'Pre':
            reference_info = img_info
            break
    
    if reference_info is None:
        reference_info = all_images_info[0]
    
    reference_transform = reference_info['transform']
    projected_crs = reference_info['crs']
    
    print(f"   Referencia: {reference_info['name']}")
    print(f"   Transform: {reference_transform}")
    
    for img_info in all_images_info:
        try:
            output_folder = temp_folder_pre if img_info['type'] == 'Pre' else temp_folder_post
            output_path = os.path.join(output_folder, img_info['name'])
            
            with rasterio.open(img_info['path']) as src:
                transform = src.transform
                row_ul, col_ul = ~transform * (common_bounds[0], common_bounds[3])
                
                col_ul = int(np.floor(col_ul))
                row_ul = int(np.floor(row_ul))
                
                col_ul = max(0, col_ul)
                row_ul = max(0, row_ul)
                col_lr = min(src.width, col_ul + min_width)
                row_lr = min(src.height, row_ul + min_height)
                
                actual_width = col_lr - col_ul
                actual_height = row_lr - row_ul
                
                window = rasterio.windows.Window(col_ul, row_ul, actual_width, actual_height)
                data = src.read(window=window)
                
                out_width = min_width
                out_height = min_height
                
                if actual_width != out_width or actual_height != out_height:
                    output_data = np.full((src.count, out_height, out_width), 
                                         src.nodata if src.nodata is not None else 0, 
                                         dtype=data.dtype)
                    copy_w = min(actual_width, out_width)
                    copy_h = min(actual_height, out_height)
                    output_data[:, :copy_h, :copy_w] = data[:, :copy_h, :copy_w]
                    data = output_data
                
                row_ul_ref, col_ul_ref = ~reference_transform * (common_bounds[0], common_bounds[3])
                col_ul_ref = int(np.floor(col_ul_ref))
                row_ul_ref = int(np.floor(row_ul_ref))
                
                new_transform = rasterio.Affine.translation(col_ul_ref, row_ul_ref) * reference_transform
                
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=out_height,
                    width=out_width,
                    count=src.count,
                    dtype=data.dtype,
                    crs=projected_crs,
                    transform=new_transform,
                    compress='lzw',
                    nodata=src.nodata
                ) as dst:
                    dst.write(data)
            
            print(f"   [OK] {img_info['name']}: Recortado a {out_width}x{out_height} px")
            
        except Exception as e:
            print(f"   [ERROR] Error al procesar {img_info['name']}: {e}")
            return False
    
    print("\n[INFO] Reemplazando imagenes originales con las alineadas...")
    for img_file in pre_images:
        src_path = os.path.join(temp_folder_pre, img_file)
        dst_path = os.path.join(images_pre_folder, img_file)
        backup_path = dst_path + '.backup'
        if os.path.exists(dst_path) and not os.path.exists(backup_path):
            shutil.copy2(dst_path, backup_path)
        shutil.move(src_path, dst_path)
        print(f"   [OK] {img_file} (Pre) - Backup guardado en: {backup_path}")
    
    for img_file in post_images:
        src_path = os.path.join(temp_folder_post, img_file)
        dst_path = os.path.join(images_post_folder, img_file)
        backup_path = dst_path + '.backup'
        if os.path.exists(dst_path) and not os.path.exists(backup_path):
            shutil.copy2(dst_path, backup_path)
        shutil.move(src_path, dst_path)
        print(f"   [OK] {img_file} (Post) - Backup guardado en: {backup_path}")
    
    try:
        os.rmdir(temp_folder_pre)
        os.rmdir(temp_folder_post)
    except Exception:
        pass
    
    print("\n[OK] Alineacion completada exitosamente")
    print("=" * 80)
    return True


def validate_images_aligned(proyecto_folder, images_pre_folder, images_post_folder):
    """
    Valida que las imagenes estan alineadas correctamente.
    
    Args:
        proyecto_folder (str): Ruta a la carpeta del proyecto
        images_pre_folder (str): Ruta a la carpeta de imagenes Pre
        images_post_folder (str): Ruta a la carpeta de imagenes Post
        
    Returns:
        bool: True si todas las imagenes estan alineadas
    """
    print("\n[INFO] Validando alineacion de imagenes...")
    
    pre_images = [f for f in os.listdir(images_pre_folder) if f.endswith('.tif')]
    post_images = [f for f in os.listdir(images_post_folder) if f.endswith('.tif')]
    
    all_images = []
    for img_file in pre_images:
        img_path = os.path.join(images_pre_folder, img_file)
        info = get_image_info(img_path)
        info['name'] = img_file
        all_images.append(info)
    
    for img_file in post_images:
        img_path = os.path.join(images_post_folder, img_file)
        info = get_image_info(img_path)
        info['name'] = img_file
        all_images.append(info)
    
    if len(all_images) == 0:
        print("[ERROR] No se encontraron imagenes")
        return False
    
    first_shape = all_images[0]['shape']
    same_size = all(info['shape'] == first_shape for info in all_images)
    
    if not same_size:
        print("[ERROR] Las imagenes no tienen el mismo tamano")
        for info in all_images:
            print(f"   {info['name']}: {info['shape'][1]}x{info['shape'][0]} px")
        return False
    
    first_bounds = all_images[0]['bounds']
    tolerance = 1.0
    same_geography = True
    for info in all_images:
        bounds = info['bounds']
        if not (abs(bounds.left - first_bounds.left) < tolerance and
                abs(bounds.bottom - first_bounds.bottom) < tolerance and
                abs(bounds.right - first_bounds.right) < tolerance and
                abs(bounds.top - first_bounds.top) < tolerance):
            same_geography = False
            break
    
    if not same_geography:
        print("[WARNING] Las imagenes no tienen la misma geografia (aunque si el mismo tamano)")
        first_transform = all_images[0]['transform']
        pixel_size_x = abs(first_transform[0])
        pixel_size_y = abs(first_transform[4])
        
        max_diff = 0
        for info in all_images:
            bounds = info['bounds']
            diff = max(abs(bounds.left - first_bounds.left),
                      abs(bounds.bottom - first_bounds.bottom),
                      abs(bounds.right - first_bounds.right),
                      abs(bounds.top - first_bounds.top))
            max_diff = max(max_diff, diff)
        
        if max_diff < max(pixel_size_x, pixel_size_y):
            print(f"[OK] Las imagenes estan suficientemente alineadas (diferencia max {max_diff:.2f}m < pixel {max(pixel_size_x, pixel_size_y):.2f}m)")
            return True
        else:
            print("[ERROR] Las diferencias exceden el tamano de pixel")
            return False
    
    print(f"[OK] Todas las imagenes estan perfectamente alineadas: {first_shape[1]}x{first_shape[0]} px")
    return True
