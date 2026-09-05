# -*- coding: utf-8 -*-
"""
Modulo de utilidades de georreferenciacion y conversion de coordenadas cartograficas.
Convierte coordenadas de pixel a sistemas de referencia geograficos WGS84 (EPSG:4326).
"""

import numpy as np

try:
    import rasterio
    from rasterio.transform import xy
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False


def pixel_to_geo_coords(x, y, transform):
    """
    Convierte coordenadas de pixel (columna x, fila y) a coordenadas proyectadas usando la matriz afin.
    
    Args:
        x (int): Indice de columna en pixeles
        y (int): Indice de fila en pixeles
        transform: Transformacion afin de rasterio
        
    Returns:
        tuple: (x_coord, y_coord) o None si no esta disponible
    """
    if transform is None or not RASTERIO_AVAILABLE:
        return None
    try:
        x_coord, y_coord = xy(transform, y, x)
        return x_coord, y_coord
    except Exception as e:
        print(f"[WARNING] Error al convertir coordenadas de pixel: {e}")
        return None


def convert_to_geographic(x_coord, y_coord, source_crs, target_crs='EPSG:4326'):
    """
    Reproyecta coordenadas desde el CRS fuente hacia el CRS geografico objetivo (WGS84 por defecto).
    
    Args:
        x_coord (float): Coordenada X proyectada (Easting)
        y_coord (float): Coordenada Y proyectada (Northing)
        source_crs: Sistema de Referencia de Coordenadas fuente
        target_crs (str): CRS objetivo ('EPSG:4326')
        
    Returns:
        tuple: (longitud, latitud) o None si falla
    """
    if source_crs is None:
        return None

    try:
        if PYPROJ_AVAILABLE:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            lon, lat = transformer.transform(x_coord, y_coord)
            return lon, lat
        elif RASTERIO_AVAILABLE:
            from rasterio.warp import transform as r_transform
            lons, lats = r_transform(source_crs, target_crs, [x_coord], [y_coord])
            return lons[0], lats[0]
    except Exception as e:
        print(f"[WARNING] Error al convertir a coordenadas geograficas: {e}")
        return None
    return None


def format_geo_label(val, is_lat=True):
    """
    Formatea un valor decimal de latitud/longitud al formato academico de grados y minutos.
    Ejemplo: -42.15 -> 42°09'S
    
    Args:
        val (float): Valor decimal de coordenada
        is_lat (bool): True si es latitud, False si es longitud
        
    Returns:
        str: Etiqueta formateada
    """
    if val is None or np.isnan(val):
        return ""

    deg = abs(val)
    d = int(deg)
    m = int(round((deg - d) * 60))
    if m == 60:
        d += 1
        m = 0

    direction = ('S' if val < 0 else 'N') if is_lat else ('W' if val < 0 else 'E')
    return f"{d}°{m:02d}'{direction}"
