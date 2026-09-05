# -*- coding: utf-8 -*-
"""
Modulo de generacion de mapas cartograficos de clasificacion y probabilidad continua.
Renderiza mapas PNG de alta resolucion con grilla georreferenciada WGS84.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from src.visualization.geo_utils import (
    pixel_to_geo_coords,
    convert_to_geographic,
    format_geo_label
)


def generar_imagen_clasificacion_geo(ruta_csv, columna_prediccion, ruta_salida, colores=None, ruta_imagen_original=None):
    """
    Genera y guarda un mapa de clasificacion binaria georreferenciado (PNG).
    
    Args:
        ruta_csv (str): Ruta al CSV con predicciones y coordenadas (X, Y)
        columna_prediccion (str): Nombre de la columna con las clases predichas
        ruta_salida (str): Ruta de destino para el archivo PNG
        colores (dict, optional): Mapa de colores por clase {0: 'green', 1: 'red'}
        ruta_imagen_original (str, optional): Ruta al GeoTIFF original para extraer metadatos CRS
    """
    print(f"[INFO] Generando mapa de clasificacion desde: {ruta_csv}")
    
    try:
        df = pd.read_csv(ruta_csv)
    except Exception as e:
        print(f"[ERROR] Error al leer CSV de predicciones: {e}")
        return None

    if 'X' not in df.columns or 'Y' not in df.columns or columna_prediccion not in df.columns:
        print(f"[ERROR] El CSV debe contener las columnas X, Y y {columna_prediccion}")
        return None

    max_x, min_x = int(df['X'].max()), int(df['X'].min())
    max_y, min_y = int(df['Y'].max()), int(df['Y'].min())
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    matriz_clasificacion = np.full((height, width), np.nan)
    for _, row in df.iterrows():
        x = int(row['X'] - min_x)
        y = int(row['Y'] - min_y)
        matriz_clasificacion[y, x] = row[columna_prediccion]

    clases_unicas = sorted([c for c in df[columna_prediccion].unique() if not pd.isna(c)])
    
    if colores is None:
        colores = {0: 'green', 1: 'red', 2: 'blue', 3: 'yellow'}

    colores_lista = [colores.get(clase, 'gray') for clase in clases_unicas]
    cmap = mcolors.ListedColormap(colores_lista)

    transform = None
    crs = None
    if ruta_imagen_original and RASTERIO_AVAILABLE and os.path.exists(ruta_imagen_original):
        try:
            with rasterio.open(ruta_imagen_original) as src:
                transform = src.transform
                crs = src.crs
        except Exception as e:
            print(f"[WARNING] No se pudo extraer georreferenciacion del GeoTIFF: {e}")

    fig, ax = plt.subplots(figsize=(12, 10))
    matriz_rotada = np.rot90(matriz_clasificacion, k=1)
    im = ax.imshow(matriz_rotada, cmap=cmap, interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Clasificacion', rotation=270, labelpad=20)
    cbar.set_ticks(clases_unicas)
    cbar.set_ticklabels([f'Clase {int(c)}' for c in clases_unicas])

    ax.set_title(f'Mapa de Clasificacion - {columna_prediccion}', fontsize=14, fontweight='bold', pad=15)

    if transform and crs:
        ax.set_xlabel('Longitud', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitud', fontsize=12, fontweight='bold')

        num_ticks = 5
        x_ticks = np.linspace(0, width, num_ticks)
        y_ticks = np.linspace(0, height, num_ticks)

        x_labels = []
        for x_tick in x_ticks:
            coords = pixel_to_geo_coords(int(x_tick + min_x), int(height / 2 + min_y), transform)
            if coords:
                res = convert_to_geographic(coords[0], coords[1], crs)
                x_labels.append(format_geo_label(res[0], is_lat=False) if res else f"{coords[0]:.2f}")
            else:
                x_labels.append(f"{x_tick:.0f}")

        y_labels = []
        for y_tick in y_ticks:
            coords = pixel_to_geo_coords(int(width / 2 + min_x), int(y_tick + min_y), transform)
            if coords:
                res = convert_to_geographic(coords[0], coords[1], crs)
                y_labels.append(format_geo_label(res[1], is_lat=True) if res else f"{coords[1]:.2f}")
            else:
                y_labels.append(f"{y_tick:.0f}")

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=35, ha='right')
        ax.set_yticklabels(y_labels)
    else:
        ax.set_xlabel('Coordenada X (pixeles)', fontsize=12)
        ax.set_ylabel('Coordenada Y (pixeles)', fontsize=12)

    ax.invert_yaxis()
    plt.tight_layout()

    if not os.path.exists(os.path.dirname(ruta_salida)):
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Mapa de clasificacion guardado en: {ruta_salida}")

    return ruta_salida


def generar_imagen_clasificacion_con_intensidad_geo(ruta_csv, columna_probabilidad_clase_1, ruta_salida, ruta_imagen_original=None):
    """
    Genera y guarda un mapa continuo de intensidad de probabilidad continua (PNG).
    
    Args:
        ruta_csv (str): Ruta al CSV con predicciones y probabilidades
        columna_probabilidad_clase_1 (str): Nombre de la columna con la probabilidad de quemado
        ruta_salida (str): Ruta de destino para el archivo PNG
        ruta_imagen_original (str, optional): Ruta al GeoTIFF original para extraer metadatos CRS
    """
    print(f"[INFO] Generando mapa de intensidad de probabilidad desde: {ruta_csv}")
    
    try:
        df = pd.read_csv(ruta_csv)
    except Exception as e:
        print(f"[ERROR] Error al leer CSV: {e}")
        return None

    if 'X' not in df.columns or 'Y' not in df.columns or columna_probabilidad_clase_1 not in df.columns:
        print(f"[ERROR] Columna de probabilidad {columna_probabilidad_clase_1} no encontrada en CSV")
        return None

    max_x, min_x = int(df['X'].max()), int(df['X'].min())
    max_y, min_y = int(df['Y'].max()), int(df['Y'].min())
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    matriz_rgb = np.ones((height, width, 3))

    for _, row in df.iterrows():
        x = int(row['X'] - min_x)
        y = int(row['Y'] - min_y)
        prob = row[columna_probabilidad_clase_1]

        if prob > 0.5:
            intensidad = (1.0 - prob) * 2.0
            matriz_rgb[y, x] = [1.0, np.clip(intensidad, 0.0, 1.0), np.clip(intensidad, 0.0, 1.0)]
        else:
            intensidad = prob * 2.0
            matriz_rgb[y, x] = [np.clip(intensidad, 0.0, 1.0), 1.0, np.clip(intensidad, 0.0, 1.0)]

    transform = None
    crs = None
    if ruta_imagen_original and RASTERIO_AVAILABLE and os.path.exists(ruta_imagen_original):
        try:
            with rasterio.open(ruta_imagen_original) as src:
                transform = src.transform
                crs = src.crs
        except Exception as e:
            print(f"[WARNING] Error leyendo metadatos georreferenciados: {e}")

    fig, ax = plt.subplots(figsize=(12, 10))
    matriz_rotada = np.rot90(matriz_rgb, k=1)
    ax.imshow(matriz_rotada, interpolation='nearest')

    ax.set_title(f'Mapa de Intensidad de Probabilidad de Quemado\n({columna_probabilidad_clase_1})',
                 fontsize=14, fontweight='bold', pad=15)

    if transform and crs:
        ax.set_xlabel('Longitud', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitud', fontsize=12, fontweight='bold')

        num_ticks = 5
        x_ticks = np.linspace(0, width, num_ticks)
        y_ticks = np.linspace(0, height, num_ticks)

        x_labels = []
        for x_tick in x_ticks:
            coords = pixel_to_geo_coords(int(x_tick + min_x), int(height / 2 + min_y), transform)
            if coords:
                res = convert_to_geographic(coords[0], coords[1], crs)
                x_labels.append(format_geo_label(res[0], is_lat=False) if res else f"{coords[0]:.2f}")
            else:
                x_labels.append(f"{x_tick:.0f}")

        y_labels = []
        for y_tick in y_ticks:
            coords = pixel_to_geo_coords(int(width / 2 + min_x), int(y_tick + min_y), transform)
            if coords:
                res = convert_to_geographic(coords[0], coords[1], crs)
                y_labels.append(format_geo_label(res[1], is_lat=True) if res else f"{coords[1]:.2f}")
            else:
                y_labels.append(f"{y_tick:.0f}")

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=35, ha='right')
        ax.set_yticklabels(y_labels)

    ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor='red', label='Alta Probabilidad de Quemado (Prob > 0.5)'),
        Patch(facecolor='green', label='Baja Probabilidad de Quemado (Prob <= 0.5)'),
        Patch(facecolor='white', label='Incertidumbre Moderada (Prob ~ 0.5)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if not os.path.exists(os.path.dirname(ruta_salida)):
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Mapa de intensidad guardado en: {ruta_salida}")

    return ruta_salida
