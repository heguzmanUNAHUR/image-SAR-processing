# -*- coding: utf-8 -*-
"""
Modulo de seleccion de zonas (poligonos), trazabilidad JSON y generacion de datasets CSV.
Preserva el 100% de la logica core, encabezados CSV, clasificacion e interfaz de usuario.
"""

import tkinter as tk
import cv2
import numpy as np
import csv
import os
import pandas as pd
import json
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import rasterio
    from rasterio.transform import xy
    from rasterio.warp import transform as rasterio_transform
    RASTERIO_AVAILABLE = True
except ImportError:
    print("[WARNING] rasterio no esta disponible. Las coordenadas geograficas no se mostraran.")
    RASTERIO_AVAILABLE = False

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    print("[WARNING] pyproj no esta disponible. La conversion a coordenadas geograficas puede no funcionar.")
    PYPROJ_AVAILABLE = False


def load_selected_zones(nombreProyecto, nombrePrueba):
    """
    Carga las zonas previamente seleccionadas desde un archivo JSON.
    """
    zones_file = os.path.join(nombreProyecto, nombrePrueba, 'selected_zones.json')
    
    if not os.path.exists(zones_file):
        return []
    
    try:
        with open(zones_file, 'r') as f:
            data = json.load(f)
            return data.get('zones', [])
    except Exception as e:
        print(f"Error al cargar zonas: {e}")
        return []


def save_selected_zone(x1, y1, x2, y2, clasificacion, nombreProyecto, nombrePrueba):
    """
    Guarda una zona seleccionada en el archivo JSON.
    """
    zones_file = os.path.join(nombreProyecto, nombrePrueba, 'selected_zones.json')
    
    if os.path.exists(zones_file):
        with open(zones_file, 'r') as f:
            data = json.load(f)
    else:
        data = {'zones': [], 'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    new_zone = {
        'x1': int(x1),
        'y1': int(y1),
        'x2': int(x2),
        'y2': int(y2),
        'classification': clasificacion,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    data['zones'].append(new_zone)
    data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    os.makedirs(os.path.dirname(zones_file), exist_ok=True)
    with open(zones_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Zona guardada: {new_zone}")


def pixel_to_geo_coords(x, y, transform):
    """
    Convierte coordenadas de pixeles a coordenadas geograficas o proyectadas.
    """
    if transform is None or not RASTERIO_AVAILABLE:
        return None
    try:
        x_coord, y_coord = xy(transform, y, x)
        return x_coord, y_coord
    except Exception as e:
        print(f"Error al convertir coordenadas: {e}")
        return None


def convert_to_geographic(x_coord, y_coord, source_crs, target_crs='EPSG:4326'):
    """
    Convierte coordenadas de un sistema de coordenadas a coordenadas geograficas (WGS84).
    """
    if not PYPROJ_AVAILABLE or source_crs is None:
        return None
    
    try:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        lon, lat = transformer.transform(x_coord, y_coord)
        return lon, lat
    except Exception as e:
        print(f"Error al convertir a coordenadas geograficas: {e}")
        return None


def save_image_with_zones(image_path, nombreProyecto, nombrePrueba):
    """
    Genera y guarda una imagen con todas las zonas seleccionadas marcadas y ejes geograficos.
    """
    transform = None
    bounds = None
    crs = None
    if RASTERIO_AVAILABLE:
        try:
            with rasterio.open(image_path) as src:
                transform = src.transform
                bounds = src.bounds
                crs = src.crs
                img = src.read(1)
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                print(f"[OK] Informacion georreferenciada leida del TIFF")
                if crs:
                    print(f"  CRS: {crs}")
        except Exception as e:
            print(f"[WARNING] No se pudo leer informacion georreferenciada: {e}")
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) > 2:
                img = img[..., 0]
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) > 2:
            img = img[..., 0]
    
    zones = load_selected_zones(nombreProyecto, nombrePrueba)
    
    if not zones:
        print("No hay zonas seleccionadas para mostrar")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, cmap='gray', interpolation='nearest')
    
    color_map = {
        'Burned': 'red',
        'Semiburned': 'orange',
        'Not_Burned': 'green'
    }
    
    for idx, zone in enumerate(zones, 1):
        x1, y1 = zone['x1'], zone['y1']
        x2, y2 = zone['x2'], zone['y2']
        classification = zone['classification']
        color = color_map.get(classification, 'white')
        
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle((x1, y1), width, height, linewidth=3, 
                                edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        label_num = f"#{idx}"
        ax.text(x1 + 5, y1 + 10, label_num, fontsize=8, color=color, 
               weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='black', alpha=0.8))
        
        classification_label = classification.replace('_', ' ')
        if classification == 'Burned':
            label_bg_color = 'red'
            label_text_color = 'white'
        elif classification == 'Not_Burned':
            label_bg_color = 'green'
            label_text_color = 'white'
        else:
            label_bg_color = 'orange'
            label_text_color = 'white'
        
        label_x = (x1 + x2) / 2
        label_y = y1 - 4
        ax.text(label_x, label_y, classification_label, fontsize=10, color=label_text_color,
               weight='bold', bbox=dict(boxstyle='round,pad=0.4', 
                                        facecolor=label_bg_color, alpha=0.9),
               verticalalignment='top', horizontalalignment='center')
        
        if transform and crs:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            projected_coords = pixel_to_geo_coords(int(center_x), int(center_y), transform)
            if projected_coords:
                x_coord, y_coord = projected_coords
                latlon_coords = convert_to_geographic(x_coord, y_coord, crs)
                if latlon_coords:
                    lon, lat = latlon_coords
                    coord_text = f"Lat: {lat:.6f}\nLon: {lon:.6f}"
                else:
                    coord_text = f"X: {x_coord:.1f}\nY: {y_coord:.1f}"
                coord_x = x2 + 3
                coord_y = (y1 + y2) / 2
                ax.text(coord_x, coord_y, coord_text, fontsize=6, color='yellow', 
                       weight='bold', bbox=dict(boxstyle='round,pad=0.2', 
                                                facecolor='black', alpha=0.8),
                       verticalalignment='center', horizontalalignment='left')
    
    if bounds and RASTERIO_AVAILABLE and crs:
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        
        height_px, width_px = img.shape[:2]
        left_proj, bottom_proj, right_proj, top_proj = bounds
        
        if PYPROJ_AVAILABLE:
            try:
                left_lon, bottom_lat = convert_to_geographic(left_proj, bottom_proj, crs) or (left_proj, bottom_proj)
                right_lon, top_lat = convert_to_geographic(right_proj, top_proj, crs) or (right_proj, top_proj)
                left_lon2, top_lat2 = convert_to_geographic(left_proj, top_proj, crs) or (left_proj, top_proj)
                right_lon2, bottom_lat2 = convert_to_geographic(right_proj, bottom_proj, crs) or (right_proj, bottom_proj)
                lon_min = min(left_lon, left_lon2)
                lon_max = max(right_lon, right_lon2)
                lat_min = min(bottom_lat, bottom_lat2)
                lat_max = max(top_lat, top_lat2)
            except Exception:
                lon_min, lat_min, lon_max, lat_max = left_proj, bottom_proj, right_proj, top_proj
        else:
            lon_min, lat_min, lon_max, lat_max = left_proj, bottom_proj, right_proj, top_proj
        
        num_ticks = 5
        x_ticks = np.linspace(0, width_px, num_ticks)
        y_ticks = np.linspace(0, height_px, num_ticks)
        
        x_labels = []
        for x_tick in x_ticks:
            if transform:
                coords = pixel_to_geo_coords(int(x_tick), int(height_px/2), transform)
                if coords:
                    tick_x_proj, tick_y_proj = coords
                    if PYPROJ_AVAILABLE:
                        result = convert_to_geographic(tick_x_proj, tick_y_proj, crs)
                        if result:
                            tick_lon, _ = result
                            x_labels.append(f"{tick_lon:.6f}")
                        else:
                            x_labels.append(f"{tick_x_proj:.2f}")
                    else:
                        x_labels.append(f"{tick_x_proj:.2f}")
                else:
                    tick_lon = lon_min + (lon_max - lon_min) * (x_tick / width_px)
                    x_labels.append(f"{tick_lon:.6f}")
            else:
                tick_lon = lon_min + (lon_max - lon_min) * (x_tick / width_px)
                x_labels.append(f"{tick_lon:.6f}")
        
        y_labels = []
        for y_tick in y_ticks:
            if transform:
                coords = pixel_to_geo_coords(int(width_px/2), int(y_tick), transform)
                if coords:
                    tick_x_proj, tick_y_proj = coords
                    if PYPROJ_AVAILABLE:
                        result = convert_to_geographic(tick_x_proj, tick_y_proj, crs)
                        if result:
                            _, tick_lat = result
                            y_labels.append(f"{tick_lat:.6f}")
                        else:
                            y_labels.append(f"{tick_y_proj:.2f}")
                    else:
                        y_labels.append(f"{tick_y_proj:.2f}")
                else:
                    tick_lat = lat_min + (lat_max - lat_min) * (1 - (y_tick / height_px))
                    y_labels.append(f"{tick_lat:.6f}")
            else:
                tick_lat = lat_min + (lat_max - lat_min) * (1 - (y_tick / height_px))
                y_labels.append(f"{tick_lat:.6f}")
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels(y_labels)
    else:
        ax.set_xlabel('X (pixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (pixels)', fontsize=12, fontweight='bold')
    
    ax.set_title('Zonas Seleccionadas - Imagen SAR', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_dir = os.path.join(nombreProyecto, nombrePrueba)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'zonas_seleccionadas.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Imagen con zonas guardada en: {output_path}")
    print(f"  Total de zonas marcadas: {len(zones)}")
    return output_path


def save_values_to_csv(ndpi_Pre_values, ndpi_Post_values, ndbi_Pre_values, ndbi_Post_values, normalized_ndbi_values, normalized_VV_values, normalized_VH_values,
                       rbr_ndbi_values, rbr_VV_values, rbr_VH_values, HH_Pre_values, HV_Pre_values, VH_Pre_values,
                       VV_Pre_values, HH_Post_values, HV_Post_values, VH_Post_values, VV_Post_values, irv_Pre_values, irv_Post_values,
                       x1, x2, y1, y2, burn_classification, nombreProyecto, nombrePrueba, image_complete=False):
    """
    Guarda los vectores de caracteristicas por pixel en archivo CSV.
    Preserva exactamente las 21 columnas y valores de clasificacion (0: Not_Burned, 1: Burned, 2: Semiburned).
    """
    if not (len(ndpi_Pre_values) == len(ndpi_Post_values) == len(ndbi_Pre_values) == len(ndbi_Post_values) ==
            len(normalized_ndbi_values) == len(normalized_VV_values) ==
            len(normalized_VH_values) == len(rbr_ndbi_values) ==
            len(rbr_VV_values) == len(rbr_VH_values) == len(HH_Pre_values) == len(HV_Pre_values) == len(VH_Pre_values) ==
            len(VV_Pre_values) == len(HH_Post_values) == len(HV_Post_values) == len(VH_Post_values) == len(VV_Post_values) ==
            len(irv_Pre_values) == len(irv_Post_values)):
        raise ValueError("All vectors must have the same length")

    output_dir = os.path.join(nombreProyecto, nombrePrueba)
    os.makedirs(output_dir, exist_ok=True)

    name = f"{x1}_{x2}_{y1}_{y2}"
    vectors_output = f"vectors_output_{name}_{burn_classification}.csv"
    output_path = os.path.join(output_dir, vectors_output)

    if not image_complete:
        with open(output_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            csv_writer.writerow(['NDPI_Pre', 'NDPI_Post', 'NDBI_Pre', 'NDBI_Post', 'NORMALIZED_NDBI', 'NORMALIZED_VV', 'NORMALIZED_VH', 'RBR_NDBI',
                                 'RBR_VV', 'RBR_VH', 'HH_Pre', 'HV_Pre', 'VH_Pre', 'VV_Pre',
                                 'HH_Post', 'HV_Post', 'VH_Post', 'VV_Post', 'IRV_Pre', 'IRV_Post', 'Burn_Classification'])
            numBurnClassification = 0
            if burn_classification == "Burned":
                numBurnClassification = 1
            elif burn_classification == "Semiburned":
                numBurnClassification = 2

            x_totales = x2 - x1
            y_totales = y2 - y1
            contador = 0
            for i in range(x_totales):
                for j in range(y_totales):
                    csv_writer.writerow([
                        ndpi_Pre_values[contador],
                        ndpi_Post_values[contador],
                        ndbi_Pre_values[contador],
                        ndbi_Post_values[contador],
                        normalized_ndbi_values[contador],
                        normalized_VV_values[contador],
                        normalized_VH_values[contador],
                        rbr_ndbi_values[contador],
                        rbr_VV_values[contador],
                        rbr_VH_values[contador],
                        HH_Pre_values[contador],
                        HV_Pre_values[contador],
                        VH_Pre_values[contador],
                        VV_Pre_values[contador],
                        HH_Post_values[contador],
                        HV_Post_values[contador],
                        VH_Post_values[contador],
                        VV_Post_values[contador],
                        irv_Pre_values[contador],
                        irv_Post_values[contador],
                        numBurnClassification
                    ])
                    contador += 1
    else:
        output_path_complete = os.path.join(output_dir, 'complete_image.csv')
        with open(output_path_complete, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            csv_writer.writerow(['X', 'Y', 'NDPI_Pre', 'NDPI_Post', 'NDBI_Pre', 'NDBI_Post', 'NORMALIZED_NDBI', 'NORMALIZED_VV', 'NORMALIZED_VH', 'RBR_NDBI',
                                 'RBR_VV', 'RBR_VH', 'HH_Pre', 'HV_Pre', 'VH_Pre', 'VV_Pre',
                                 'HH_Post', 'HV_Post', 'VH_Post', 'VV_Post', 'IRV_Pre', 'IRV_Post', 'Burn_Classification'])
            numBurnClassification = 0
            if burn_classification == "Burned":
                numBurnClassification = 1
            elif burn_classification == "Semiburned":
                numBurnClassification = 2

            x_totales = x2 - x1
            y_totales = y2 - y1
            contador = 0
            for i in range(x_totales):
                for j in range(y_totales):
                    csv_writer.writerow([
                        i,
                        j,
                        ndpi_Pre_values[contador],
                        ndpi_Post_values[contador],
                        ndbi_Pre_values[contador],
                        ndbi_Post_values[contador],
                        normalized_ndbi_values[contador],
                        normalized_VV_values[contador],
                        normalized_VH_values[contador],
                        rbr_ndbi_values[contador],
                        rbr_VV_values[contador],
                        rbr_VH_values[contador],
                        HH_Pre_values[contador],
                        HV_Pre_values[contador],
                        VH_Pre_values[contador],
                        VV_Pre_values[contador],
                        HH_Post_values[contador],
                        HV_Post_values[contador],
                        VH_Post_values[contador],
                        VV_Post_values[contador],
                        irv_Pre_values[contador],
                        irv_Post_values[contador],
                        numBurnClassification
                    ])
                    contador += 1

    print(f"Vectors saved to {output_path}")


class BurnClassificationApp:
    def __init__(self, master):
        self.master = master
        master.title("Burn Classification")
        master.geometry("300x200")

        self.label = tk.Label(master, text="Classify the selected region:", font=("Arial", 12))
        self.label.pack(pady=20)

        self.burned_button = tk.Button(master, text="Burned", command=lambda: self.classify_region("Burned"),
                                       bg="red", fg="white", font=("Arial", 12))
        self.burned_button.pack(pady=10)

        self.semiburned_button = tk.Button(master, text="Semiburned", command=lambda: self.classify_region("Semiburned"),
                                           bg="orange", fg="white", font=("Arial", 12))
        self.semiburned_button.pack(pady=10)

        self.not_burned_button = tk.Button(master, text="Not Burned",
                                           command=lambda: self.classify_region("Not_Burned"),
                                           bg="green", fg="white", font=("Arial", 12))
        self.not_burned_button.pack(pady=10)

        self.classification = None

    def classify_region(self, classification):
        self.classification = classification
        self.master.destroy()


def printRegion(region, y1, x1, titulo):
    pixel_values = []
    print(f"\n*************: {titulo}")
    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            if len(region.shape) > 2:
                pixel_value = region[i, j][0]
            else:
                pixel_value = region[i, j]
            pixel_values.append(pixel_value)
            print(f"Position ({i + y1}, {j + x1}): {pixel_value}")
    return pixel_values


def select_zone(image_path, ndpi_Pre_matrix_path, ndpi_Post_matrix_path, ndbi_Pre_matrix_path, ndbi_Post_matrix_path, normalized_ndbi_matrix_path, normalized_vh_matrix_path,
                normalized_VV_matrix_path, rbr_ndbi_matrix_path, rbr_VH_matrix_path, rbr_VV_matrix_path, HH_Post_matrix_path,
                HV_Post_matrix_path, VH_Post_matrix_path, VV_Post_matrix_path, HH_Pre_matrix_path, HV_Pre_matrix_path,
                VH_Pre_matrix_path, VV_Pre_matrix_path, irv_Pre_matrix_path, 
                irv_Post_matrix_path, nombreProyecto, nombrePrueba):
    if RASTERIO_AVAILABLE:
        try:
            with rasterio.open(image_path) as src:
                img = src.read(1)
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
        except Exception as e:
            print(f"[WARNING] Error al leer imagen con rasterio: {e}")
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) > 2:
                img = img[..., 0]
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) > 2:
            img = img[..., 0]

    ndpi_data = np.load(ndpi_Pre_matrix_path)
    ndpi_matrix_Pre = ndpi_data['ndpi']

    ndpi_data = np.load(ndpi_Post_matrix_path)
    ndpi_matrix_Post = ndpi_data['ndpi']

    ndbi_data = np.load(ndbi_Pre_matrix_path)
    ndbi_matrix_Pre = ndbi_data['ndbi']

    ndbi_data = np.load(ndbi_Post_matrix_path)
    ndbi_matrix_Post = ndbi_data['ndbi']

    normalized_ndbi_data = np.load(normalized_ndbi_matrix_path)
    normalized_ndbi_matrix = normalized_ndbi_data['normalized']

    normalized_vh_data = np.load(normalized_vh_matrix_path)
    normalized_vh_matrix = normalized_vh_data['normalized']

    normalized_VV_data = np.load(normalized_VV_matrix_path)
    normalized_VV_matrix = normalized_VV_data['normalized']

    rbr_ndbi_data = np.load(rbr_ndbi_matrix_path)
    rbr_ndbi_matrix = rbr_ndbi_data['rbr']

    rbr_VH_data = np.load(rbr_VH_matrix_path)
    rbr_VH_matrix = rbr_VH_data['rbr']

    rbr_VV_data = np.load(rbr_VV_matrix_path)
    rbr_VV_matrix = rbr_VV_data['rbr']

    HH_Post_data = np.load(HH_Post_matrix_path)
    HH_Post_matrix = HH_Post_data['band']

    HV_Post_data = np.load(HV_Post_matrix_path)
    HV_Post_matrix = HV_Post_data['band']

    VH_Post_data = np.load(VH_Post_matrix_path)
    VH_Post_matrix = VH_Post_data['band']

    VV_Post_data = np.load(VV_Post_matrix_path)
    VV_Post_matrix = VV_Post_data['band']

    HH_Pre_data = np.load(HH_Pre_matrix_path)
    HH_Pre_matrix = HH_Pre_data['band']

    HV_Pre_data = np.load(HV_Pre_matrix_path)
    HV_Pre_matrix = HV_Pre_data['band']

    VH_Pre_data = np.load(VH_Pre_matrix_path)
    VH_Pre_matrix = VH_Pre_data['band']

    VV_Pre_data = np.load(VV_Pre_matrix_path)
    VV_Pre_matrix = VV_Pre_data['band']

    irv_data = np.load(irv_Pre_matrix_path)
    irv_matrix_Pre = irv_data['irv']

    irv_data = np.load(irv_Post_matrix_path)
    irv_matrix_Post = irv_data['irv']

    display_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    clone = img.copy()
    points = []

    previous_zones = load_selected_zones(nombreProyecto, nombrePrueba)
    
    color_map = {
        'Burned': (0, 0, 255),
        'Semiburned': (0, 165, 255),
        'Not_Burned': (0, 255, 0)
    }
    
    if previous_zones:
        print(f"\n=== ZONAS PREVIAMENTE SELECCIONADAS: {len(previous_zones)} ===")
        for idx, zone in enumerate(previous_zones, 1):
            x1, y1 = zone['x1'], zone['y1']
            x2, y2 = zone['x2'], zone['y2']
            classification = zone['classification']
            color = color_map.get(classification, (255, 255, 255))
            
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 1)
            label = f"#{idx}"
            cv2.putText(display_img, label, 
                       (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display_img, (x, y), 2, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.rectangle(display_img, points[0], points[1], (0, 255, 0), 1)
            cv2.imshow("Image", display_img)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1200, 800)
    cv2.setMouseCallback("Image", mouse_callback)
    cv2.imshow("Image", display_img)

    while len(points) < 2:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None, []

    x1, y1 = min(points[0][0], points[1][0]), min(points[0][1], points[1][1])
    x2, y2 = max(points[0][0], points[1][0]), max(points[0][1], points[1][1])

    cv2.destroyAllWindows()

    root = tk.Tk()
    app = BurnClassificationApp(root)
    root.mainloop()

    if app.classification is None:
        print("No classification selected. Aborting.")
        return None, []

    region_ndpi_Pre = ndpi_matrix_Pre[y1:y2, x1:x2]
    ndpi_Pre_values = printRegion(region_ndpi_Pre, y1, x1, 'NDPI')

    region_ndpi_Post = ndpi_matrix_Post[y1:y2, x1:x2]
    ndpi_Post_values = printRegion(region_ndpi_Post, y1, x1, 'NDPI')

    region_ndbi_Pre = ndbi_matrix_Pre[y1:y2, x1:x2]
    ndbi_Pre_values = printRegion(region_ndbi_Pre, y1, x1, 'NDBI')

    region_ndbi_Post = ndbi_matrix_Post[y1:y2, x1:x2]
    ndbi_Post_values = printRegion(region_ndbi_Post, y1, x1, 'NDBI')

    region_normalized_ndbi = normalized_ndbi_matrix[y1:y2, x1:x2]
    normalized_ndbi_values = printRegion(region_normalized_ndbi, y1, x1, 'Normalized ndbi')

    region_normalized_vh = normalized_vh_matrix[y1:y2, x1:x2]
    normalized_vh_values = printRegion(region_normalized_vh, y1, x1, 'Normalized VH')

    region_normalized_VV = normalized_VV_matrix[y1:y2, x1:x2]
    normalized_VV_values = printRegion(region_normalized_VV, y1, x1, 'Normalized VV')

    region_rbr_ndbi = rbr_ndbi_matrix[y1:y2, x1:x2]
    rbr_ndbi_values = printRegion(region_rbr_ndbi, y1, x1, 'RBR NDBI')

    region_rbr_VV = rbr_VV_matrix[y1:y2, x1:x2]
    rbr_VV_values = printRegion(region_rbr_VV, y1, x1, 'RBR VV')

    region_rbr_VH = rbr_VH_matrix[y1:y2, x1:x2]
    rbr_VH_values = printRegion(region_rbr_VH, y1, x1, 'RBR VH')

    region_HH_Pre = HH_Pre_matrix[y1:y2, x1:x2]
    HH_Pre_values = printRegion(region_HH_Pre, y1, x1, 'HH Pre')

    region_HV_Pre = HV_Pre_matrix[y1:y2, x1:x2]
    HV_Pre_values = printRegion(region_HV_Pre, y1, x1, 'HV Pre')

    region_VH_Pre = VH_Pre_matrix[y1:y2, x1:x2]
    VH_Pre_values = printRegion(region_VH_Pre, y1, x1, 'VH Pre')

    region_VV_Pre = VV_Pre_matrix[y1:y2, x1:x2]
    VV_Pre_values = printRegion(region_VV_Pre, y1, x1, 'VV Pre')

    region_HH_Post = HH_Post_matrix[y1:y2, x1:x2]
    HH_Post_values = printRegion(region_HH_Post, y1, x1, 'HH Post')

    region_HV_Post = HV_Post_matrix[y1:y2, x1:x2]
    HV_Post_values = printRegion(region_HV_Post, y1, x1, 'HV Post')

    region_VH_Post = VH_Post_matrix[y1:y2, x1:x2]
    VH_Post_values = printRegion(region_VH_Post, y1, x1, 'VH Post')

    region_VV_Post = VV_Post_matrix[y1:y2, x1:x2]
    VV_Post_values = printRegion(region_VV_Post, y1, x1, 'VV Post')

    region_irv_Pre = irv_matrix_Pre[y1:y2, x1:x2]
    irv_Pre_values = printRegion(region_irv_Pre, y1, x1, 'IRV')

    region_irv_Post = irv_matrix_Post[y1:y2, x1:x2]
    irv_Post_values = printRegion(region_irv_Post, y1, x1, 'IRV')

    save_values_to_csv(
        ndpi_Pre_values,
        ndpi_Post_values,
        ndbi_Pre_values,
        ndbi_Post_values,
        normalized_ndbi_values,
        normalized_VV_values,
        normalized_vh_values,
        rbr_ndbi_values,
        rbr_VV_values,
        rbr_VH_values,
        HH_Pre_values,
        HV_Pre_values,
        VH_Pre_values,
        VV_Pre_values,
        HH_Post_values,
        HV_Post_values,
        VH_Post_values,
        VV_Post_values,
        irv_Pre_values,
        irv_Post_values,
        x1, x2, y1, y2,
        app.classification,
        nombreProyecto,
        nombrePrueba
    )

    save_selected_zone(x1, y1, x2, y2, app.classification, nombreProyecto, nombrePrueba)
    
    print(f"\n[OK] Zona clasificada como: {app.classification}")
    print(f"  Coordenadas: ({x1},{y1}) a ({x2},{y2})")

    return region_rbr_ndbi, []


def combine_all_csv_files(nombreProyecto, nombrePrueba):
    """
    Combina todos los archivos CSV generados en un solo dataset master.
    """
    import glob
    
    csv_directory = os.path.join(nombreProyecto, nombrePrueba)
    csv_pattern = os.path.join(csv_directory, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No se encontraron archivos CSV en {csv_directory}")
        return None
    
    print(f"Encontrados {len(csv_files)} archivos CSV:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    all_dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_dataframes.append(df)
            print(f"[OK] Leido: {os.path.basename(csv_file)} - {len(df)} filas")
        except Exception as e:
            print(f"[ERROR] Error al leer {csv_file}: {e}")
    
    if not all_dataframes:
        print("No se pudieron leer archivos CSV validos")
        return None
    
    try:
        dataset_total = pd.concat(all_dataframes, ignore_index=True)
        print(f"\nDataset total creado exitosamente:")
        print(f"  - Total de filas: {len(dataset_total)}")
        print(f"  - Total de columnas: {len(dataset_total.columns)}")
        print(f"  - Columnas: {list(dataset_total.columns)}")
        
        clean_project_name = os.path.basename(os.path.normpath(nombreProyecto))
        output_filename = f"datasetTotal_{clean_project_name}_{nombrePrueba}.csv"
        output_path = os.path.join(csv_directory, output_filename)
        dataset_total.to_csv(output_path, index=False)
        print(f"  - Guardado en: {output_path}")
        
        return dataset_total
        
    except Exception as e:
        print(f"Error al combinar los DataFrames: {e}")
        return None


def selected_zone_image(imageSelectionPath, matrixsPath, nombreProyecto, nombrePrueba):
    """
    Encuentra las matrices .npz en matrixsPath y ejecuta select_zone.
    """
    files = [f for f in os.listdir(matrixsPath)]
    ndpi_Pre_matrix_path = os.path.join(matrixsPath, [f for f in files if 'ndpi_Pre' in f][0])
    ndpi_Post_matrix_path = os.path.join(matrixsPath, [f for f in files if 'ndpi_Post' in f][0])
    ndbi_Pre_matrix_path = os.path.join(matrixsPath, [f for f in files if 'ndbi_Pre' in f][0])
    ndbi_Post_matrix_path = os.path.join(matrixsPath, [f for f in files if 'ndbi_Post' in f][0])
    normalized_ndbi_matrix_path = os.path.join(matrixsPath, [f for f in files if 'normalized_ndbi' in f][0])
    normalized_VH_matrix_path = os.path.join(matrixsPath, [f for f in files if 'normalized_VH' in f][0])
    normalized_VV_matrix_path = os.path.join(matrixsPath, [f for f in files if 'normalized_VV' in f][0])
    rbr_ndbi_matrix_path = os.path.join(matrixsPath, [f for f in files if 'rbr_ndbi' in f][0])
    rbr_VH_matrix_path = os.path.join(matrixsPath, [f for f in files if 'rbr_VH' in f][0])
    rbr_VV_matrix_path = os.path.join(matrixsPath, [f for f in files if 'rbr_VV' in f][0])
    HH_Post_matrix_path = os.path.join(matrixsPath, [f for f in files if 'HH_Post' in f][0])
    HV_Post_matrix_path = os.path.join(matrixsPath, [f for f in files if 'HV_Post' in f][0])
    VH_Post_matrix_path = os.path.join(matrixsPath, [f for f in files if 'VH_Post' in f][0])
    VV_Post_matrix_path = os.path.join(matrixsPath, [f for f in files if 'VV_Post' in f][0])
    HH_Pre_matrix_path = os.path.join(matrixsPath, [f for f in files if 'HH_Pre' in f][0])
    HV_Pre_matrix_path = os.path.join(matrixsPath, [f for f in files if 'HV_Pre' in f][0])
    VH_Pre_matrix_path = os.path.join(matrixsPath, [f for f in files if 'VH_Pre' in f][0])
    VV_Pre_matrix_path = os.path.join(matrixsPath, [f for f in files if 'VV_Pre' in f][0])
    irv_Pre_matrix_path = os.path.join(matrixsPath, [f for f in files if 'irv_Pre' in f][0])
    irv_Post_matrix_path = os.path.join(matrixsPath, [f for f in files if 'irv_Post' in f][0])

    return select_zone(
        imageSelectionPath,
        ndpi_Pre_matrix_path,
        ndpi_Post_matrix_path,
        ndbi_Pre_matrix_path,
        ndbi_Post_matrix_path,
        normalized_ndbi_matrix_path,
        normalized_VH_matrix_path,
        normalized_VV_matrix_path,
        rbr_ndbi_matrix_path,
        rbr_VH_matrix_path,
        rbr_VV_matrix_path,
        HH_Post_matrix_path,
        HV_Post_matrix_path,
        VH_Post_matrix_path,
        VV_Post_matrix_path,
        HH_Pre_matrix_path,
        HV_Pre_matrix_path,
        VH_Pre_matrix_path,
        VV_Pre_matrix_path,
        irv_Pre_matrix_path,
        irv_Post_matrix_path,
        nombreProyecto,
        nombrePrueba
    )


def almacenar_prueba(carpeta_imagenes, carpeta_matrices, carpeta_prueba, nombre_prueba):
    """
    Funcion para almacenar una copia de las imagenes y matrices utilizadas en la carpeta de pruebas.
    """
    try:
        carpeta_prueba_full = carpeta_prueba + nombre_prueba
        if not os.path.exists(carpeta_prueba_full):
            os.makedirs(carpeta_prueba_full)
        
        carpeta_csvs = os.path.join(carpeta_prueba_full, 'csvs')
        if not os.path.exists(carpeta_csvs):
            os.makedirs(carpeta_csvs)
        
        archivos_csv = [f for f in os.listdir(carpeta_prueba_full) if f.endswith('.csv')]
        for archivo_csv in archivos_csv:
            origen = os.path.join(carpeta_prueba_full, archivo_csv)
            destino = os.path.join(carpeta_csvs, archivo_csv)
            shutil.move(origen, destino)
            print(f"  - Archivo CSV movido: {archivo_csv}")
        
        print(f"Archivos CSV organizados en: {carpeta_csvs}")
        
        carpeta_imagenes_prueba = os.path.join(carpeta_prueba_full, 'Imagenes_Utilizadas')
        carpeta_matrices_prueba = os.path.join(carpeta_prueba_full, 'Matrices_Utilizadas')
        
        if not os.path.exists(carpeta_imagenes_prueba):
            os.makedirs(carpeta_imagenes_prueba)
        if not os.path.exists(carpeta_matrices_prueba):
            os.makedirs(carpeta_matrices_prueba)
        
        print(f"Copiando imagenes desde {carpeta_imagenes}...")
        if os.path.exists(carpeta_imagenes):
            carpeta_pre = os.path.join(carpeta_imagenes, 'Pre')
            if os.path.exists(carpeta_pre):
                carpeta_pre_destino = os.path.join(carpeta_imagenes_prueba, 'Pre')
                if os.path.exists(carpeta_pre_destino):
                    shutil.rmtree(carpeta_pre_destino)
                shutil.copytree(carpeta_pre, carpeta_pre_destino)
                print("  - Carpeta Pre copiada")
            
            carpeta_post = os.path.join(carpeta_imagenes, 'Post')
            if os.path.exists(carpeta_post):
                carpeta_post_destino = os.path.join(carpeta_imagenes_prueba, 'Post')
                if os.path.exists(carpeta_post_destino):
                    shutil.rmtree(carpeta_post_destino)
                shutil.copytree(carpeta_post, carpeta_post_destino)
                print("  - Carpeta Post copiada")
        
        print(f"Copiando matrices desde {carpeta_matrices}...")
        if os.path.exists(carpeta_matrices):
            if os.path.exists(carpeta_matrices_prueba):
                shutil.rmtree(carpeta_matrices_prueba)
            shutil.copytree(carpeta_matrices, carpeta_matrices_prueba)
            print("  - Carpeta de matrices copiada")
        
        print(f"Eliminando archivos de la carpeta de matrices: {carpeta_matrices}")
        if os.path.exists(carpeta_matrices):
            for archivo in os.listdir(carpeta_matrices):
                ruta_archivo = os.path.join(carpeta_matrices, archivo)
                if os.path.isfile(ruta_archivo):
                    os.remove(ruta_archivo)
                    print(f"  - Archivo eliminado: {archivo}")
                elif os.path.isdir(ruta_archivo):
                    shutil.rmtree(ruta_archivo)
                    print(f"  - Carpeta eliminada: {archivo}")
            
            print("Carpeta de matrices limpiada completamente")
        print(f"Proceso de almacenamiento completado en: {carpeta_prueba_full}")
        
    except Exception as e:
        print(f"Error al almacenar la prueba: {str(e)}")
