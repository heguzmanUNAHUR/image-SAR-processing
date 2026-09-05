# -*- coding: utf-8 -*-
"""
Módulo de Selección e Inspección Dinámica de Incendios (Step 4)
--------------------------------------------------------------
Proporciona utilidades para escanear datasets de incendios, validar el
cumplimiento de la regla de mínimo 6 polígonos (3 Burned + 3 Not Burned),
organizar polígonos por tamaño (1 menor por clase a Test, resto a Train/Val)
y desplegar la interfaz gráfica interactiva de selección múltiple N -> M.
"""

import os
import glob
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox


def escanear_incendios_disponibles(base_dir="data/raw"):
    """
    Escanea dinámicamente el directorio base de datos y ubicaciones de respaldo
    para detectar cada prueba realizada por incendio que contenga archivos de polígonos CSV válidos y simétricos (>= 3B / >= 3NB).
    
    Retorna:
        dict: Diccionario con la estructura de cada dataset de prueba válido encontrado.
    """
    incendios = {}
    carpetas_procesadas = set()
    
    def procesar_carpeta_csvs(csvs_path, nombre_incendio, nombre_prueba):
        abs_path = os.path.abspath(csvs_path)
        if abs_path in carpetas_procesadas:
            return
            
        if os.path.exists(abs_path):
            all_csv_files = glob.glob(os.path.join(abs_path, "vectors_output_*.csv"))
            b_files = [f for f in all_csv_files if f.endswith("_Burned.csv") and not f.endswith("_Not_Burned.csv")]
            nb_files = [f for f in all_csv_files if f.endswith("_Not_Burned.csv")]
            
            # REGLA ESTRICTA: Simétricos y mínimo 3 polígonos por clase (>= 6 en total)
            if len(b_files) >= 3 and len(nb_files) >= 3 and len(b_files) == len(nb_files):
                carpetas_procesadas.add(abs_path)
                dataset_ref = glob.glob(os.path.join(abs_path, "datasetTotal_*.csv"))
                ref_path = dataset_ref[0] if dataset_ref else None
                
                key_name = f"{nombre_incendio.upper()}__{nombre_prueba.upper()}".replace(" ", "_").replace("-", "_")
                label_lindo = f"{nombre_incendio.replace('_', ' ')} (Prueba: {nombre_prueba})"
                
                incendios[key_name] = {
                    "nombre_id": key_name,
                    "incendio_base": nombre_incendio.replace("_", " "),
                    "nombre_prueba": nombre_prueba,
                    "nombre_lindo": label_lindo,
                    "carpeta_csvs": abs_path,
                    "dataset_ref": ref_path,
                    "archivos_burned": b_files,
                    "archivos_not_burned": nb_files
                }

    # 1. Buscar en data/raw/<Incendio>/Pruebas/<Prueba>/csvs/
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # Buscar en subcarpetas Pruebas
                pruebas_dir = os.path.join(item_path, "Pruebas")
                if os.path.exists(pruebas_dir):
                    for p in os.listdir(pruebas_dir):
                        cand_csvs = os.path.join(pruebas_dir, p, "csvs")
                        procesar_carpeta_csvs(cand_csvs, item, p)
                
                # Buscar directamente en data/raw/<Incendio>/csvs/
                csvs_dir = os.path.join(item_path, "csvs")
                procesar_carpeta_csvs(csvs_dir, item, "Base")

    # 2. Respaldo: Buscar en la ubicación original 'e:/Procesamiento Imagenes/Incendios/'
    fallback_base = r"e:\Procesamiento Imagenes\Incendios"
    if os.path.exists(fallback_base):
        fallback_map = {
            "LOS_ALERCES": ("Los Alerces (Chubut)", r"Los Alerces (Chubut)\Pruebas\2026-01-11 Los Alerces (Chubut) Topsar\csvs", "2026-01-11 Topsar"),
            "SAN_JUAN": ("San Juan (Misiones)", r"San_Juan_(Misiones)\Pruebas\2025-10-29 San_Juan_(Misiones) Topsar\csvs", "2025-10-29 Topsar"),
            "PONTON": ("Pontón (Corrientes)", r"Ponton\Pruebas\2026-01-12 Ponton Topsar\csvs", "2026-01-12 Topsar")
        }
        for key_name, (nombre_lindo, rel_path, p_name) in fallback_map.items():
            tiene_alguna = any(k.startswith(key_name) for k in incendios.keys())
            if not tiene_alguna:
                full_path = os.path.join(fallback_base, rel_path)
                procesar_carpeta_csvs(full_path, nombre_lindo, p_name)

    return incendios


def validar_y_organizar_poligonos(archivos_b, archivos_nb, nombre_sitio):
    """
    Valida y organiza los polígonos de un incendio aplicando la regla dinámica:
    - Se requiere un mínimo de 6 polígonos (3 Burned + 3 Not Burned).
    - Se ordena cada clase por cantidad de píxeles (filas CSV).
    - El menor polígono de Burned y el menor de Not Burned se reservan para Prueba Espacial (Holdout).
    - Todos los demás polígonos (los de mayor tamaño, >= 4) se asignan a Entrenamiento/Validación.
    
    Retorna:
        tuple: (poligonos_train_val, poligonos_holdout, resumen_dict)
    """
    if len(archivos_b) < 3 or len(archivos_nb) < 3:
        raise ValueError(
            f"El incendio '{nombre_sitio}' contiene {len(archivos_b)} polígonos Burned y "
            f"{len(archivos_nb)} Not Burned. Se requiere un mínimo de 3 por clase (6 total) "
            f"para garantizar la partición espacial con StratifiedGroupKFold."
        )
    
    counts_b = [(f, len(pd.read_csv(f))) for f in archivos_b]
    counts_nb = [(f, len(pd.read_csv(f))) for f in archivos_nb]
    
    # Ordenar de mayor a menor número de píxeles
    counts_b.sort(key=lambda x: x[1], reverse=True)
    counts_nb.sort(key=lambda x: x[1], reverse=True)
    
    # El menor de cada clase va a Holdout / Test Espacial
    holdout_b = counts_b[-1]
    holdout_nb = counts_nb[-1]
    
    # Todos los demás (los mayores) van a Train / Val
    train_val_b = counts_b[:-1]
    train_val_nb = counts_nb[:-1]
    
    poligonos_tv = {}
    for idx, (filepath, count) in enumerate(train_val_b, 1):
        poligonos_tv[f"{nombre_sitio}_Q{idx}_Burned"] = filepath
    for idx, (filepath, count) in enumerate(train_val_nb, 1):
        poligonos_tv[f"{nombre_sitio}_NQ{idx}_NotBurned"] = filepath
        
    poligonos_h = {
        f"{nombre_sitio}_Q_Menor_Burned": holdout_b[0],
        f"{nombre_sitio}_NQ_Menor_NotBurned": holdout_nb[0]
    }
    
    total_px_tv = sum(c[1] for c in train_val_b) + sum(c[1] for c in train_val_nb)
    total_px_h = holdout_b[1] + holdout_nb[1]
    
    resumen = {
        "nombre_sitio": nombre_sitio,
        "n_poligonos_tv": len(poligonos_tv),
        "n_poligonos_h": len(poligonos_h),
        "total_px_tv": total_px_tv,
        "total_px_h": total_px_h
    }
    
    return poligonos_tv, poligonos_h, resumen


def crear_interfaz_seleccion_incendios_multiples(incendios_disponibles):
    """
    Despliega una ventana gráfica interactiva Tkinter con dos columnas de checkboxes:
    - Columna Izquierda: Selección de Incendios para Entrenamiento (S_train)
    - Columna Derecha: Selección de Incendios para Evaluación / Prueba Espacial (S_eval)
    
    Retorna:
        tuple: (lista_incendios_train, lista_incendios_test)
    """
    root = tk.Tk()
    root.title("Selección de Incendios para Entrenamiento y Evaluación (Step 4)")
    root.geometry("780x520")
    root.configure(bg="#f4f6f9")
    
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background="#f4f6f9")
    style.configure("TLabel", background="#f4f6f9", font=("Helvetica", 10))
    style.configure("Header.TLabel", font=("Helvetica", 13, "bold"), foreground="#1e293b")
    style.configure("SubHeader.TLabel", font=("Helvetica", 9, "italic"), foreground="#64748b")
    style.configure("TitleCol.TLabel", font=("Helvetica", 11, "bold"), foreground="#0f172a")

    header_frame = ttk.Frame(root, padding=15)
    header_frame.pack(fill="x")
    
    ttk.Label(
        header_frame, 
        text="Configuracion de Experimentos Multi-Incendio (N -> M)", 
        style="Header.TLabel"
    ).pack(anchor="w")
    
    ttk.Label(
        header_frame, 
        text="Selecciona libremente que incendios alimentaran el Entrenamiento y cuales se usaran para la Evaluacion.\n"
             "Regla de Poligonos: El menor poligono por clase va a Test Espacial; todos los demas van a Entrenamiento.",
        style="SubHeader.TLabel"
    ).pack(anchor="w", pady=(5, 0))

    columns_frame = ttk.Frame(root, padding=15)
    columns_frame.pack(fill="both", expand=True)

    # Frame Izquierdo: Entrenamiento
    frame_train = ttk.LabelFrame(columns_frame, text=" Incendios para ENTRENAMIENTO ", padding=10)
    frame_train.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)
    
    # Frame Derecho: Evaluación
    frame_test = ttk.LabelFrame(columns_frame, text=" Incendios para EVALUACION / TEST ", padding=10)
    frame_test.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)

    columns_frame.grid_columnconfigure(0, weight=1)
    columns_frame.grid_columnconfigure(1, weight=1)

    vars_train = {}
    vars_test = {}

    for key, info in incendios_disponibles.items():
        v_train = tk.BooleanVar(value=True)
        v_test = tk.BooleanVar(value=True)
        vars_train[key] = v_train
        vars_test[key] = v_test
        
        lbl_text = f"{info['nombre_lindo']} ({len(info['archivos_burned'])}B / {len(info['archivos_not_burned'])}NB)"
        
        cb_tr = ttk.Checkbutton(frame_train, text=lbl_text, variable=v_train)
        cb_tr.pack(anchor="w", pady=4)
        
        cb_te = ttk.Checkbutton(frame_test, text=lbl_text, variable=v_test)
        cb_te.pack(anchor="w", pady=4)

    resultado = {"train": [], "test": [], "nombre_prueba": "Prueba_MultiIncendio_Tesis"}

    name_frame = ttk.Frame(root, padding=(15, 5))
    name_frame.pack(fill="x")
    
    ttk.Label(
        name_frame, 
        text="Nombre Identificador de la Prueba (Guardado en results/models/):", 
        font=("Helvetica", 10, "bold")
    ).pack(anchor="w")
    
    var_nombre_prueba = tk.StringVar(value="Prueba_MultiIncendio_Tesis")
    entry_nombre_prueba = ttk.Entry(name_frame, textvariable=var_nombre_prueba, font=("Helvetica", 10))
    entry_nombre_prueba.pack(anchor="w", pady=(2, 0), fill="x")

    def on_confirmar():
        sel_train = [k for k, var in vars_train.items() if var.get()]
        sel_test = [k for k, var in vars_test.items() if var.get()]
        nom_prueba = var_nombre_prueba.get().strip()
        
        if not sel_train:
            messagebox.showwarning("Atención", "Debes seleccionar al menos 1 incendio para entrenamiento.")
            return
        if not sel_test:
            messagebox.showwarning("Atención", "Debes seleccionar al menos 1 incendio para evaluación.")
            return
        if not nom_prueba:
            messagebox.showwarning("Atención", "Debes ingresar un nombre identificador para la prueba.")
            return
            
        resultado["train"] = sel_train
        resultado["test"] = sel_test
        resultado["nombre_prueba"] = nom_prueba
        root.destroy()

    btn_frame = ttk.Frame(root, padding=15)
    btn_frame.pack(fill="x")
    
    btn_confirmar = tk.Button(
        btn_frame, 
        text="Confirmar Seleccion y Continuar ->", 
        font=("Helvetica", 10, "bold"), 
        bg="#2563eb", 
        fg="white", 
        activebackground="#1d4ed8", 
        activeforeground="white", 
        relief="flat", 
        padx=15, 
        pady=8,
        command=on_confirmar
    )
    btn_confirmar.pack(side="right")

    root.mainloop()
    return resultado["train"], resultado["test"], resultado["nombre_prueba"]
