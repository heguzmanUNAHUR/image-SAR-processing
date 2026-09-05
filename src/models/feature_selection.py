# -*- coding: utf-8 -*-
"""
Modulo de seleccion y filtrado de caracteristicas espectrales SAR.
Preserva el 100% de la logica original de lectura e interfaz de seleccion.
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox


def obtener_columnas_dataset(ruta_archivo):
    """
    Obtiene las columnas disponibles en un archivo CSV de dataset.
    
    Args:
        ruta_archivo (str): Ruta completa al archivo CSV del dataset
        
    Returns:
        list: Lista con los nombres de las columnas disponibles
    """
    try:
        df = pd.read_csv(ruta_archivo, nrows=0)
        columnas_disponibles = df.columns.tolist()
        
        print("[INFO] Columnas disponibles en el dataset:")
        for i, columna in enumerate(columnas_disponibles, 1):
            print(f"  {i:2d}. {columna}")
        
        return columnas_disponibles
        
    except FileNotFoundError:
        print(f"[ERROR] No se encontro el archivo en la ruta: {ruta_archivo}")
        return []
    except Exception as e:
        print(f"[ERROR] Error al leer el archivo: {str(e)}")
        return []


def crear_interfaz_seleccion_indices(ruta_archivo):
    """
    Crea una interfaz grafica Tkinter para seleccionar los indices SAR a evaluar.
    
    Args:
        ruta_archivo (str): Ruta al archivo CSV del dataset
        
    Returns:
        list: Lista con los indices seleccionados por el usuario
    """
    columnas_disponibles = obtener_columnas_dataset(ruta_archivo)
    
    if not columnas_disponibles:
        print("[WARNING] No se pudieron cargar las columnas del dataset")
        return []
    
    try:
        ventana = tk.Tk()
        ventana.title("Seleccion de Indices SAR para Evaluacion")
        ventana.geometry("800x600")
        ventana.resizable(True, True)
    except Exception as e:
        print(f"[WARNING] Entorno sin pantalla detectado ({e}). Se seleccionan todas las caracteristicas.")
        return [col for col in columnas_disponibles if 'Burn_Classification' not in col and 'Classification' not in col]
    
    indices_seleccionados = []
    
    def confirmar_seleccion():
        for i, var in enumerate(variables_checkbox):
            if var.get():
                indices_seleccionados.append(columnas_filtradas[i])
        
        if not indices_seleccionados:
            messagebox.showwarning("Advertencia", "Debes seleccionar al menos un indice")
            return
        
        ventana.destroy()
    
    def toggle_todos():
        estado = variables_checkbox[0].get()
        for var in variables_checkbox:
            var.set(not estado)
    
    titulo = tk.Label(ventana, text="Selecciona los indices SAR que deseas evaluar:", 
                     font=("Arial", 14, "bold"))
    titulo.pack(pady=20)
    
    frame_principal = ttk.Frame(ventana)
    frame_principal.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    canvas = tk.Canvas(frame_principal)
    scrollbar = ttk.Scrollbar(frame_principal, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    btn_toggle = tk.Button(scrollable_frame, text="Seleccionar/Deseleccionar Todo", 
                          command=toggle_todos, bg="#4CAF50", fg="white", 
                          font=("Arial", 10, "bold"))
    btn_toggle.pack(pady=(0, 20))
    
    frame_columnas = tk.Frame(scrollable_frame)
    frame_columnas.pack(fill=tk.X, expand=True)
    
    columna_izquierda = tk.Frame(frame_columnas)
    columna_derecha = tk.Frame(frame_columnas)
    columna_izquierda.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    columna_derecha.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    variables_checkbox = []
    columnas_filtradas = [col for col in columnas_disponibles 
                         if 'Burn_Classification' not in col and 'Classification' not in col]
    
    mitad = len(columnas_filtradas) // 2
    
    for i, columna in enumerate(columnas_filtradas):
        var = tk.BooleanVar()
        variables_checkbox.append(var)
        
        if i < mitad:
            frame_columna = columna_izquierda
        else:
            frame_columna = columna_derecha
        
        frame_checkbox = tk.Frame(frame_columna)
        frame_checkbox.pack(fill=tk.X, pady=2, padx=5)
        
        checkbox = tk.Checkbutton(frame_checkbox, text=columna, variable=var, 
                                font=("Arial", 10), anchor=tk.W)
        checkbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        separador = ttk.Separator(frame_columna, orient='horizontal')
        separador.pack(fill=tk.X, pady=2)
    
    frame_botones = tk.Frame(ventana)
    frame_botones.pack(pady=20)
    
    btn_confirmar = tk.Button(frame_botones, text="Confirmar Seleccion", 
                             command=confirmar_seleccion, bg="#2196F3", fg="white",
                             font=("Arial", 12, "bold"), width=15)
    btn_confirmar.pack(side=tk.LEFT, padx=10)
    
    btn_cancelar = tk.Button(frame_botones, text="Cancelar", 
                            command=ventana.destroy, bg="#f44336", fg="white",
                            font=("Arial", 12, "bold"), width=15)
    btn_cancelar.pack(side=tk.LEFT, padx=10)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    ventana.update_idletasks()
    x = (ventana.winfo_screenwidth() // 2) - (ventana.winfo_width() // 2)
    y = (ventana.winfo_screenheight() // 2) - (ventana.winfo_height() // 2)
    ventana.geometry(f"+{x}+{y}")
    
    ventana.mainloop()
    
    return indices_seleccionados
