# -*- coding: utf-8 -*-
"""
Modulo de creacion, entrenamiento y almacenamiento de modelos de Machine Learning.
Preserva el 100% de hiperparametros, nombres y configuraciones originales de la tesis.
"""

import os
import time
import pickle
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier

from src.models.visualization import plot_confusion_matrix, plot_feature_importances


def obtener_modelos_disponibles(random_state=111):
    """
    Retorna un diccionario con todas las instancias de modelos habilitados y sus hiperparametros exactos.
    """
    return {
        # SVM
        'SVM_Linear': SVC(kernel='linear', probability=True, random_state=random_state),
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=random_state),
        'SVM_Polynomial': SVC(kernel='poly', degree=3, probability=True, random_state=random_state),
        'SVM_Sigmoid': SVC(kernel='sigmoid', probability=True, random_state=random_state),

        # Logistic Regression
        'Logistics Regression': LogisticRegression(random_state=random_state, max_iter=1000),

        # Arboles y Ensembles
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=random_state),

        # Gradient & Adaptive Boosting
        'Gradient Boosting Balanced': GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            min_samples_split=10, min_samples_leaf=5, subsample=0.8,
            max_features=0.8, random_state=random_state
        ),
        'Gradient Boosting Precision': GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            min_samples_split=5, min_samples_leaf=3, subsample=0.9,
            max_features=0.9, validation_fraction=0.1, n_iter_no_change=15,
            tol=1e-4, random_state=random_state
        ),
        'Gradient Boosting Prevent Overfit': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.075, max_depth=4,
            min_samples_split=15, min_samples_leaf=8, subsample=0.7,
            max_features=0.7, validation_fraction=0.15, n_iter_no_change=10,
            random_state=random_state
        ),
        'Gradient Boosting Fast': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.15, max_depth=3,
            min_samples_split=20, min_samples_leaf=10, subsample=0.6,
            max_features=0.6, random_state=random_state
        ),
        'Adaptive Boosting': AdaBoostClassifier(n_estimators=100, random_state=random_state),

        # Redes Neuronales (MLP)
        'NN-ReLU-2L(200,100)-r2': MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu', max_iter=1000, solver='adam', random_state=random_state),
        'NN-ReLU-2L(300,100)-r2': MLPClassifier(hidden_layer_sizes=(300, 100), activation='relu', max_iter=1000, solver='adam', random_state=random_state),
        'NN-ReLU-2L(300,200)-r2': MLPClassifier(hidden_layer_sizes=(300, 200), activation='relu', max_iter=1000, solver='adam', random_state=random_state),
        'NN-Log-2L(200,100)-r2': MLPClassifier(hidden_layer_sizes=(200, 100), activation='logistic', max_iter=1000, solver='adam', random_state=random_state),
        'NN-Log-2L(300,100)-r2': MLPClassifier(hidden_layer_sizes=(300, 100), activation='logistic', max_iter=1000, solver='adam', random_state=random_state),
        'NN-Log-2L(300,200)-r2': MLPClassifier(hidden_layer_sizes=(300, 200), activation='logistic', max_iter=1000, solver='adam', random_state=random_state),
        'NN-Tanh-2L(300,150)-r2': MLPClassifier(hidden_layer_sizes=(300, 150), activation='tanh', max_iter=1000, solver='adam', random_state=random_state),
        'NN-Tanh-3L(300,150,50)-r2': MLPClassifier(hidden_layer_sizes=(300, 150, 50), activation='tanh', max_iter=1000, solver='adam', random_state=random_state),
        'NN-ReLU-2L(200,100)-r3-alpha1': MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu', max_iter=1000, solver='adam', random_state=random_state, learning_rate_init=0.0001, alpha=0.01),
        'NN-ReLU-3L(200,100,50)-r3': MLPClassifier(hidden_layer_sizes=(200, 100, 50), activation='relu', max_iter=1000, solver='adam', random_state=random_state, learning_rate_init=0.0001),
    }


def crear_interfaz_seleccion_modelos():
    """
    Crea una interfaz grafica Tkinter para la seleccion de modelos.
    En entorno sin pantalla, devuelve un conjunto por defecto de modelos representativos.
    """
    modelos_dict = obtener_modelos_disponibles()
    nombres_modelos = list(modelos_dict.keys())

    try:
        ventana = tk.Tk()
        ventana.title("Seleccion de Modelos de Machine Learning")
        ventana.geometry("800x600")
        ventana.resizable(True, True)
    except Exception as e:
        print(f"[WARNING] Entorno sin pantalla detectado ({e}). Se seleccionan los modelos principales por defecto.")
        return ['SVM_RBF', 'Logistics Regression', 'Random Forest', 'Decision Tree', 'Extra Trees', 'Gradient Boosting Balanced', 'NN-ReLU-2L(200,100)-r2']

    modelos_seleccionados = []

    def confirmar_seleccion():
        for i, var in enumerate(variables_checkbox):
            if var.get():
                modelos_seleccionados.append(nombres_modelos[i])

        if not modelos_seleccionados:
            messagebox.showwarning("Advertencia", "Debes seleccionar al menos un modelo")
            return

        ventana.destroy()

    def toggle_todos():
        estado = variables_checkbox[0].get()
        for var in variables_checkbox:
            var.set(not estado)

    titulo = tk.Label(ventana, text="Selecciona los modelos de ML que deseas evaluar:",
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
    mitad = len(nombres_modelos) // 2

    for i, modelo in enumerate(nombres_modelos):
        var = tk.BooleanVar()
        variables_checkbox.append(var)

        frame_columna = columna_izquierda if i < mitad else columna_derecha

        frame_checkbox = tk.Frame(frame_columna)
        frame_checkbox.pack(fill=tk.X, pady=2, padx=5)

        checkbox = tk.Checkbutton(frame_checkbox, text=modelo, variable=var,
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
    return modelos_seleccionados


def entrenar_modelo(X_train, X_test, y_train, y_test, nombre_modelo, random_state=111):
    """
    Entrena un modelo especifico y retorna un diccionario con el modelo y sus predicciones.
    """
    modelos = obtener_modelos_disponibles(random_state=random_state)
    if nombre_modelo not in modelos:
        print(f"[WARNING] Modelo '{nombre_modelo}' no esta en el registro de modelos.")
        return None

    modelo = modelos[nombre_modelo]
    print(f"[INFO] Entrenando {nombre_modelo}...")

    tiempo_inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - tiempo_inicio

    y_pred = modelo.predict(X_test)

    y_prob = None
    if hasattr(modelo, 'predict_proba'):
        try:
            if len(np.unique(y_test)) == 2:
                y_prob = modelo.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None

    print(f"[INFO] {nombre_modelo} entrenado exitosamente en {tiempo_entrenamiento:.2f} s")

    return {
        'modelo': modelo,
        'nombre': nombre_modelo,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'X_test': X_test,
        'time_train': tiempo_entrenamiento
    }


def guardar_modelo_y_artefactos(resultado, scaler, carpeta_destino, feature_names=None):
    """
    Guarda el archivo PKL del modelo, el scaler y genera los graficos de confusion e importancia.
    """
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino, exist_ok=True)

    nombre = resultado['nombre']
    nombre_safe = nombre.replace(' ', '_').replace('/', '_').replace('\\', '_')

    # Guardar modelo
    modelo_path = os.path.join(carpeta_destino, f"{nombre_safe}_modelo.pkl")
    with open(modelo_path, 'wb') as f:
        pickle.dump(resultado['modelo'], f)
    print(f"[INFO] Modelo {nombre} guardado en: {modelo_path}")

    # Guardar scaler
    scaler_path = os.path.join(carpeta_destino, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Guardar predicciones basicas
    predicciones_path = os.path.join(carpeta_destino, "predicciones_basicas.pkl")
    with open(predicciones_path, 'wb') as f:
        pickle.dump({
            'y_test': resultado['y_test'],
            'y_pred': resultado['y_pred'],
            'time_train': resultado['time_train']
        }, f)

    # Generar graficos
    plot_confusion_matrix(resultado['y_test'], resultado['y_pred'], nombre, carpeta_destino)
    plot_feature_importances(
        resultado['modelo'],
        feature_names,
        nombre,
        carpeta_destino,
        X_test=resultado['X_test'],
        y_test=resultado['y_test']
    )
