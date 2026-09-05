# -*- coding: utf-8 -*-
"""
Modulo de visualizacion de resultados de modelos ML.
Genera matrices de confusion con porcentajes y graficos de importancia de caracteristicas.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_test, y_pred, nombre_modelo, carpeta_destino):
    """
    Genera y guarda el grafico de la matriz de confusion con porcentajes y recuentos absolutos.
    
    Args:
        y_test: Etiquetas reales
        y_pred: Predicciones del modelo
        nombre_modelo (str): Nombre identificador del modelo
        carpeta_destino (str): Carpeta donde se guardara la imagen PNG
    """
    try:
        class_labels = ["ANQ", "AQ"]
        cm = confusion_matrix(y_test, y_pred)
        
        # Porcentajes por fila (clase real)
        sum_rows = cm.sum(axis=1, keepdims=True)
        cm_percent = np.zeros_like(cm, dtype=float)
        np.divide(cm.astype(float), sum_rows, out=cm_percent, where=sum_rows != 0)
        cm_percent *= 100
        
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('(%)', fontsize=11)
        
        n_classes = len(class_labels)
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_labels, fontsize=12)
        ax.set_yticklabels(class_labels, fontsize=12)
        ax.set_xlabel('Clase predicha', fontsize=12, fontweight='bold')
        ax.set_ylabel('Clase real', fontsize=12, fontweight='bold')
        
        thresh = 50.0
        for i in range(n_classes):
            for j in range(n_classes):
                pct = cm_percent[i, j]
                abs_val = cm[i, j]
                text_color = "white" if pct > thresh else "black"
                ax.text(j, i - 0.1, f"{pct:.1f}%",
                        ha="center", va="center",
                        fontsize=16, fontweight='bold', color=text_color)
                ax.text(j, i + 0.22, f"(n = {abs_val:,})",
                        ha="center", va="center",
                        fontsize=9, color=text_color, alpha=0.85)
        
        ax.set_title(f'Matriz de Confusion - {nombre_modelo}', fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()
        
        if not os.path.exists(carpeta_destino):
            os.makedirs(carpeta_destino, exist_ok=True)
            
        nombre_safe = nombre_modelo.replace(' ', '_').replace('/', '_').replace('\\', '_')
        cm_path = os.path.join(carpeta_destino, f"{nombre_safe}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Matriz de confusion guardada en: {cm_path}")
        
    except Exception as e:
        print(f"[ERROR] Error al generar matriz de confusion para {nombre_modelo}: {str(e)}")


def plot_feature_importances(modelo, feature_names, nombre_modelo, carpeta_destino, top_n=20, X_test=None, y_test=None):
    """
    Genera y guarda un grafico de barras horizontal con las caracteristicas mas importantes.
    Soporta:
    - Modelos de arbol (feature_importances_)
    - Modelos lineales (coef_)
    - Permutation importance para cualquier modelo (requiere X_test, y_test)
    """
    try:
        importances = None
        titulo_metrica = "Importancia"
        usar_permutation = False
        
        if hasattr(modelo, 'feature_importances_'):
            importances = modelo.feature_importances_
            titulo_metrica = "Importancia"
        elif hasattr(modelo, 'coef_'):
            coef = modelo.coef_
            if coef.ndim > 1:
                coef = coef.flatten() if coef.shape[0] == 1 else np.mean(np.abs(coef), axis=0)
            importances = np.abs(coef)
            titulo_metrica = "Importancia (|Coeficiente|)"
        elif X_test is not None and y_test is not None:
            print(f"[INFO] Calculando permutation importance para {nombre_modelo}...")
            perm_importance = permutation_importance(
                modelo, X_test, y_test,
                n_repeats=10,
                random_state=111,
                n_jobs=-1
            )
            importances = perm_importance.importances_mean
            titulo_metrica = "Importancia (Permutation)"
            usar_permutation = True
        else:
            print(f"[WARNING] El modelo {nombre_modelo} no expone caracteristicas de importancia directamente.")
            return

        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f'Feature_{i}' for i in range(len(importances))]

        label_map = {
            'NORMALIZED_VH':   r'$\mathrm{VH_{Norm}}$',
            'NORMALIZED_VV':   r'$\mathrm{VV_{Norm}}$',
            'NORMALIZED_NDBI': r'$\mathrm{NDBI_{Norm}}$',
            'HH_Pre':          r'$\mathrm{HH_{Pre}}$',
            'HV_Pre':          r'$\mathrm{HV_{Pre}}$',
            'VH_Pre':          r'$\mathrm{VH_{Pre}}$',
            'VV_Pre':          r'$\mathrm{VV_{Pre}}$',
            'HH_Post':         r'$\mathrm{HH_{Post}}$',
            'HV_Post':         r'$\mathrm{HV_{Post}}$',
            'VH_Post':         r'$\mathrm{VH_{Post}}$',
            'VV_Post':         r'$\mathrm{VV_{Post}}$',
            'NDPI_Pre':        r'$\mathrm{NDPI_{Pre}}$',
            'NDPI_Post':       r'$\mathrm{NDPI_{Post}}$',
            'NDBI_Pre':        r'$\mathrm{NDBI_{Pre}}$',
            'NDBI_Post':       r'$\mathrm{NDBI_{Post}}$',
            'IRV_Pre':         r'$\mathrm{IRV_{Pre}}$',
            'IRV_Post':        r'$\mathrm{IRV_{Post}}$',
            'RBR_VV':          r'$\mathrm{RBR_{VV}}$',
            'RBR_VH':          r'$\mathrm{RBR_{VH}}$',
            'RBR_NDBI':        r'$\mathrm{RBR_{NDBI}}$',
        }
        feature_names_formatted = [label_map.get(name, name) for name in feature_names]

        importances_dict = dict(zip(feature_names_formatted, importances))
        sorted_importances = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)

        top_features = sorted_importances[:top_n]
        feature_names_sorted = [x[0] for x in top_features]
        importances_sorted = [x[1] for x in top_features]

        fig, ax = plt.subplots(figsize=(12, max(8, len(top_features) * 0.5)))
        y_pos = np.arange(len(feature_names_sorted))
        bars = ax.barh(y_pos, importances_sorted, align='center', color='steelblue', alpha=0.8)

        ax.set_yticks(y_pos)
        feature_names_display = [name[:30] + '...' if len(name) > 30 else name for name in feature_names_sorted]
        ax.set_yticklabels(feature_names_display, fontsize=9)
        ax.invert_yaxis()

        ax.set_xlabel(titulo_metrica, fontsize=12, fontweight='bold')
        ax.set_ylabel('Caracteristicas', fontsize=12, fontweight='bold')

        titulo = f'Top {len(top_features)} Caracteristicas Importantes - {nombre_modelo}'
        if usar_permutation:
            titulo += '\n(Calculado con Permutation Importance)'
        ax.set_title(titulo, fontsize=14, fontweight='bold', pad=20)

        for bar, importance in zip(bars, importances_sorted):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.4f}', 
                   ha='left', va='center', fontsize=9)

        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()

        if not os.path.exists(carpeta_destino):
            os.makedirs(carpeta_destino, exist_ok=True)

        nombre_safe = nombre_modelo.replace(' ', '_').replace('/', '_').replace('\\', '_')
        ruta_guardado = os.path.join(carpeta_destino, f"{nombre_safe}_feature_importances.png")
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Grafico de importancias guardado en: {ruta_guardado}")

    except Exception as e:
        print(f"[ERROR] Error al generar grafico de importancias para {nombre_modelo}: {str(e)}")
