# Procesamiento de Imagenes SAR y Aprendizaje Automatico para la Clasificacion de Areas Quemadas

Este repositorio contiene el codigo fuente modular, los scripts de procesamiento y la arquitectura metodologica desarrollada para el procesamiento de imagenes satelitales de Radar de Apertura Sintetica (SAR) y la clasificacion de areas quemadas en incendios forestales mediante algoritmos de Aprendizaje Automatico.

---

## Estructura del Repositorio

```text
tesis-sar-incendios/
├── README.md                           # Guia principal de arquitectura y uso del repositorio
├── LICENSE                             # Licencia MIT de codigo abierto
├── requirements.txt                    # Dependencias y librerias de Python requeridas
├── .gitignore                          # Exclusion de archivos pesados y temporales
├── src/                                # Paquete modular de Python
│   ├── data/                           # Modulos de procesamiento de datos e indices SAR (Fase 1)
│   │   ├── alignment.py                # Alineacion espacial e interseccion de coberturas GeoTIFF
│   │   ├── process_matrix.py           # Conversion a matrices .npz y validacion dimensional
│   │   ├── sar_indices.py              # Calculo de indices espectrales (IRV, NDPI, NDBI, RBR, Normalized)
│   │   └── sampling.py                 # Muestreo de poligonos, trazabilidad JSON y exportacion CSV
│   ├── models/                         # Modulos de seleccion de caracteristicas, clasificadores e inferencia (Fase 2 y 3)
│   │   ├── feature_selection.py        # Seleccion interactiva y filtrado de caracteristicas espectrales
│   │   ├── model_factory.py            # Registro unificado, entrenamiento y guardado de modelos
│   │   └── inference.py                # Inferencia masiva en batch sobre escenas satelitales completas
│   ├── evaluation/                     # Modulos de validacion espacial y evaluacion (Fase 2)
│   │   ├── cross_validation.py         # Asignacion por tamano de poligono, StratifiedGroupKFold y escalado
│   │   └── metrics.py                  # Calculo de metricas (Accuracy, F1, Precision, Recall, AUC, Specificity)
│   └── visualization/                  # Modulos de visualizacion cartografica y georreferenciacion (Fase 2 y 3)
│       ├── geo_utils.py                # Reproyeccion WGS84 y formateo academico de coordenadas
│       └── map_generator.py            # Generador de mapas PNG georreferenciados binarios y de probabilidad
├── scripts/                            # Scripts de ejecucion secuencial
│   ├── 01_generate_dataset.py          # Script principal para la generacion del dataset CSV (Fase 1)
│   ├── 02_train_and_evaluate.py        # Script orquestador de entrenamiento y evaluacion espacial (Fase 2)
│   └── 03_generate_classification_maps.py # Script orquestador de inferencia masiva y generacion de mapas PNG (Fase 3)
├── docs/                               # Documentacion tecnica detallada por fases
│   ├── DATA_PROCESSING.md              # Especificacion tecnica del pipeline de datos (Fase 1)
│   ├── MODEL_TRAINING.md               # Especificacion tecnica del entrenamiento y evaluacion (Fase 2)
│   ├── MAP_GENERATION.md               # Especificacion tecnica de la inferencia y mapas cartograficos (Fase 3)
│   └── images/                         # Capturas de pantalla de interfaces graficas y esquemas
└── data/                               # Estructura de almacenamiento de datos
    └── raw/
        └── Ponton/                     # Dataset de muestra GeoTIFF y CSVs de poligonos
```

---

## Fases del Proyecto y Documentacion Tecnica

### Fase 1: Procesamiento de Imagenes e Indices SAR
Esta fase abarca la alineacion espacial de imagenes GeoTIFF Pre y Post incendio, la conversion a matrices numericas, el calculo de 10 indices espectrales de radar, el muestreo de Regiones de Interes (ROIs) y la generacion del dataset tabulado de 21 columnas.

- **Documentacion detallada de la Fase 1**: [docs/DATA_PROCESSING.md](docs/DATA_PROCESSING.md)
- **Script de ejecucion**: `scripts/01_generate_dataset.py`

### Fase 2: Entrenamiento y Evaluacion de Modelos ML
Esta fase abarca la seleccion interactiva de caracteristicas espectrales, la particion espacial estricta por poligonos (**StratifiedGroupKFold**), el entrenamiento de 16 clasificadores de Machine Learning (SVM, Logistic Regression, Arboles de Decision, Ensembles de Boosting y Redes Neuronales MLP) y la evaluacion cuantitativa final sobre un **Conjunto de Prueba Espacialmente Aislado**.

- **Documentacion detallada de la Fase 2**: [docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md)
- **Script de ejecucion**: `scripts/02_train_and_evaluate.py`

### Fase 3: Inferencia Espacial y Generacion de Mapas Cartograficos
Esta fase abarca la inferencia masiva en batch sobre la escena satelital completa (`complete_image.csv`), la georreferenciacion cartografica en coordenadas WGS84 ($\text{Latitud}/\text{Longitud}$) y la exportacion de mapas de alta resolucion en formato PNG (clasificacion binaria Rojo/Verde y mapas de intensidad de probabilidad continua).

- **Documentacion detallada de la Fase 3**: [docs/MAP_GENERATION.md](docs/MAP_GENERATION.md)
- **Script de ejecucion**: `scripts/03_generate_classification_maps.py`

---

## Instalacion y Requisitos

### Requisitos de Software
- Python 3.10 o superior
- Dependencias indicadas en `requirements.txt` (`rasterio`, `numpy`, `opencv-python`, `pandas`, `pyproj`, `matplotlib`, `scikit-learn`)

### Instalacion del Entorno

```bash
git clone https://github.com/heguzmanUNAHUR/image-SAR-processing.git
cd image-SAR-processing
pip install -r requirements.txt
```

---

## Ejemplos de Ejecucion

### 1. Ejecucion de la Fase 1 (Generacion del Dataset)

```bash
# Modo interactivo (prompts en consola para nombre de prueba y poligonos)
python scripts/01_generate_dataset.py

# Modo CLI parametrizado
python scripts/01_generate_dataset.py --prueba "2026-09-05_Ponton_Exp01" --muestras 3
```

### 2. Ejecucion de la Fase 2 (Entrenamiento y Evaluacion de Modelos)

```bash
# Modo interactivo (GUI Tkinter para seleccion de indices y modelos)
python scripts/02_train_and_evaluate.py --incendio PONTON

# Modo no interactivo (Headless)
python scripts/02_train_and_evaluate.py --incendio PONTON --headless
```

### 3. Ejecucion de la Fase 3 (Inferencia Espacial y Mapas Cartograficos)

```bash
# Modo interactivo (seleccion interactiva de prueba y modelos)
python scripts/03_generate_classification_maps.py --incendio PONTON

# Modo no interactivo (Headless)
python scripts/03_generate_classification_maps.py --incendio PONTON --prueba "Prueba Ponton GKF Spatial Holdout (Tesis)" --headless
```

---

## Licencia

Este proyecto esta licenciado bajo la Licencia MIT. Para mas detalles, consulte el archivo [LICENSE](LICENSE).
