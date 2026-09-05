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
│   └── data/                           # Modulos de procesamiento de datos e indices SAR
│       ├── alignment.py                # Alineacion espacial e interseccion de coberturas GeoTIFF
│       ├── process_matrix.py           # Conversion a matrices .npz y validacion dimensional
│       ├── sar_indices.py              # Calculo de indices espectrales (IRV, NDPI, NDBI, RBR, Normalized)
│       └── sampling.py                 # Muestreo de poligonos, trazabilidad JSON y exportacion CSV
├── scripts/                            # Scripts de ejecucion secuencial
│   └── 01_generate_dataset.py          # Script principal para la generacion del dataset CSV (Fase 1)
├── docs/                               # Documentacion tecnica detallada por fases
│   └── DATA_PROCESSING.md              # Especificacion tecnica completa del pipeline de datos (Fase 1)
└── data/                               # Estructura de almacenamiento de datos
    └── raw/
        └── Ponton/                     # Dataset de muestra GeoTIFF (Pre y Post incendio)
```

---

## Fases del Proyecto y Documentacion Tecnica

### Fase 1: Procesamiento de Imagenes e Indices SAR
Esta fase abarca la alineacion espacial de imagenes GeoTIFF Pre y Post incendio, la conversion a matrices numericas, el calculo de 10 indices espectrales de radar, el muestreo de Regiones de Interes (ROIs) y la generacion del dataset tabulado de 21 columnas.

- **Documentacion detallada de la Fase 1**: [docs/DATA_PROCESSING.md](docs/DATA_PROCESSING.md)
- **Script de ejecucion**: `scripts/01_generate_dataset.py`

---

## Instalacion y Requisitos

### Requisitos de Software
- Python 3.10 o superior
- Dependencias indicadas en `requirements.txt` (`rasterio`, `numpy`, `opencv-python`, `pandas`, `pyproj`, `matplotlib`, `scikit-learn`)

### Instalacion del Entorno

```bash
git clone https://github.com/usuario/tesis-sar-incendios.git
cd tesis-sar-incendios
pip install -r requirements.txt
```

---

## Ejemplo de Ejecucion (Fase 1)

Para ejecutar el pipeline completo de procesamiento de datos sobre el dataset de muestra incluido en `data/raw/Ponton/`:

```bash
python scripts/01_generate_dataset.py
```

---

## Licencia

Este proyecto esta licenciado bajo la Licencia MIT. Para mas detalles, consulte el archivo [LICENSE](LICENSE).
