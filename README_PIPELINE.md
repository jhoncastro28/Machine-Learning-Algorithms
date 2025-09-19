# Pipeline Batch Reproducible - Machine Learning

## Descripción

Pipeline batch reproducible para análisis de Machine Learning de ingresos de cafeterías. Este sistema ejecuta un flujo completo sin interfaz gráfica, desde la carga de datos hasta la generación de reportes.

## Características

- ✅ **Carga y exploración de datos** con análisis estadístico completo
- ✅ **EDA automatizado** con generación de gráficos PNG
- ✅ **Preprocesamiento** con escalado y guardado de scaler
- ✅ **Entrenamiento de múltiples modelos** con búsqueda de hiperparámetros
- ✅ **Evaluación comparativa** con métricas completas
- ✅ **Generación de reportes** en CSV y gráficos
- ✅ **Configuración flexible** via JSON
- ✅ **Reproducibilidad** con semillas aleatorias

## Estructura de Archivos

```
├── cli/
│   ├── __init__.py
│   └── run_batch.py          # Pipeline principal
├── config.json               # Configuración del pipeline
├── run_pipeline.py           # Script de ejecución
├── reports/
│   ├── tables/
│   │   ├── comparison.csv    # Tabla de comparación de modelos
│   │   └── predictions.csv   # Predicciones de todos los modelos
│   └── figures/
│       ├── eda_distributions.png
│       ├── corr_heatmap.png
│       ├── scatter_pairs.png
│       ├── metrics_comparison.png
│       └── predictions_vs_actual.png
└── models_store/
    ├── scaler.pkl
    ├── linear_regression.pkl
    ├── svm.pkl
    ├── decision_tree.pkl
    ├── random_forest.pkl
    └── neural_network.pkl
```

## Uso

### Ejecución Básica
```bash
python run_pipeline.py
```

### Ejecución con Configuración Personalizada
```bash
python run_pipeline.py mi_config.json
```

### Ejecución Directa del Pipeline
```python
from cli.run_batch import run_batch
success = run_batch("config.json")
```

## Configuración

El archivo `config.json` permite configurar:

- **Datos**: ruta del CSV, columna objetivo
- **Preprocesamiento**: test_size, random_state, tipo de scaler
- **Modelos**: habilitar/deshabilitar modelos y sus hiperparámetros
- **Entrenamiento**: CV folds, iteraciones, métricas
- **Salida**: directorios, qué guardar

## Modelos Incluidos

1. **Regresión Lineal** - Modelo base
2. **SVM** - Máquinas de Vector de Soporte
3. **Árbol de Decisión** - Modelo interpretable
4. **Random Forest** - Ensemble robusto
5. **Red Neuronal** - MLP Regressor

## Métricas Evaluadas

- **MSE** - Error Cuadrático Medio
- **RMSE** - Raíz del Error Cuadrático Medio
- **MAE** - Error Absoluto Medio
- **R²** - Coeficiente de Determinación
- **MAPE** - Error Porcentual Absoluto Medio

## Resultados del Último Ejecución

| Modelo | MSE | RMSE | MAE | R² | MAPE (%) |
|--------|-----|------|-----|----|---------| 
| Regresión Lineal | 97,569.72 | 312.36 | 244.21 | 0.8956 | 19.42 |
| SVM | 56,166.27 | 236.99 | 184.31 | 0.9399 | 14.45 |
| Árbol de Decisión | 70,540.05 | 265.59 | 213.63 | 0.9245 | 15.86 |
| **Random Forest** | **48,778.10** | **220.86** | 177.72 | **0.9478** | **13.63** |
| Red Neuronal | 50,381.81 | 224.46 | **176.54** | 0.9461 | 13.83 |

**Mejor modelo general**: Random Forest (mejor R² y menor MAPE)

## Requisitos

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.1.0
- joblib

## Instalación

```bash
pip install -r requirements.txt
```

## Reproducibilidad

El pipeline garantiza reproducibilidad mediante:
- Semillas aleatorias configurables
- Guardado de scalers y modelos
- Configuración versionada
- Logs detallados de ejecución
