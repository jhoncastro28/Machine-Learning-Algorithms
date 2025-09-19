# ☕ Análisis de Machine Learning - Ingresos de Cafeterías

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis completo de Machine Learning para predecir los ingresos diarios de cafeterías utilizando múltiples algoritmos. Desarrollado como parte del curso de Inteligencia Computacional de la Universidad Pedagógica y Tecnológica de Colombia.

## 🎯 Objetivos

- Implementar y comparar 5 algoritmos de Machine Learning diferentes
- Analizar la efectividad de cada modelo en la predicción de ingresos
- Generar visualizaciones claras y reportes detallados
- Proporcionar una interfaz interactiva para el análisis
- Garantizar reproducibilidad completa de los resultados

## 🤖 Algoritmos Implementados

### 1. **Regresión Lineal**
- **Tipo**: Modelo lineal para regresión
- **Hiperparámetros**: `fit_intercept`, `normalize`
- **Ventajas**: Rápido, interpretable, sin hiperparámetros complejos

### 2. **Máquinas de Vector de Soporte (SVM)**
- **Tipo**: Algoritmo de aprendizaje supervisado
- **Hiperparámetros buscados**:
  - `C`: [0.1, 1, 10, 100] - Parámetro de regularización
  - `gamma`: ["scale", "auto", 0.001, 0.01, 0.1, 1] - Coeficiente del kernel
  - `kernel`: ["rbf", "linear", "poly"] - Tipo de kernel
- **Ventajas**: Efectivo en espacios de alta dimensión

### 3. **Árboles de Decisión**
- **Tipo**: Modelo de árbol para regresión
- **Hiperparámetros buscados**:
  - `max_depth`: [3, 5, 10, 15, 20] - Profundidad máxima
  - `min_samples_split`: [2, 5, 10, 20] - Mínimo de muestras para dividir
  - `min_samples_leaf`: [1, 2, 4, 8] - Mínimo de muestras por hoja
- **Ventajas**: Interpretable, maneja datos no lineales

### 4. **Random Forest**
- **Tipo**: Ensemble de árboles de decisión
- **Hiperparámetros buscados**:
  - `n_estimators`: [50, 100, 200, 300] - Número de árboles
  - `max_depth`: [3, 5, 10, 15] - Profundidad máxima
  - `min_samples_split`: [2, 5, 10] - Mínimo de muestras para dividir
  - `min_samples_leaf`: [1, 2, 4] - Mínimo de muestras por hoja
- **Ventajas**: Robusto, reduce overfitting

### 5. **Redes Neuronales Artificiales (MLP)**
- **Tipo**: Perceptrón multicapa
- **Hiperparámetros buscados**:
  - `hidden_layer_sizes`: [[50], [100], [50, 50], [100, 50]] - Arquitectura de capas
  - `activation`: ["relu", "tanh"] - Función de activación
  - `solver`: ["adam", "lbfgs"] - Algoritmo de optimización
  - `alpha`: [0.0001, 0.001, 0.01] - Parámetro de regularización
- **Ventajas**: Aprende patrones complejos no lineales

## 📊 Dataset

- **Fuente**: Kaggle - Coffee Shop Daily Revenue Prediction Dataset
- **Variables**: 6 características de entrada + 1 variable objetivo
- **Tamaño**: 2,000 registros
- **Objetivo**: Predecir ingresos diarios de cafeterías
- **Hash SHA-256**: `65cbae03b79e3896c6a61e4cc76b228d7468a8f2169182d34d4f6c9e93b58a2b`

### Características del Dataset:
- `Number_of_Customers_Per_Day`: Número de clientes por día
- `Average_Order_Value`: Valor promedio de orden
- `Operating_Hours_Per_Day`: Horas de operación por día
- `Number_of_Employees`: Número de empleados
- `Marketing_Spend_Per_Day`: Gasto en marketing por día
- `Location_Foot_Traffic`: Tráfico peatonal de la ubicación
- `Daily_Revenue`: Ingresos diarios (variable objetivo)

## 🚀 Instalación y Uso

### Requisitos Previos
```bash
pip install -r requirements.txt
```

### Modos de Ejecución

#### 🖥️ **Modo GUI (Interfaz Gráfica) - Recomendado**
```bash
python main.py
```
- Interfaz visual intuitiva
- Procesamiento paso a paso
- Visualizaciones integradas
- Ideal para exploración interactiva

#### ⚡ **Modo Batch (Línea de Comandos) - Para Producción**
```bash
# Ejecutar con configuración por defecto
python run_pipeline.py

# Ejecutar con configuración personalizada
python run_pipeline.py mi_config.json
```
- Ejecución completa automatizada
- Genera todos los reportes automáticamente
- Ideal para análisis repetitivos
- Genera metadatos de reproducibilidad

## 📁 Estructura de Carpetas de Salida

```
reports/                          # Directorio principal de reportes
├── run_metadata.json            # Metadatos de ejecución y reproducibilidad
├── tables/                      # Tablas de resultados
│   ├── comparison.csv           # Comparación de métricas por modelo
│   └── predictions.csv          # Predicciones vs valores reales
└── figures/                     # Gráficos y visualizaciones
    ├── eda_distributions.png    # Distribuciones de variables
    ├── corr_heatmap.png         # Mapa de calor de correlaciones
    ├── scatter_pairs.png        # Gráficos de pares de dispersión
    ├── metrics_comparison.png   # Comparación de métricas
    └── predictions_vs_actual.png # Predicciones vs valores reales

models_store/                     # Modelos entrenados
├── linear_regression.pkl        # Modelo de regresión lineal
├── svm.pkl                      # Modelo SVM
├── decision_tree.pkl            # Modelo de árbol de decisión
├── random_forest.pkl            # Modelo Random Forest
├── neural_network.pkl           # Modelo de red neuronal
└── scaler.pkl                   # Escalador de características
```

## ⚙️ Configuración Personalizable

### Ejemplo de `config.json` Editable

```json
{
  "data": {
    "csv_path": "coffee_shop_revenue.csv",
    "target_column": "Daily_Revenue"
  },
  "preprocessing": {
    "test_size": 0.2,
    "random_state": 42,
    "scaler_type": "StandardScaler"
  },
  "models": {
    "linear_regression": {
      "enabled": true,
      "hyperparameters": {
        "fit_intercept": true,
        "normalize": false
      }
    },
    "svm": {
      "enabled": true,
      "hyperparameters": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        "kernel": ["rbf", "linear", "poly"]
      }
    },
    "decision_tree": {
      "enabled": true,
      "hyperparameters": {
        "max_depth": [3, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8]
      }
    },
    "random_forest": {
      "enabled": true,
      "hyperparameters": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
      }
    },
    "neural_network": {
      "enabled": true,
      "hyperparameters": {
        "hidden_layer_sizes": [[50], [100], [50, 50], [100, 50]],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "lbfgs"],
        "alpha": [0.0001, 0.001, 0.01]
      }
    }
  },
  "training": {
    "cv_folds": 5,
    "n_iter": 50,
    "scoring": "neg_mean_squared_error",
    "n_jobs": -1,
    "random_state": 42
  },
  "output": {
    "reports_dir": "reports",
    "tables_dir": "reports/tables",
    "figures_dir": "reports/figures",
    "models_dir": "models_store",
    "save_models": true,
    "save_scaler": true,
    "save_predictions": true,
    "save_comparison": true
  },
  "eda": {
    "generate_plots": true,
    "save_plots": true,
    "plot_types": ["distributions", "correlations", "scatter_pairs"]
  }
}
```

### Parámetros Configurables

- **`test_size`**: Proporción de datos para testing (0.1-0.3 recomendado)
- **`random_state`**: Semilla para reproducibilidad
- **`cv_folds`**: Número de folds para validación cruzada
- **`n_iter`**: Iteraciones para búsqueda aleatoria de hiperparámetros
- **`n_jobs`**: Número de procesos paralelos (-1 = todos los cores)

## 📁 Estructura del Proyecto

```
├── src/                         # Código fuente principal
│   ├── core/                   # Funcionalidades principales
│   │   ├── data_handler.py     # Manejo y preprocesamiento de datos
│   │   └── model_comparator.py # Comparación y evaluación de modelos
│   ├── models/                 # Algoritmos de Machine Learning
│   │   ├── base_model.py       # Clase base para modelos
│   │   ├── linear_regression.py
│   │   ├── svm_model.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── neural_network.py
│   │   └── regression_functions.py # Funciones de entrenamiento
│   ├── gui/                    # Interfaz gráfica
│   │   └── main_window.py      # Ventana principal de la GUI
│   ├── eda/                    # Análisis exploratorio
│   │   └── eda_plots.py        # Generación de gráficos EDA
│   ├── evaluation/             # Evaluación de modelos
│   │   ├── metrics.py          # Cálculo de métricas
│   │   └── plotting.py         # Gráficos de evaluación
│   └── utils/                  # Utilidades y helpers
│       ├── constants.py        # Constantes del sistema
│       ├── helpers.py          # Funciones auxiliares
│       └── metadata.py         # Generación de metadatos
├── cli/                        # Interfaz de línea de comandos
│   └── run_batch.py           # Pipeline batch reproducible
├── main.py                     # Lanzador principal (GUI)
├── run_pipeline.py            # Lanzador batch
├── config.json                # Configuración del proyecto
├── coffee_shop_revenue.csv    # Dataset de cafeterías
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación
```

## 🎮 Funcionalidades

### Modo GUI
1. **Explorar y visualizar datos** - Análisis exploratorio completo
2. **Preparar datos** - Preprocesamiento y división train/test
3. **Entrenar modelos** - Entrenamiento de los 5 algoritmos
4. **Comparar modelos** - Evaluación y comparación de rendimiento
5. **Visualizar resultados** - Gráficos de comparación y análisis
6. **Generar reporte** - Reporte completo de resultados
7. **Guardar resultados** - Exportar resultados a CSV
8. **Análisis completo** - Ejecución automática de todo el proceso

### Modo Batch
- **Ejecución completa automatizada** - Todo el pipeline en una sola ejecución
- **Generación automática de reportes** - Todos los archivos de salida
- **Metadatos de reproducibilidad** - Información completa de la ejecución
- **Validación de configuración** - Verificación automática de parámetros

## 📊 Métricas de Evaluación

- **MSE** (Mean Squared Error): Error cuadrático medio
- **RMSE** (Root Mean Squared Error): Raíz del error cuadrático medio
- **MAE** (Mean Absolute Error): Error absoluto medio
- **R²** (Coefficient of Determination): Coeficiente de determinación
- **MAPE** (Mean Absolute Percentage Error): Error porcentual absoluto medio

## 🔒 Reproducibilidad

El proyecto garantiza **reproducibilidad completa** mediante:

- **Semillas fijas**: `random_state=42` en todos los componentes
- **Metadatos de ejecución**: Archivo `run_metadata.json` con:
  - Timestamp de ejecución
  - Versiones de todas las librerías
  - Configuración completa utilizada
  - Hash SHA-256 del dataset
  - Información del sistema
- **Validación automática**: Verificación de configuración de reproducibilidad
- **Semillas globales**: Establecimiento de semillas de numpy y Python

## 🏆 Resultados Esperados

El análisis proporciona:
- Tabla comparativa de métricas por modelo
- Identificación del mejor modelo según diferentes criterios
- Visualizaciones detalladas de rendimiento
- Reporte completo con recomendaciones
- Archivos CSV con resultados exportables
- Metadatos completos para reproducibilidad

## 🎓 Información Académica

**Universidad Pedagógica y Tecnológica de Colombia**  
**Facultad**: Ingeniería - Escuela de Ingeniería de Sistemas y Computación  
**Materia**: Inteligencia Computacional  

## 👥 Uso Académico

Este proyecto está diseñado para:
- Demostrar la implementación práctica de algoritmos ML
- Comparar diferentes enfoques de aprendizaje automático
- Generar reportes profesionales con visualizaciones
- Proporcionar una base para análisis similares
- Enseñar buenas prácticas de reproducibilidad en ML

## 📝 Notas Técnicas

- **Optimización**: Se utiliza RandomizedSearchCV para optimizar hiperparámetros
- **Validación**: Validación cruzada de 5 folds
- **Preprocesamiento**: Estandarización de características con StandardScaler
- **Reproducibilidad**: Semillas aleatorias fijas para resultados consistentes
- **Paralelización**: Uso de todos los cores disponibles (`n_jobs=-1`)
- **Caché**: Sistema de caché para modelos y scalers

## 🔧 Personalización

El código está diseñado para ser modular y fácilmente extensible:

### Añadir Nuevos Modelos
1. Crear clase en `src/models/`
2. Implementar función de entrenamiento en `regression_functions.py`
3. Agregar configuración en `config.json`
4. Actualizar pipeline en `run_batch.py`

### Modificar Métricas
- Editar `src/evaluation/metrics.py`
- Actualizar funciones de cálculo en `data_handler.py`

### Personalizar Visualizaciones
- Modificar `src/eda/eda_plots.py` para gráficos EDA
- Editar `src/evaluation/plotting.py` para gráficos de evaluación

### Adaptar Configuración
- Modificar `config.json` para cambiar hiperparámetros
- Ajustar `src/utils/constants.py` para valores por defecto

## 🚀 Comandos Rápidos

```bash
# Instalación
pip install -r requirements.txt

# Modo GUI
python main.py

# Modo Batch
python run_pipeline.py

# Con configuración personalizada
python run_pipeline.py mi_config.json

# Verificar metadatos
cat reports/run_metadata.json
```

---

*Desarrollado con ❤️ para el aprendizaje de Machine Learning*