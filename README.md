# ☕ Análisis de Machine Learning - Ingresos de Cafeterías

**Universidad Pedagógica y Tecnológica de Colombia**  
**Facultad de Ingeniería - Escuela de Ingeniería de Sistemas y Computación**  
**Inteligencia Computacional**

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis completo de Machine Learning para **clasificar los ingresos diarios de cafeterías** en categorías (Bajo, Medio, Alto) utilizando múltiples algoritmos de clasificación.

## 👥 Desarrolladores

Creado por estudiantes de **Ingeniería de Sistemas y Computación** de la **UPTC**:

- **Jhon Castro Mancipe**
- **Juan Sebastián Zárate**
- **Juan David Carrillo**

## 🎯 Objetivos

- ✅ **Comparar 5 algoritmos de clasificación** diferentes para categorizar ingresos
- ✅ **Identificar el mejor modelo** según diferentes métricas de clasificación
- ✅ **Generar reportes profesionales** con visualizaciones de alta calidad
- ✅ **Proporcionar dos modos de uso**: GUI interactiva y batch automatizado
- ✅ **Garantizar reproducibilidad completa** con metadatos detallados
- ✅ **Crear un sistema modular** fácil de extender y personalizar

## 🤖 Algoritmos Implementados

### 1. **Regresión Logística** ⭐ **PRINCIPAL**
- **Tipo**: Modelo lineal para clasificación
- **Hiperparámetros**: `C`, `penalty`, `solver`, `max_iter`, `class_weight`
- **Ventajas**: Rápido, interpretable, excelente para clasificación multiclase

### 2. **Máquinas de Vector de Soporte (SVM)** 🏆 **MEJOR MODELO**
- **Tipo**: Algoritmo de clasificación supervisado
- **Hiperparámetros buscados**:
  - `C`: [0.1, 1, 10, 100] - Parámetro de regularización
  - `gamma`: ["scale", "auto", 0.001, 0.01, 0.1, 1] - Coeficiente del kernel
  - `kernel`: ["rbf", "linear", "poly"] - Tipo de kernel
- **Ventajas**: Efectivo en espacios de alta dimensión, excelente rendimiento

### 3. **Árboles de Decisión**
- **Tipo**: Modelo de árbol para clasificación
- **Hiperparámetros buscados**:
  - `max_depth`: [3, 5, 10, 15, 20, None] - Profundidad máxima
  - `min_samples_split`: [2, 5, 10, 20] - Mínimo de muestras para dividir
  - `min_samples_leaf`: [1, 2, 4, 8] - Mínimo de muestras por hoja
  - `criterion`: ["gini", "entropy"] - Criterio de división
- **Ventajas**: Interpretable, maneja datos no lineales

### 4. **Random Forest**
- **Tipo**: Ensemble de árboles de decisión para clasificación
- **Hiperparámetros buscados**:
  - `n_estimators`: [50, 100, 200, 300] - Número de árboles
  - `max_depth`: [3, 5, 10, 15, None] - Profundidad máxima
  - `min_samples_split`: [2, 5, 10] - Mínimo de muestras para dividir
  - `min_samples_leaf`: [1, 2, 4] - Mínimo de muestras por hoja
  - `criterion`: ["gini", "entropy"] - Criterio de división
- **Ventajas**: Robusto, reduce overfitting

### 5. **Redes Neuronales Artificiales (MLP)**
- **Tipo**: Perceptrón multicapa para clasificación
- **Hiperparámetros buscados**:
  - `hidden_layer_sizes`: [[50], [100], [50, 50], [100, 50]] - Arquitectura de capas
  - `activation`: ["relu", "tanh"] - Función de activación
  - `solver`: ["adam", "lbfgs"] - Algoritmo de optimización
  - `alpha`: [0.0001, 0.001, 0.01] - Parámetro de regularización
  - `max_iter`: [200, 500, 1000] - Iteraciones máximas
- **Ventajas**: Aprende patrones complejos no lineales

## 📊 Dataset

- **Fuente**: Kaggle - Coffee Shop Daily Revenue Prediction Dataset
- **Variables**: 6 características de entrada + 1 variable objetivo
- **Tamaño**: 2,000 registros
- **Objetivo**: Clasificar ingresos diarios en categorías (Bajo, Medio, Alto)
- **Hash SHA-256**: `65cbae03b79e3896c6a61e4cc76b228d7468a8f2169182d34d4f6c9e93b58a2b`

### Características del Dataset:
- `Number_of_Customers_Per_Day`: Número de clientes por día
- `Average_Order_Value`: Valor promedio de orden
- `Operating_Hours_Per_Day`: Horas de operación por día
- `Number_of_Employees`: Número de empleados
- `Marketing_Spend_Per_Day`: Gasto en marketing por día
- `Location_Foot_Traffic`: Tráfico peatonal de la ubicación
- `Daily_Revenue`: Ingresos diarios (convertido a categorías: Bajo, Medio, Alto)

## 🚀 Instalación y Uso

### 📦 **Instalación (Solo una vez)**
```bash
pip install -r requirements.txt
```

### 🎮 **Modos de Ejecución**

#### 🖥️ **Modo GUI - Para Exploración y Aprendizaje**
```bash
python main.py
```
**¿Qué hace?**
- Abre interfaz visual intuitiva con botones paso a paso
- Permite explorar datos, entrenar modelos y ver resultados
- Ideal para aprender y entender el proceso
- Muestra gráficos integrados en la aplicación

#### ⚡ **Modo Batch - Para Análisis Completos y Producción**
```bash
# Ejecutar análisis completo con configuración por defecto
python run_pipeline.py

# Ejecutar con configuración personalizada
python run_pipeline.py config.json
```
**¿Qué hace?**
- Ejecuta TODO el análisis automáticamente
- Genera todos los reportes y gráficos
- Ideal para análisis repetitivos o presentaciones
- Genera metadatos completos de reproducibilidad

## 📁 **Dónde Quedan los Resultados**

Después de ejecutar el análisis, todos los resultados se guardan en carpetas organizadas:

```
📊 reports/                          # 🎯 TODOS LOS RESULTADOS AQUÍ
├── run_metadata.json            # 🔒 Metadatos de reproducibilidad
├── tables/                      # 📋 Tablas de datos
│   ├── comparison.csv           # Comparación de métricas por modelo
│   └── predictions.csv          # Predicciones vs valores reales
└── figures/                     # 🎨 Gráficos profesionales
    ├── eda_distributions.png    # Distribuciones de variables
    ├── corr_heatmap.png         # Mapa de calor de correlaciones
    ├── scatter_pairs.png        # Gráficos de pares de dispersión
    ├── metrics_comparison.png   # Comparación de métricas
    └── confusion_matrices.png    # Matrices de confusión

🤖 models_store/                     # Modelos entrenados listos para usar
├── logistic_regression.pkl      # Modelo de regresión logística
├── svm.pkl                      # Modelo SVM
├── decision_tree.pkl            # Modelo de árbol de decisión
├── random_forest.pkl            # Modelo Random Forest
├── neural_network.pkl           # Modelo de red neuronal
└── scaler.pkl                   # Escalador de características
```

### 📈 **Ejemplo de Resultados Reales**

Según el último análisis ejecutado:

| 🏆 Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------| 
| **SVM** | **0.9000** | **0.9010** | **0.9000** | **0.9003** |
| Random Forest | 0.9000 | 0.9004 | 0.9000 | 0.9000 |
| Red Neuronal | 0.9000 | 0.9021 | 0.9000 | 0.9003 |
| Regresión Logística | 0.8850 | 0.8861 | 0.8850 | 0.8853 |
| Árbol de Decisión | 0.8600 | 0.8646 | 0.8600 | 0.8603 |

**🎯 Mejor modelo general: SVM** (mejor Accuracy, Recall y F1-Score)

## ⚙️ **Configuración Personalizable**

Puedes personalizar el análisis editando el archivo `config.json`:

### 🔧 **Parámetros Principales que Puedes Cambiar:**

- **`test_size`**: Proporción de datos para testing (0.1-0.3 recomendado)
- **`random_state`**: Semilla para reproducibilidad (cambia para diferentes resultados)
- **`cv_folds`**: Número de folds para validación cruzada (3-10 recomendado)
- **`n_iter`**: Iteraciones para búsqueda de hiperparámetros (más = mejor, pero más lento)
- **`n_jobs`**: Número de procesos paralelos (-1 = todos los cores)

### 📝 **Ejemplo de Configuración Personalizada:**

```json
{
  "preprocessing": {
    "test_size": 0.25,        // 25% para testing
    "random_state": 123       // Semilla diferente
  },
  "training": {
    "cv_folds": 10,           // 10 folds para validación
    "n_iter": 100,            // 100 iteraciones de búsqueda
    "n_jobs": -1              // Usar todos los cores
  },
  "models": {
    "random_forest": {
      "enabled": true,
      "hyperparameters": {
        "n_estimators": [100, 200, 500],  // Más árboles
        "max_depth": [10, 20, 30]         // Profundidad mayor
      }
    }
  }
}
```

### 🎯 **Casos de Uso Comunes:**

- **Análisis rápido**: `n_iter: 20`, `cv_folds: 3`
- **Análisis preciso**: `n_iter: 100`, `cv_folds: 10`
- **Solo un modelo**: Deshabilitar otros en `"enabled": false`

## 📁 **Estructura del Proyecto**

```
📂 Machine-Learning-Algorithms/
├── 🚀 main.py                     # Lanzador GUI (python main.py)
├── ⚡ run_pipeline.py            # Lanzador batch (python run_pipeline.py)
├── ⚙️ config.json                # Configuración personalizable
├── 📊 coffee_shop_revenue.csv    # Dataset de cafeterías
├── 📋 requirements.txt           # Dependencias (pip install -r requirements.txt)
├── 📖 README.md                  # Esta documentación
│
├── 📂 src/                       # Código fuente principal
│   ├── 📂 core/                  # Funcionalidades principales
│   │   ├── data_handler.py       # Manejo y preprocesamiento de datos
│   │   └── model_comparator.py   # Comparación y evaluación de modelos
│   ├── 📂 models/                # Algoritmos de Machine Learning
│   │   ├── base_model.py         # Clase base para modelos
│   │   ├── linear_regression.py  # Regresión lineal
│   │   ├── svm_model.py          # Máquinas de Vector de Soporte
│   │   ├── decision_tree.py      # Árboles de decisión
│   │   ├── random_forest.py      # Random Forest
│   │   ├── neural_network.py     # Redes neuronales
│   │   └── regression_functions.py # Funciones de entrenamiento
│   ├── 📂 gui/                   # Interfaz gráfica
│   │   └── main_window.py        # Ventana principal de la GUI
│   ├── 📂 eda/                   # Análisis exploratorio
│   │   └── eda_plots.py          # Generación de gráficos EDA
│   ├── 📂 evaluation/            # Evaluación de modelos
│   │   ├── metrics.py            # Cálculo de métricas
│   │   └── plotting.py           # Gráficos de evaluación
│   └── 📂 utils/                 # Utilidades y helpers
│       ├── constants.py          # Constantes del sistema
│       ├── helpers.py            # Funciones auxiliares
│       └── metadata.py           # Generación de metadatos
│
└── 📂 cli/                       # Interfaz de línea de comandos
    └── run_batch.py             # Pipeline batch reproducible
```

## 🎮 **Funcionalidades por Modo**

### 🖥️ **Modo GUI - Interactivo**
1. **📊 Explorar y visualizar datos** - Análisis exploratorio completo
2. **🔧 Preparar datos** - Preprocesamiento y división train/test
3. **🤖 Entrenar modelos** - Entrenamiento de los 5 algoritmos
4. **📈 Comparar modelos** - Evaluación y comparación de rendimiento
5. **🎯 Visualizar resultados** - Gráficos de comparación y análisis
6. **📋 Generar reporte** - Reporte completo de resultados
7. **💾 Guardar resultados** - Exportar resultados a CSV
8. **🚀 Análisis completo** - Ejecución automática de todo el proceso

### ⚡ **Modo Batch - Automatizado**
- **🔄 Ejecución completa automatizada** - Todo el pipeline en una sola ejecución
- **📊 Generación automática de reportes** - Todos los archivos de salida
- **🔒 Metadatos de reproducibilidad** - Información completa de la ejecución
- **✅ Validación de configuración** - Verificación automática de parámetros
- **⚡ Procesamiento paralelo** - Usa todos los cores disponibles

## 📊 **Métricas de Evaluación**

El sistema evalúa cada modelo con 4 métricas estándar de clasificación:

- **🎯 Accuracy**: Precisión general - Proporción de predicciones correctas
- **📊 Precision**: Precisión por clase - Proporción de predicciones positivas correctas
- **📈 Recall**: Sensibilidad - Proporción de casos positivos detectados correctamente
- **⚖️ F1-Score**: Media armónica entre Precision y Recall - Balance entre precisión y sensibilidad

### 🎯 **Interpretación de Métricas:**
- **Todas las métricas**: **Mayor es mejor** (1 = perfecto, 0 = peor)
- **Accuracy**: Mide la precisión general del modelo
- **Precision**: Mide qué tan precisas son las predicciones positivas
- **Recall**: Mide qué tan bien detecta el modelo los casos positivos
- **F1-Score**: Balance entre precisión y sensibilidad

## 🔒 **Reproducibilidad Garantizada**

El proyecto garantiza **reproducibilidad completa** mediante:

- **🎲 Semillas fijas**: `random_state=42` en todos los componentes
- **📋 Metadatos de ejecución**: Archivo `run_metadata.json` con:
  - Timestamp de ejecución
  - Versiones de todas las librerías
  - Configuración completa utilizada
  - Hash SHA-256 del dataset
  - Información del sistema
- **✅ Validación automática**: Verificación de configuración de reproducibilidad
- **🌱 Semillas globales**: Establecimiento de semillas de numpy y Python

## 🏆 **Resultados que Obtienes**

Después de ejecutar el análisis, obtienes:

- **📊 Tabla comparativa** de métricas de clasificación por modelo
- **🏆 Identificación del mejor modelo** según diferentes criterios de clasificación
- **🎨 Visualizaciones profesionales** (matrices de confusión, gráficos de métricas)
- **📋 Reporte completo** con recomendaciones de clasificación
- **💾 Archivos CSV** con resultados exportables
- **🔒 Metadatos completos** para reproducibilidad
- **🤖 Modelos entrenados** listos para usar en producción
- **📈 Categorización automática** de ingresos (Bajo, Medio, Alto)

## 🎓 **Información Académica**

**Universidad Pedagógica y Tecnológica de Colombia**  
**Facultad**: Ingeniería - Escuela de Ingeniería de Sistemas y Computación  
**Materia**: Inteligencia Computacional  

## 👥 **Uso Académico**

Este proyecto está diseñado para:
- ✅ **Demostrar la implementación práctica** de algoritmos de clasificación
- ✅ **Comparar diferentes enfoques** de aprendizaje automático para clasificación
- ✅ **Generar reportes profesionales** con visualizaciones de clasificación
- ✅ **Proporcionar una base** para análisis similares de clasificación
- ✅ **Enseñar buenas prácticas** de reproducibilidad en ML
- ✅ **Mostrar el flujo completo** de un proyecto de clasificación profesional

## 📝 **Notas Técnicas**

- **🔍 Optimización**: RandomizedSearchCV para búsqueda eficiente de hiperparámetros
- **✅ Validación**: Validación cruzada de 5 folds por defecto
- **🔧 Preprocesamiento**: Estandarización con StandardScaler
- **🎯 Clasificación**: Categorización automática de ingresos en 3 clases
- **🎲 Reproducibilidad**: Semillas aleatorias fijas para resultados consistentes
- **⚡ Paralelización**: Uso de todos los cores disponibles (`n_jobs=-1`)
- **💾 Caché**: Sistema de caché para modelos y scalers

## 🚀 **Comandos Rápidos**

```bash
# 📦 Instalación
pip install -r requirements.txt

# 🖥️ Modo GUI
python main.py

# ⚡ Modo Batch
python run_pipeline.py

# ⚙️ Con configuración personalizada
python run_pipeline.py config.json

# 📋 Verificar metadatos
cat reports/run_metadata.json
```

---

*Desarrollado con ❤️ para el aprendizaje de Machine Learning*