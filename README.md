# ☕ Análisis de Machine Learning - Ingresos de Cafeterías

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis completo de Machine Learning para predecir los ingresos diarios de cafeterías utilizando múltiples algoritmos. Desarrollado como parte del curso de Inteligencia Computacional de la Universidad Pedagógica y Tecnológica de Colombia.

## 🎯 Objetivos

- Implementar y comparar 5 algoritmos de Machine Learning diferentes
- Analizar la efectividad de cada modelo en la predicción de ingresos
- Generar visualizaciones claras y reportes detallados
- Proporcionar una interfaz interactiva para el análisis

## 🤖 Algoritmos Implementados

1. **Regresión Lineal** - Modelo lineal para regresión
2. **Máquinas de Vector de Soporte (SVM)** - Algoritmo de aprendizaje supervisado
3. **Árboles de Decisión** - Modelo de árbol para regresión
4. **Random Forest** - Ensemble de árboles de decisión
5. **Redes Neuronales Artificiales** - Perceptrón multicapa

## 📊 Dataset

- **Fuente**: Kaggle - Coffee Shop Daily Revenue Prediction Dataset
- **Variables**: 6 características de entrada + 1 variable objetivo
- **Tamaño**: ~2000 registros
- **Objetivo**: Predecir ingresos diarios de cafeterías

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

### Ejecución

#### Interfaz Gráfica (Recomendado)
```bash
python main.py
```
o directamente:
```bash
python gui_app.py
```

#### Interfaz de Consola
```bash
python main.py
```

## 📁 Estructura del Proyecto

```
├── src/                   # Código fuente principal
│   ├── core/             # Funcionalidades principales
│   │   ├── data_handler.py      # Manejo y preprocesamiento de datos
│   │   └── model_comparator.py  # Comparación y evaluación de modelos
│   ├── models/           # Algoritmos de Machine Learning
│   │   ├── base_model.py        # Clase base para modelos
│   │   ├── linear_regression.py
│   │   ├── svm_model.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   └── neural_network.py
│   ├── gui/              # Interfaz gráfica
│   │   └── main_window.py       # Ventana principal de la GUI
│   └── utils/            # Utilidades y helpers
│       ├── constants.py         # Constantes del sistema
│       └── helpers.py           # Funciones auxiliares
├── main.py               # Lanzador principal
├── run_gui.py           # Lanzador alternativo
├── setup.py             # Configuración del proyecto
├── coffee_shop_revenue.csv # Dataset de cafeterías
├── requirements.txt     # Dependencias del proyecto
└── README.md           # Documentación
```

## 🎮 Funcionalidades

### Menú Interactivo
1. **Explorar y visualizar datos** - Análisis exploratorio completo
2. **Preparar datos** - Preprocesamiento y división train/test
3. **Entrenar modelos** - Entrenamiento de los 5 algoritmos
4. **Comparar modelos** - Evaluación y comparación de rendimiento
5. **Visualizar resultados** - Gráficos de comparación y análisis
6. **Generar reporte** - Reporte completo de resultados
7. **Guardar resultados** - Exportar resultados a CSV
8. **Análisis completo** - Ejecución automática de todo el proceso

### Características de la Interfaz Gráfica
- 🎨 **Diseño moderno y atractivo** con colores profesionales
- 📱 **Interfaz intuitiva** con botones grandes y claros
- 📊 **Gráficos integrados** directamente en la aplicación
- 🔄 **Procesamiento en segundo plano** para operaciones largas
- 📋 **Ventanas organizadas** con pestañas para diferentes funciones
- 💾 **Guardado automático** de resultados y reportes

### Visualizaciones Incluidas
- 📊 Análisis exploratorio de datos
- 📈 Comparación de métricas por modelo
- 🎯 Predicciones vs valores reales
- 🔍 Importancia de características
- 🌳 Visualización de árboles de decisión
- 📉 Curvas de entrenamiento de redes neuronales

## 📊 Métricas de Evaluación

- **MSE** (Mean Squared Error): Error cuadrático medio
- **RMSE** (Root Mean Squared Error): Raíz del error cuadrático medio
- **MAE** (Mean Absolute Error): Error absoluto medio
- **R²** (Coefficient of Determination): Coeficiente de determinación
- **MAPE** (Mean Absolute Percentage Error): Error porcentual absoluto medio

## 🏆 Resultados Esperados

El análisis proporciona:
- Tabla comparativa de métricas por modelo
- Identificación del mejor modelo según diferentes criterios
- Visualizaciones detalladas de rendimiento
- Reporte completo con recomendaciones
- Archivos CSV con resultados exportables

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

## 📝 Notas Técnicas

- **Optimización**: Se utiliza GridSearchCV para optimizar hiperparámetros
- **Validación**: Validación cruzada de 5 folds
- **Preprocesamiento**: Estandarización de características
- **Reproducibilidad**: Semillas aleatorias fijas para resultados consistentes

## 🔧 Personalización

El código está diseñado para ser modular y fácilmente extensible:
- Añadir nuevos algoritmos en `ml_models.py`
- Modificar métricas de evaluación en `model_comparator.py`
- Personalizar visualizaciones en `data_handler.py`
- Adaptar la interfaz en `main.py`

---

*Desarrollado con ❤️ para el aprendizaje de Machine Learning*
