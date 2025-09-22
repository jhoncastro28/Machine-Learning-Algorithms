"""
Funciones independientes para algoritmos de regresión y clasificación
"""

import numpy as np
import joblib
import os
from typing import Optional, Dict, Any, Union
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def train_linear_regression(X_train_t, y_train, X_val_t=None, y_val=None, random_state=42):
    """
    Entrena un modelo de regresión lineal
    
    Args:
        X_train_t: Datos de entrenamiento transformados
        y_train: Etiquetas de entrenamiento
        X_val_t: Datos de validación transformados (opcional)
        y_val: Etiquetas de validación (opcional)
        random_state: Semilla para reproducibilidad
        
    Returns:
        fitted_estimator: Modelo entrenado
    """
    model = LinearRegression()
    model.fit(X_train_t, y_train)
    return model


def train_svm_regressor(X_train_t, y_train, search_space, cv=5, n_iter=50, random_state=42):
    """
    Entrena un modelo SVM para regresión con búsqueda de hiperparámetros
    
    Args:
        X_train_t: Datos de entrenamiento transformados
        y_train: Etiquetas de entrenamiento
        search_space: Espacio de búsqueda de hiperparámetros
        cv: Número de folds para validación cruzada
        n_iter: Número de iteraciones para búsqueda aleatoria
        random_state: Semilla para reproducibilidad
        
    Returns:
        best_estimator: Mejor modelo encontrado
    """
    base_model = SVR()
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    random_search.fit(X_train_t, y_train)
    return random_search.best_estimator_


def train_decision_tree_regressor(X_train_t, y_train, search_space_sin_auto, cv=5, n_iter=50, random_state=42):
    """
    Entrena un modelo de árbol de decisión para regresión con búsqueda de hiperparámetros
    
    Args:
        X_train_t: Datos de entrenamiento transformados
        y_train: Etiquetas de entrenamiento
        search_space_sin_auto: Espacio de búsqueda sin parámetros automáticos
        cv: Número de folds para validación cruzada
        n_iter: Número de iteraciones para búsqueda aleatoria
        random_state: Semilla para reproducibilidad
        
    Returns:
        best_estimator: Mejor modelo encontrado
    """
    base_model = DecisionTreeRegressor(random_state=random_state)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space_sin_auto,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    random_search.fit(X_train_t, y_train)
    return random_search.best_estimator_


def train_random_forest_regressor(X_train_t, y_train, search_space_sin_auto, cv=5, n_iter=50, random_state=42):
    """
    Entrena un modelo de Random Forest para regresión con búsqueda de hiperparámetros
    
    Args:
        X_train_t: Datos de entrenamiento transformados
        y_train: Etiquetas de entrenamiento
        search_space_sin_auto: Espacio de búsqueda sin parámetros automáticos
        cv: Número de folds para validación cruzada
        n_iter: Número de iteraciones para búsqueda aleatoria
        random_state: Semilla para reproducibilidad
        
    Returns:
        best_estimator: Mejor modelo encontrado
    """
    base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space_sin_auto,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    random_search.fit(X_train_t, y_train)
    return random_search.best_estimator_


def train_mlp_regressor(X_train_t, y_train, search_space, cv=5, n_iter=50, random_state=42):
    """
    Entrena un modelo MLP (Multi-Layer Perceptron) para regresión con búsqueda de hiperparámetros
    
    Args:
        X_train_t: Datos de entrenamiento transformados
        y_train: Etiquetas de entrenamiento
        search_space: Espacio de búsqueda de hiperparámetros
        cv: Número de folds para validación cruzada
        n_iter: Número de iteraciones para búsqueda aleatoria
        random_state: Semilla para reproducibilidad
        
    Returns:
        best_estimator: Mejor modelo encontrado
    """
    base_model = MLPRegressor(random_state=random_state, max_iter=1000)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    random_search.fit(X_train_t, y_train)
    return random_search.best_estimator_


def predict(estimator, X_test_t):
    """
    Realiza predicciones usando un estimador entrenado
    
    Args:
        estimator: Modelo entrenado
        X_test_t: Datos de prueba transformados
        
    Returns:
        y_pred: Predicciones
    """
    return estimator.predict(X_test_t)


def feature_importance(estimator) -> Optional[np.ndarray]:
    """
    Obtiene la importancia de las características según el tipo de modelo
    
    Args:
        estimator: Modelo entrenado
        
    Returns:
        np.ndarray | None: Importancia de características o None si no está disponible
    """
    # Árbol de decisión y Random Forest
    if hasattr(estimator, 'feature_importances_'):
        return estimator.feature_importances_
    
    # SVM lineal
    elif hasattr(estimator, 'coef_') and estimator.coef_ is not None:
        # Para SVM lineal, retornamos los coeficientes absolutos
        return np.abs(estimator.coef_.flatten())
    
    # MLP y SVM RBF - no tienen importancia directa
    else:
        return None


def feature_importance_permutation(estimator, X_val, y_val, n_repeats=10, random_state=42):
    """
    Calcula la importancia de características por permutación (para MLP y SVM RBF)
    
    Args:
        estimator: Modelo entrenado
        X_val: Datos de validación
        y_val: Etiquetas de validación
        n_repeats: Número de repeticiones para permutación
        random_state: Semilla para reproducibilidad
        
    Returns:
        np.ndarray: Importancia por permutación
    """
    perm_importance = permutation_importance(
        estimator, X_val, y_val, 
        n_repeats=n_repeats, 
        random_state=random_state,
        scoring='neg_mean_squared_error'
    )
    return perm_importance.importances_mean


def save_model(estimator, name):
    """
    Guarda un modelo entrenado usando joblib
    
    Args:
        estimator: Modelo entrenado
        name: Nombre del archivo (sin extensión)
    """
    # Crear directorio models si no existe
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    filepath = os.path.join(models_dir, f"{name}.joblib")
    joblib.dump(estimator, filepath)
    print(f"Modelo guardado en: {filepath}")


def load_model(name):
    """
    Carga un modelo guardado usando joblib
    
    Args:
        name: Nombre del archivo (sin extensión)
        
    Returns:
        estimator: Modelo cargado
    """
    filepath = os.path.join("models", f"{name}.joblib")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modelo no encontrado: {filepath}")
    
    estimator = joblib.load(filepath)
    print(f"Modelo cargado desde: {filepath}")
    return estimator


# =============================================================================
# FUNCIONES DE CLASIFICACIÓN
# =============================================================================

def train_logistic_regression(X_train_t, y_train, search_space=None, cv=5, n_iter=50, random_state=42):
    """
    Entrena un modelo de Regresión Logística con búsqueda de hiperparámetros
    
    Args:
        X_train_t: Datos de entrenamiento transformados
        y_train: Etiquetas de entrenamiento
        search_space: Espacio de búsqueda de hiperparámetros
        cv: Número de folds para validación cruzada
        n_iter: Número de iteraciones para búsqueda aleatoria
        random_state: Semilla para reproducibilidad
        
    Returns:
        fitted_estimator: Mejor modelo encontrado
    """
    print("🔄 Entrenando Regresión Logística...")
    
    # Configuración por defecto si no se proporciona espacio de búsqueda
    if search_space is None:
        search_space = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'lbfgs', 'saga'],
            'max_iter': [100, 200, 500, 1000],
            'class_weight': [None, 'balanced']
        }
    
    # Crear modelo base
    base_model = LogisticRegression(random_state=random_state)
    
    # Configurar búsqueda aleatoria
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    # Entrenar modelo
    random_search.fit(X_train_t, y_train)
    
    # Obtener mejor modelo
    best_model = random_search.best_estimator_
    
    print(f"✅ Regresión Logística entrenada")
    print(f"   • Mejores parámetros: {random_search.best_params_}")
    print(f"   • Mejor score CV: {random_search.best_score_:.4f}")
    
    return best_model


def train_svm_classifier(X_train_t, y_train, search_space=None, cv=5, n_iter=50, random_state=42):
    """
    Entrena un modelo SVM para clasificación con búsqueda de hiperparámetros
    
    Args:
        X_train_t: Datos de entrenamiento transformados
        y_train: Etiquetas de entrenamiento
        search_space: Espacio de búsqueda de hiperparámetros
        cv: Número de folds para validación cruzada
        n_iter: Número de iteraciones para búsqueda aleatoria
        random_state: Semilla para reproducibilidad
        
    Returns:
        fitted_estimator: Mejor modelo encontrado
    """
    print("🔄 Entrenando SVM Clasificador...")
    
    # Configuración por defecto si no se proporciona espacio de búsqueda
    if search_space is None:
        search_space = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
    
    # Crear modelo base
    base_model = SVC(random_state=random_state, probability=True)
    
    # Configurar búsqueda aleatoria
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    # Entrenar modelo
    random_search.fit(X_train_t, y_train)
    
    # Obtener mejor modelo
    best_model = random_search.best_estimator_
    
    print(f"✅ SVM Clasificador entrenado")
    print(f"   • Mejores parámetros: {random_search.best_params_}")
    print(f"   • Mejor score CV: {random_search.best_score_:.4f}")
    
    return best_model


def train_decision_tree_classifier(X_train_t, y_train, search_space=None, cv=5, n_iter=50, random_state=42):
    """
    Entrena un modelo de Árbol de Decisión para clasificación con búsqueda de hiperparámetros
    
    Args:
        X_train_t: Datos de entrenamiento transformados
        y_train: Etiquetas de entrenamiento
        search_space: Espacio de búsqueda de hiperparámetros
        cv: Número de folds para validación cruzada
        n_iter: Número de iteraciones para búsqueda aleatoria
        random_state: Semilla para reproducibilidad
        
    Returns:
        fitted_estimator: Mejor modelo encontrado
    """
    print("🔄 Entrenando Árbol de Decisión Clasificador...")
    
    # Configuración por defecto si no se proporciona espacio de búsqueda
    if search_space is None:
        search_space = {
            'max_depth': [3, 5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        }
    
    # Crear modelo base
    base_model = DecisionTreeClassifier(random_state=random_state)
    
    # Configurar búsqueda aleatoria
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    # Entrenar modelo
    random_search.fit(X_train_t, y_train)
    
    # Obtener mejor modelo
    best_model = random_search.best_estimator_
    
    print(f"✅ Árbol de Decisión Clasificador entrenado")
    print(f"   • Mejores parámetros: {random_search.best_params_}")
    print(f"   • Mejor score CV: {random_search.best_score_:.4f}")
    
    return best_model


def train_random_forest_classifier(X_train_t, y_train, search_space=None, cv=5, n_iter=50, random_state=42):
    """
    Entrena un modelo Random Forest para clasificación con búsqueda de hiperparámetros
    
    Args:
        X_train_t: Datos de entrenamiento transformados
        y_train: Etiquetas de entrenamiento
        search_space: Espacio de búsqueda de hiperparámetros
        cv: Número de folds para validación cruzada
        n_iter: Número de iteraciones para búsqueda aleatoria
        random_state: Semilla para reproducibilidad
        
    Returns:
        fitted_estimator: Mejor modelo encontrado
    """
    print("🔄 Entrenando Random Forest Clasificador...")
    
    # Configuración por defecto si no se proporciona espacio de búsqueda
    if search_space is None:
        search_space = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    
    # Crear modelo base
    base_model = RandomForestClassifier(random_state=random_state)
    
    # Configurar búsqueda aleatoria
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    # Entrenar modelo
    random_search.fit(X_train_t, y_train)
    
    # Obtener mejor modelo
    best_model = random_search.best_estimator_
    
    print(f"✅ Random Forest Clasificador entrenado")
    print(f"   • Mejores parámetros: {random_search.best_params_}")
    print(f"   • Mejor score CV: {random_search.best_score_:.4f}")
    
    return best_model


def train_mlp_classifier(X_train_t, y_train, search_space=None, cv=5, n_iter=50, random_state=42):
    """
    Entrena un modelo MLP para clasificación con búsqueda de hiperparámetros
    
    Args:
        X_train_t: Datos de entrenamiento transformados
        y_train: Etiquetas de entrenamiento
        search_space: Espacio de búsqueda de hiperparámetros
        cv: Número de folds para validación cruzada
        n_iter: Número de iteraciones para búsqueda aleatoria
        random_state: Semilla para reproducibilidad
        
    Returns:
        fitted_estimator: Mejor modelo encontrado
    """
    print("🔄 Entrenando Red Neuronal Clasificador...")
    
    # Configuración por defecto si no se proporciona espacio de búsqueda
    if search_space is None:
        search_space = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [200, 500, 1000]
        }
    
    # Crear modelo base
    base_model = MLPClassifier(random_state=random_state)
    
    # Configurar búsqueda aleatoria
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    # Entrenar modelo
    random_search.fit(X_train_t, y_train)
    
    # Obtener mejor modelo
    best_model = random_search.best_estimator_
    
    print(f"✅ Red Neuronal Clasificador entrenada")
    print(f"   • Mejores parámetros: {random_search.best_params_}")
    print(f"   • Mejor score CV: {random_search.best_score_:.4f}")
    
    return best_model


def evaluate_classification_model(model, X_test, y_test, model_name="Modelo"):
    """
    Evalúa un modelo de clasificación
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        model_name: Nombre del modelo para reportes
        
    Returns:
        dict: Diccionario con métricas de evaluación
    """
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Crear diccionario de métricas
    metrics = {
        'Modelo': model_name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    }
    
    print(f"\n📊 Evaluación de {model_name}:")
    print(f"   • Accuracy: {accuracy:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall: {recall:.4f}")
    print(f"   • F1-Score: {f1:.4f}")
    
    return metrics, y_pred
