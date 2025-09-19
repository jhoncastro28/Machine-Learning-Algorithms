"""
Funciones independientes para algoritmos de regresión
"""

import numpy as np
import joblib
import os
from typing import Optional, Dict, Any, Union
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
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
