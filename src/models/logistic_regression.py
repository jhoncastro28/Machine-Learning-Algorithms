"""
Módulo para Regresión Logística - Clasificación de Ingresos de Cafeterías
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def train_logistic_regression(X_train, y_train, search_space=None, cv_folds=5, n_iter=50, random_state=42):
    """
    Entrena un modelo de Regresión Logística con búsqueda de hiperparámetros
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        search_space: Diccionario con espacio de búsqueda de hiperparámetros
        cv_folds: Número de folds para validación cruzada
        n_iter: Número de iteraciones para búsqueda aleatoria
        random_state: Semilla para reproducibilidad
        
    Returns:
        Mejor modelo de Regresión Logística encontrado
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
        cv=cv_folds,
        scoring='accuracy',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    # Entrenar modelo
    random_search.fit(X_train, y_train)
    
    # Obtener mejor modelo
    best_model = random_search.best_estimator_
    
    print(f"✅ Regresión Logística entrenada")
    print(f"   • Mejores parámetros: {random_search.best_params_}")
    print(f"   • Mejor score CV: {random_search.best_score_:.4f}")
    
    return best_model

def evaluate_logistic_regression(model, X_test, y_test, model_name="Regresión Logística"):
    """
    Evalúa un modelo de Regresión Logística
    
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
    y_pred_proba = model.predict_proba(X_test)
    
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
    
    # Mostrar reporte de clasificación
    print(f"\n📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Mostrar matriz de confusión
    print(f"\n🔍 Matriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return metrics, y_pred, y_pred_proba

def get_feature_importance_logistic(model, feature_names):
    """
    Obtiene la importancia de las características para Regresión Logística
    
    Args:
        model: Modelo entrenado
        feature_names: Lista de nombres de características
        
    Returns:
        dict: Diccionario con importancia de características
    """
    # Para regresión logística multiclase, obtenemos los coeficientes promedio
    if hasattr(model, 'coef_'):
        # Calcular importancia promedio de todas las clases
        importance = np.mean(np.abs(model.coef_), axis=0)
        
        # Crear diccionario
        feature_importance = dict(zip(feature_names, importance))
        
        # Ordenar por importancia
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    else:
        return {}

def save_logistic_model(model, filepath):
    """
    Guarda un modelo de Regresión Logística
    
    Args:
        model: Modelo a guardar
        filepath: Ruta donde guardar el modelo
    """
    import joblib
    joblib.dump(model, filepath)
    print(f"✅ Modelo de Regresión Logística guardado en {filepath}")

def load_logistic_model(filepath):
    """
    Carga un modelo de Regresión Logística
    
    Args:
        filepath: Ruta del modelo a cargar
        
    Returns:
        Modelo cargado
    """
    import joblib
    model = joblib.load(filepath)
    print(f"✅ Modelo de Regresión Logística cargado desde {filepath}")
    return model
