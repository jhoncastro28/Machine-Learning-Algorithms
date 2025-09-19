"""
Módulo de métricas y evaluación de modelos
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from typing import Dict, Any, Union


def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de regresión para evaluar el rendimiento del modelo.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        Dict con las métricas: MSE, RMSE, MAE, R2, MAPE_handled
    """
    # Convertir a numpy arrays si es necesario
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # MSE (Mean Squared Error)
    mse = mean_squared_error(y_true, y_pred)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    
    # R² (Coefficient of Determination)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error) - manejado para evitar división por cero
    # Filtramos valores donde y_true es 0 o muy cercano a 0
    mask = np.abs(y_true) > 1e-8
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan  # Si todos los valores reales son 0, MAPE no está definido
    
    return {
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4),
        'MAPE_handled': round(mape, 4) if not np.isnan(mape) else None
    }


def cv_score(estimator: Any, X: np.ndarray, y: np.ndarray, 
             cv: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, float]:
    """
    Realiza validación cruzada y retorna estadísticas de los scores.
    
    Args:
        estimator: Modelo a evaluar
        X: Características de entrada
        y: Variable objetivo
        cv: Número de folds para validación cruzada
        scoring: Métrica de evaluación
        
    Returns:
        Dict con mean y std de los scores de validación cruzada
    """
    # Realizar validación cruzada
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    
    # Si la métrica es negativa (como neg_mean_squared_error), convertir a positiva
    if scoring.startswith('neg_'):
        scores = -scores
    
    return {
        'mean': round(np.mean(scores), 4),
        'std': round(np.std(scores), 4)
    }


def compare_models(y_test: np.ndarray, predictions_dict: Dict[str, np.ndarray], 
                  metrics_fn: callable) -> pd.DataFrame:
    """
    Compara múltiples modelos usando las métricas especificadas.
    
    Args:
        y_test: Valores reales de prueba
        predictions_dict: Diccionario con {nombre_modelo: predicciones}
        metrics_fn: Función para calcular métricas (ej: metrics_regression)
        
    Returns:
        pd.DataFrame: Tabla de comparación con métricas por modelo
    """
    results_list = []
    
    for model_name, y_pred in predictions_dict.items():
        # Calcular métricas para este modelo
        metrics = metrics_fn(y_test, y_pred)
        
        # Crear fila de resultados
        result_row = {'Modelo': model_name}
        result_row.update(metrics)
        results_list.append(result_row)
    
    # Crear DataFrame de comparación
    comparison_df = pd.DataFrame(results_list)
    
    return comparison_df


def save_results_table(results_df: pd.DataFrame, out_path: str) -> None:
    """
    Guarda la tabla de resultados en un archivo CSV.
    
    Args:
        results_df: DataFrame con los resultados
        out_path: Ruta donde guardar el archivo
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Guardar el DataFrame como CSV
    results_df.to_csv(out_path, index=False, encoding='utf-8')
    
    print(f"Resultados guardados en: {out_path}")
