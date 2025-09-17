"""
Módulo de visualización para evaluación de modelos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para mejor calidad
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_metrics_bar(comparison_df: pd.DataFrame, out_dir: str) -> None:
    """
    Genera gráficos de barras para cada métrica de comparación.
    
    Args:
        comparison_df: DataFrame con métricas de comparación de modelos
        out_dir: Directorio donde guardar las imágenes
    """
    # Crear directorio si no existe
    os.makedirs(out_dir, exist_ok=True)
    
    # Métricas a graficar
    metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE_handled']
    
    # Colores para cada modelo
    colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))
    
    for metric in metrics:
        if metric in comparison_df.columns:
            # Crear figura
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Crear gráfico de barras
            bars = ax.bar(comparison_df['Modelo'], comparison_df[metric], 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Configurar título y etiquetas
            metric_names = {
                'MSE': 'Error Cuadrático Medio (MSE)',
                'RMSE': 'Raíz del Error Cuadrático Medio (RMSE)',
                'MAE': 'Error Absoluto Medio (MAE)',
                'R2': 'Coeficiente de Determinación (R²)',
                'MAPE_handled': 'Error Porcentual Absoluto Medio (MAPE)'
            }
            
            ax.set_title(f'{metric_names[metric]} por Modelo', fontweight='bold', fontsize=14)
            ax.set_xlabel('Modelo', fontweight='bold')
            ax.set_ylabel(metric, fontweight='bold')
            
            # Rotar etiquetas del eje x si son largas
            ax.tick_params(axis='x', rotation=45)
            
            # Añadir grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Añadir valores en las barras
            for bar, value in zip(bars, comparison_df[metric]):
                if pd.notna(value):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Ajustar layout
            plt.tight_layout()
            
            # Guardar figura
            filename = f'metrics_{metric.lower()}.png'
            filepath = os.path.join(out_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"✅ Gráfico guardado: {filepath}")


def plot_predictions_vs_actual(y_test: np.ndarray, predictions_dict: Dict[str, np.ndarray], 
                              out_dir: str) -> None:
    """
    Genera gráficos de predicciones vs valores reales para cada modelo.
    
    Args:
        y_test: Valores reales de prueba
        predictions_dict: Diccionario con {nombre_modelo: predicciones}
        out_dir: Directorio donde guardar las imágenes
    """
    # Crear directorio si no existe
    os.makedirs(out_dir, exist_ok=True)
    
    for model_name, y_pred in predictions_dict.items():
        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot de predicciones vs reales
        ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
        
        # Línea perfecta (y=x)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
               label='Predicción Perfecta', alpha=0.8)
        
        # Calcular R² para mostrar en el gráfico
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        
        # Configurar título y etiquetas
        ax.set_title(f'Predicciones vs Valores Reales - {model_name}\nR² = {r2:.4f}', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('Valores Reales', fontweight='bold')
        ax.set_ylabel('Predicciones', fontweight='bold')
        
        # Añadir grid
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Añadir estadísticas en el gráfico
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        stats_text = f'RMSE: {np.sqrt(mse):.2f}\nMAE: {mae:.2f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar figura
        filename = f'pred_vs_actual_{model_name.lower().replace(" ", "_")}.png'
        filepath = os.path.join(out_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Gráfico guardado: {filepath}")


def plot_feature_importance(importances_dict: Dict[str, np.ndarray], 
                           feature_names: List[str], out_dir: str) -> None:
    """
    Genera gráficos de importancia de características para cada modelo.
    
    Args:
        importances_dict: Diccionario con {nombre_modelo: importancias}
        feature_names: Lista de nombres de las características
        out_dir: Directorio donde guardar las imágenes
    """
    # Crear directorio si no existe
    os.makedirs(out_dir, exist_ok=True)
    
    for model_name, importances in importances_dict.items():
        if importances is None or len(importances) == 0:
            print(f"⚠️  No hay importancias disponibles para {model_name}")
            continue
            
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Crear gráfico de barras horizontal
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, importances, alpha=0.7, color='lightcoral', 
                      edgecolor='black', linewidth=0.5)
        
        # Configurar etiquetas
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Importancia', fontweight='bold')
        ax.set_title(f'Importancia de Características - {model_name}', 
                    fontweight='bold', fontsize=12)
        
        # Invertir el eje y para mostrar la característica más importante arriba
        ax.invert_yaxis()
        
        # Añadir grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Añadir valores en las barras
        for i, (bar, value) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                   f'{value:.4f}', ha='left', va='center', fontweight='bold')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar figura
        filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        filepath = os.path.join(out_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Gráfico guardado: {filepath}")


def plot_comprehensive_comparison(comparison_df: pd.DataFrame, out_dir: str) -> None:
    """
    Genera un gráfico comprensivo de comparación de todos los modelos.
    
    Args:
        comparison_df: DataFrame con métricas de comparación
        out_dir: Directorio donde guardar la imagen
    """
    # Crear directorio si no existe
    os.makedirs(out_dir, exist_ok=True)
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparación Completa de Modelos', fontsize=16, fontweight='bold')
    
    # Métricas a graficar
    metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE_handled']
    metric_names = {
        'MSE': 'Error Cuadrático Medio',
        'RMSE': 'Raíz del Error Cuadrático Medio',
        'MAE': 'Error Absoluto Medio',
        'R2': 'Coeficiente de Determinación',
        'MAPE_handled': 'Error Porcentual Absoluto Medio'
    }
    
    # Colores para cada modelo
    colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))
    
    # Graficar cada métrica
    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Crear gráfico de barras
            bars = ax.bar(comparison_df['Modelo'], comparison_df[metric], 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_title(metric_names[metric], fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Añadir valores en las barras
            for bar, value in zip(bars, comparison_df[metric]):
                if pd.notna(value):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Gráfico de comparación normalizada en el último subplot
    ax = axes[1, 2]
    metrics_to_normalize = ['MSE', 'RMSE', 'MAE', 'MAPE_handled']
    normalized_data = comparison_df[metrics_to_normalize].copy()
    
    # Normalizar datos (0-1)
    for col in metrics_to_normalize:
        if col in normalized_data.columns:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val > min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    
    x = np.arange(len(comparison_df['Modelo']))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_normalize):
        if metric in normalized_data.columns:
            ax.bar(x + i*width, normalized_data[metric], width, 
                  label=metric, alpha=0.8)
    
    ax.set_title('Comparación Normalizada', fontweight='bold')
    ax.set_ylabel('Valor Normalizado (0-1)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df['Modelo'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    filepath = os.path.join(out_dir, 'comprehensive_comparison.png')
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✅ Gráfico comprensivo guardado: {filepath}")
