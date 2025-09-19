"""
Funciones para generar gráficos de análisis exploratorio de datos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import warnings

# Configurar estilo de matplotlib
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


def plot_distributions(df: pd.DataFrame, out_dir: str = "figures") -> None:
    """
    Genera gráficos de distribución para todas las variables numéricas del dataset.
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        out_dir (str): Directorio donde guardar la imagen (default: "figures")
    """
    # Crear directorio si no existe
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Obtener solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calcular número de filas y columnas para subplots
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes para facilitar el acceso
    axes_flat = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        ax = axes_flat[i]
        
        # Crear histograma con curva de densidad
        ax.hist(df[col], bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Agregar curva de densidad
        from scipy import stats
        x = np.linspace(df[col].min(), df[col].max(), 100)
        kde = stats.gaussian_kde(df[col])
        ax.plot(x, kde(x), 'r-', linewidth=2, label='Densidad')
        
        # Configurar el subplot
        ax.set_title(f'Distribución de {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Densidad', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Agregar estadísticas básicas
        mean_val = df[col].mean()
        std_val = df[col].std()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Media: {mean_val:.2f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
    
    # Ocultar subplots vacíos
    for i in range(len(numeric_cols), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    output_path = Path(out_dir) / "eda_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Gráfico de distribuciones guardado en: {output_path}")


def plot_correlations(df: pd.DataFrame, out_dir: str = "figures") -> None:
    """
    Genera un mapa de calor de correlaciones entre variables numéricas.
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        out_dir (str): Directorio donde guardar la imagen (default: "figures")
    """
    # Crear directorio si no existe
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Obtener solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calcular matriz de correlación
    correlation_matrix = numeric_df.corr()
    
    # Crear figura
    plt.figure(figsize=(12, 10))
    
    # Crear mapa de calor
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap='RdBu_r',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={"shrink": .8},
        linewidths=0.5
    )
    
    # Configurar título y etiquetas
    plt.title('Matriz de Correlación entre Variables', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Variables', fontsize=12)
    plt.ylabel('Variables', fontsize=12)
    
    # Rotar etiquetas del eje x
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    output_path = Path(out_dir) / "corr_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Mapa de calor de correlaciones guardado en: {output_path}")


def plot_scatter_pairs(df: pd.DataFrame, out_dir: str = "figures", max_vars: int = 6) -> None:
    """
    Genera gráficos de dispersión para pares de variables numéricas.
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        out_dir (str): Directorio donde guardar la imagen (default: "figures")
        max_vars (int): Número máximo de variables a incluir (default: 6)
    """
    # Crear directorio si no existe
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Obtener solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Limitar número de variables si es necesario
    if len(numeric_df.columns) > max_vars:
        # Seleccionar las variables con mayor varianza
        variances = numeric_df.var().sort_values(ascending=False)
        selected_cols = variances.head(max_vars).index
        numeric_df = numeric_df[selected_cols]
        print(f"Seleccionadas {max_vars} variables con mayor varianza: {list(selected_cols)}")
    
    # Crear figura con subplots
    n_vars = len(numeric_df.columns)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(3 * n_vars, 3 * n_vars))
    
    # Si solo hay una variable, convertir axes a matriz 2D
    if n_vars == 1:
        axes = np.array([[axes]])
    
    # Crear gráficos de dispersión
    for i, col1 in enumerate(numeric_df.columns):
        for j, col2 in enumerate(numeric_df.columns):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histograma
                ax.hist(numeric_df[col1], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'{col1}', fontweight='bold')
                ax.set_xlabel(col1)
                ax.set_ylabel('Frecuencia')
            else:
                # Fuera de diagonal: scatter plot
                ax.scatter(numeric_df[col2], numeric_df[col1], alpha=0.6, s=20)
                
                # Calcular y mostrar correlación
                corr = numeric_df[col1].corr(numeric_df[col2])
                ax.set_title(f'r = {corr:.3f}', fontsize=10)
                
                # Configurar etiquetas
                if i == n_vars - 1:  # Última fila
                    ax.set_xlabel(col2)
                if j == 0:  # Primera columna
                    ax.set_ylabel(col1)
            
            # Configurar grid
            ax.grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    output_path = Path(out_dir) / "scatter_pairs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Gráfico de pares de dispersión guardado en: {output_path}")


def create_eda_plots(df: pd.DataFrame, save_path: str = "figures") -> None:
    """
    Función wrapper para generar gráficos EDA (compatible con pipeline CLI).
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        save_path (str): Directorio donde guardar las imágenes (default: "figures")
    """
    generate_eda_report(df, save_path)


def generate_eda_report(df: pd.DataFrame, out_dir: str = "figures") -> None:
    """
    Genera un reporte completo de EDA con todas las visualizaciones.
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        out_dir (str): Directorio donde guardar las imágenes (default: "figures")
    """
    print("Generando reporte completo de EDA...")
    
    # Generar todas las visualizaciones
    plot_distributions(df, out_dir)
    plot_correlations(df, out_dir)
    plot_scatter_pairs(df, out_dir)
    
    print("Reporte de EDA completado exitosamente!")
