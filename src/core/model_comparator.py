"""
Módulo para comparación y evaluación de modelos de Machine Learning
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ..utils.constants import FILES
from ..utils.helpers import save_artifact
from ..utils.helpers import setup_matplotlib
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib
setup_matplotlib()

class ModelComparator:
    """
    Clase para comparar y evaluar múltiples modelos de Machine Learning
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.comparison_df = None
        
    def add_model(self, model, model_name):
        """
        Añade un modelo a la comparación
        
        Args:
            model: Instancia del modelo entrenado
            model_name: Nombre del modelo
        """
        self.models[model_name] = model
        print(f"✅ Modelo '{model_name}' añadido a la comparación")
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Evalúa todos los modelos añadidos
        
        Args:
            X_test: Datos de prueba
            y_test: Etiquetas de prueba
        """
        if not self.models:
            print("❌ No hay modelos para evaluar")
            return
        
        print("\n" + "="*80)
        print("📊 EVALUACIÓN DE MODELOS")
        print("="*80)
        
        results_list = []
        
        for model_name, model in self.models.items():
            print(f"\n🔄 Evaluando {model_name}...")
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calcular métricas adicionales
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # MAPE en %
            
            # Almacenar resultados
            result = {
                'Modelo': model_name,
                'MSE': round(mse, 4),
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'R²': round(r2, 4),
                'MAPE (%)': round(mape, 2),
                'Predicciones': y_pred
            }
            
            self.results[model_name] = result
            results_list.append(result)
            
            print(f"   • MSE: {mse:,.2f}")
            print(f"   • RMSE: {rmse:,.2f}")
            print(f"   • MAE: {mae:,.2f}")
            print(f"   • R²: {r2:.4f}")
            print(f"   • MAPE: {mape:.2f}%")
        
        # Crear DataFrame de comparación
        self.comparison_df = pd.DataFrame(results_list)
        self.comparison_df = self.comparison_df.drop('Predicciones', axis=1)  # Remover predicciones del DF
        
        print(f"\n✅ Evaluación completada para {len(self.models)} modelos")
    
    def create_comparison_table(self):
        """
        Crea y muestra una tabla de comparación
        """
        if self.comparison_df is None:
            print("❌ Primero debe evaluar los modelos")
            return
        
        print("\n" + "="*100)
        print("📋 TABLA DE COMPARACIÓN DE MODELOS")
        print("="*100)
        
        # Mostrar tabla formateada
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(self.comparison_df.to_string(index=False))
        
        # Identificar el mejor modelo para cada métrica
        print(f"\n🏆 MEJORES MODELOS POR MÉTRICA:")
        print(f"   • Mejor MSE (menor): {self.comparison_df.loc[self.comparison_df['MSE'].idxmin(), 'Modelo']}")
        print(f"   • Mejor RMSE (menor): {self.comparison_df.loc[self.comparison_df['RMSE'].idxmin(), 'Modelo']}")
        print(f"   • Mejor MAE (menor): {self.comparison_df.loc[self.comparison_df['MAE'].idxmin(), 'Modelo']}")
        print(f"   • Mejor R² (mayor): {self.comparison_df.loc[self.comparison_df['R²'].idxmax(), 'Modelo']}")
        print(f"   • Mejor MAPE (menor): {self.comparison_df.loc[self.comparison_df['MAPE (%)'].idxmin(), 'Modelo']}")
    
    def plot_comparison_metrics(self):
        """
        Crea gráficos de comparación de métricas
        """
        if self.comparison_df is None:
            print("❌ Primero debe evaluar los modelos")
            return
        
        # Configurar subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('📊 Comparación de Métricas de Modelos', fontsize=16, fontweight='bold')
        
        # Colores para cada modelo
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.comparison_df)))
        
        # 1. MSE
        axes[0, 0].bar(self.comparison_df['Modelo'], self.comparison_df['MSE'], color=colors)
        axes[0, 0].set_title('Error Cuadrático Medio (MSE)', fontweight='bold')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RMSE
        axes[0, 1].bar(self.comparison_df['Modelo'], self.comparison_df['RMSE'], color=colors)
        axes[0, 1].set_title('Raíz del Error Cuadrático Medio (RMSE)', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. MAE
        axes[0, 2].bar(self.comparison_df['Modelo'], self.comparison_df['MAE'], color=colors)
        axes[0, 2].set_title('Error Absoluto Medio (MAE)', fontweight='bold')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. R²
        axes[1, 0].bar(self.comparison_df['Modelo'], self.comparison_df['R²'], color=colors)
        axes[1, 0].set_title('Coeficiente de Determinación (R²)', fontweight='bold')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. MAPE
        axes[1, 1].bar(self.comparison_df['Modelo'], self.comparison_df['MAPE (%)'], color=colors)
        axes[1, 1].set_title('Error Porcentual Absoluto Medio (MAPE)', fontweight='bold')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Comparación múltiple (normalizada)
        metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'MAPE (%)']
        normalized_data = self.comparison_df[metrics_to_plot].copy()
        
        # Normalizar datos (0-1)
        for col in metrics_to_plot:
            normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / (normalized_data[col].max() - normalized_data[col].min())
        
        x = np.arange(len(self.comparison_df['Modelo']))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            axes[1, 2].bar(x + i*width, normalized_data[metric], width, 
                          label=metric, alpha=0.8)
        
        axes[1, 2].set_title('Comparación Normalizada de Métricas', fontweight='bold')
        axes[1, 2].set_ylabel('Valor Normalizado (0-1)')
        axes[1, 2].set_xticks(x + width * 1.5)
        axes[1, 2].set_xticklabels(self.comparison_df['Modelo'], rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, y_test):
        """
        Crea gráficos de predicciones vs valores reales
        """
        if not self.results:
            print("❌ Primero debe evaluar los modelos")
            return
        
        n_models = len(self.results)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle('🎯 Predicciones vs Valores Reales', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, result) in enumerate(self.results.items()):
            row = i // cols
            col = i % cols
            
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            y_pred = result['Predicciones']
            
            # Scatter plot
            ax.scatter(y_test, y_pred, alpha=0.6, s=50)
            
            # Línea perfecta (y=x)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
            
            # Configuración del gráfico
            ax.set_xlabel('Valores Reales', fontweight='bold')
            ax.set_ylabel('Predicciones', fontweight='bold')
            ax.set_title(f'{model_name}\nR² = {result["R²"]:.4f}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Añadir estadísticas
            ax.text(0.05, 0.95, f'RMSE: {result["RMSE"]:.2f}\nMAE: {result["MAE"]:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Ocultar subplots vacíos
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names):
        """
        Crea gráficos de importancia de características
        """
        if not self.models:
            print("❌ No hay modelos para analizar")
            return
        
        # Filtrar modelos que tienen importancia de características
        models_with_importance = {}
        for name, model in self.models.items():
            if hasattr(model, 'get_feature_importance') and model.get_feature_importance() is not None:
                models_with_importance[name] = model
        
        if not models_with_importance:
            print("ℹ️  Ningún modelo tiene información de importancia de características")
            return
        
        n_models = len(models_with_importance)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle('🔍 Importancia de Características por Modelo', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, model) in enumerate(models_with_importance.items()):
            row = i // cols
            col = i % cols
            
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            importance = model.get_feature_importance()
            
            # Crear gráfico de barras horizontal
            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, importance, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Importancia', fontweight='bold')
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Ocultar subplots vacíos
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def get_best_model(self, metric='R²'):
        """
        Retorna el mejor modelo según la métrica especificada
        
        Args:
            metric: Métrica para evaluar ('MSE', 'RMSE', 'MAE', 'R²', 'MAPE (%)')
            
        Returns:
            tuple: (nombre_del_modelo, valor_de_la_métrica)
        """
        if self.comparison_df is None:
            print("❌ Primero debe evaluar los modelos")
            return None
        
        if metric in ['MSE', 'RMSE', 'MAE', 'MAPE (%)']:
            # Para estas métricas, menor es mejor
            best_idx = self.comparison_df[metric].idxmin()
        else:
            # Para R², mayor es mejor
            best_idx = self.comparison_df[metric].idxmax()
        
        best_model = self.comparison_df.loc[best_idx, 'Modelo']
        best_value = self.comparison_df.loc[best_idx, metric]
        
        return best_model, best_value
    
    def save_results(self, filename=None):
        """
        Guarda los resultados en un archivo CSV
        
        Args:
            filename: Nombre del archivo
        """
        if self.comparison_df is None:
            print("❌ No hay resultados para guardar")
            return
        
        if filename is None:
            filename = FILES['results_csv']
        
        self.comparison_df.to_csv(filename, index=False)
        print(f"✅ Resultados guardados en {filename}")
        # Guardar un snapshot de resultados también como artefacto pkl para trazabilidad
        try:
            save_artifact(self.comparison_df, filename.replace('.csv', '.pkl'), metadata={'type': 'comparison_df'})
            print(f"[RESULTS] Snapshot saved: {filename.replace('.csv', '.pkl')}")
        except Exception:
            pass
    
    def generate_report(self):
        """
        Genera un reporte completo de la comparación
        """
        if self.comparison_df is None:
            print("❌ Primero debe evaluar los modelos")
            return
        
        print("\n" + "="*100)
        print("📋 REPORTE COMPLETO DE COMPARACIÓN DE MODELOS")
        print("="*100)
        
        # Resumen general
        print(f"\n📊 RESUMEN GENERAL:")
        print(f"   • Total de modelos evaluados: {len(self.comparison_df)}")
        print(f"   • Mejor modelo general (R²): {self.get_best_model('R²')[0]}")
        print(f"   • R² promedio: {self.comparison_df['R²'].mean():.4f}")
        print(f"   • RMSE promedio: {self.comparison_df['RMSE'].mean():.2f}")
        
        # Ranking de modelos
        print(f"\n🏆 RANKING DE MODELOS (por R²):")
        ranked_models = self.comparison_df.sort_values('R²', ascending=False)
        for i, (_, row) in enumerate(ranked_models.iterrows(), 1):
            print(f"   {i}. {row['Modelo']}: R² = {row['R²']:.4f}")
        
        # Análisis de métricas
        print(f"\n📈 ANÁLISIS DE MÉTRICAS:")
        print(f"   • Rango de R²: {self.comparison_df['R²'].min():.4f} - {self.comparison_df['R²'].max():.4f}")
        print(f"   • Rango de RMSE: {self.comparison_df['RMSE'].min():.2f} - {self.comparison_df['RMSE'].max():.2f}")
        print(f"   • Rango de MAE: {self.comparison_df['MAE'].min():.2f} - {self.comparison_df['MAE'].max():.2f}")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        best_r2_model = self.get_best_model('R²')[0]
        best_rmse_model = self.get_best_model('RMSE')[0]
        
        if best_r2_model == best_rmse_model:
            print(f"   • El modelo '{best_r2_model}' es el mejor en general")
        else:
            print(f"   • Para máxima precisión: usar '{best_r2_model}' (mejor R²)")
            print(f"   • Para menor error: usar '{best_rmse_model}' (mejor RMSE)")
        
        print(f"   • Considerar el balance entre precisión y interpretabilidad")
        print(f"   • Validar con datos adicionales antes de implementar en producción")
