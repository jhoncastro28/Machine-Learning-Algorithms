"""
Pipeline batch reproducible para análisis de Machine Learning
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional

Este módulo ejecuta un pipeline completo sin GUI:
- Carga datos y ejecuta EDA
- Prepara datos con preprocesador
- Entrena modelos con búsqueda de hiperparámetros
- Genera predicciones y reportes
- Guarda todos los artefactos
"""

import json
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Agregar el directorio raíz al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

from src.core.data_handler import DataHandler
from src.core.model_comparator import ModelComparator
from src.models.regression_functions import (
    train_linear_regression,
    train_svm_regressor,
    train_decision_tree_regressor,
    train_random_forest_regressor,
    train_mlp_regressor,
    save_model
)
from src.eda.eda_plots import create_eda_plots
from src.utils.helpers import setup_matplotlib
from src.utils.metadata import generate_run_metadata, set_global_random_seeds, validate_reproducibility_setup

warnings.filterwarnings('ignore')

class BatchPipeline:
    """
    Pipeline batch reproducible para análisis de Machine Learning
    """
    
    def __init__(self, config_path="config.json"):
        """
        Inicializa el pipeline con configuración
        
        Args:
            config_path (str): Ruta al archivo de configuración
        """
        self.config = self._load_config(config_path)
        self.data_handler = None
        self.model_comparator = ModelComparator()
        self.trained_models = {}
        self.results = {}
        
        # Validar configuración de reproducibilidad
        validate_reproducibility_setup(self.config)
        
        # Establecer semillas globales
        random_state = self.config.get('preprocessing', {}).get('random_state', 42)
        set_global_random_seeds(random_state)
        
        # Configurar matplotlib
        setup_matplotlib()
        
        # Crear directorios de salida
        self._create_output_directories()
        
        print("🚀 Pipeline batch inicializado")
        print(f"📁 Directorio de reportes: {self.config['output']['reports_dir']}")
        print(f"📁 Directorio de modelos: {self.config['output']['models_dir']}")
    
    def _load_config(self, config_path):
        """
        Carga la configuración desde archivo JSON
        
        Args:
            config_path (str): Ruta al archivo de configuración
            
        Returns:
            dict: Configuración cargada
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Configuración cargada desde {config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ Archivo de configuración no encontrado: {config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Error al parsear configuración: {e}")
            sys.exit(1)
    
    def _create_output_directories(self):
        """
        Crea los directorios de salida necesarios
        """
        directories = [
            self.config['output']['reports_dir'],
            self.config['output']['tables_dir'],
            self.config['output']['figures_dir'],
            self.config['output']['models_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Directorio creado/verificado: {directory}")
    
    def load_and_explore_data(self):
        """
        Carga datos y ejecuta análisis exploratorio
        """
        print("\n" + "="*80)
        print("📊 CARGA Y EXPLORACIÓN DE DATOS")
        print("="*80)
        
        # Inicializar manejador de datos
        csv_path = self.config['data']['csv_path']
        self.data_handler = DataHandler(csv_path)
        
        # Cargar datos
        if not self.data_handler.load_data():
            print("❌ Error al cargar los datos")
            return False
        
        # Explorar datos
        self.data_handler.explore_data()
        
        # Generar visualizaciones EDA si está habilitado
        if self.config['eda']['generate_plots']:
            print("\n📈 Generando visualizaciones EDA...")
            self._generate_eda_plots()
        
        return True
    
    def _generate_eda_plots(self):
        """
        Genera y guarda gráficos de análisis exploratorio
        """
        try:
            # Crear gráficos EDA usando el módulo existente
            create_eda_plots(self.data_handler.data, save_path=self.config['output']['figures_dir'])
            print(f"✅ Gráficos EDA guardados en {self.config['output']['figures_dir']}")
        except Exception as e:
            print(f"⚠️  Error al generar gráficos EDA: {e}")
            # Generar gráficos básicos como fallback
            self._generate_basic_eda_plots()
    
    def _generate_basic_eda_plots(self):
        """
        Genera gráficos EDA básicos como fallback
        """
        try:
            data = self.data_handler.data
            
            # Gráfico de distribución de ingresos
            plt.figure(figsize=(10, 6))
            plt.hist(data['Daily_Revenue'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Distribución de Ingresos Diarios', fontweight='bold')
            plt.xlabel('Ingresos Diarios ($)')
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['output']['figures_dir'], 'eda_revenue_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Matriz de correlación
            plt.figure(figsize=(10, 8))
            corr_matrix = data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title('Matriz de Correlación', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['output']['figures_dir'], 'eda_correlation_matrix.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Gráficos EDA básicos guardados en {self.config['output']['figures_dir']}")
        except Exception as e:
            print(f"⚠️  Error al generar gráficos EDA básicos: {e}")
    
    def prepare_data(self):
        """
        Prepara los datos para entrenamiento
        """
        print("\n" + "="*80)
        print("🔧 PREPARACIÓN DE DATOS")
        print("="*80)
        
        # Preparar datos usando el manejador existente
        test_size = self.config['preprocessing']['test_size']
        random_state = self.config['preprocessing']['random_state']
        
        if not self.data_handler.prepare_data(test_size=test_size, random_state=random_state):
            print("❌ Error al preparar los datos")
            return False
        
        # Guardar scaler si está habilitado
        if self.config['output']['save_scaler']:
            scaler_path = os.path.join(self.config['output']['models_dir'], 'scaler.pkl')
            joblib.dump(self.data_handler.scaler, scaler_path)
            print(f"✅ Scaler guardado en {scaler_path}")
        
        return True
    
    def train_models(self):
        """
        Entrena todos los modelos habilitados
        """
        print("\n" + "="*80)
        print("🤖 ENTRENAMIENTO DE MODELOS")
        print("="*80)
        
        # Obtener datos preparados
        X_train, X_test, y_train, y_test = self.data_handler.get_data()
        
        if X_train is None:
            print("❌ Los datos no están preparados")
            return False
        
        # Configuración de entrenamiento
        cv_folds = self.config['training']['cv_folds']
        n_iter = self.config['training']['n_iter']
        scoring = self.config['training']['scoring']
        n_jobs = self.config['training']['n_jobs']
        random_state = self.config['training']['random_state']
        
        # Entrenar cada modelo habilitado
        models_config = self.config['models']
        
        if models_config['linear_regression']['enabled']:
            self._train_linear_regression(X_train, y_train, X_test, y_test)
        
        if models_config['svm']['enabled']:
            self._train_svm(X_train, y_train, X_test, y_test, cv_folds, n_iter, scoring, n_jobs, random_state)
        
        if models_config['decision_tree']['enabled']:
            self._train_decision_tree(X_train, y_train, X_test, y_test, cv_folds, n_iter, scoring, n_jobs, random_state)
        
        if models_config['random_forest']['enabled']:
            self._train_random_forest(X_train, y_train, X_test, y_test, cv_folds, n_iter, scoring, n_jobs, random_state)
        
        if models_config['neural_network']['enabled']:
            self._train_neural_network(X_train, y_train, X_test, y_test, cv_folds, n_iter, scoring, n_jobs, random_state)
        
        print(f"\n✅ Entrenamiento completado para {len(self.trained_models)} modelos")
        return True
    
    def _train_linear_regression(self, X_train, y_train, X_test, y_test):
        """
        Entrena modelo de regresión lineal
        """
        print("\n🔄 Entrenando Regresión Lineal...")
        
        try:
            # Usar función existente
            model = train_linear_regression(X_train, y_train)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            metrics = self._calculate_metrics(y_test, y_pred, "Regresión Lineal")
            
            # Guardar modelo
            if self.config['output']['save_models']:
                model_path = os.path.join(self.config['output']['models_dir'], 'linear_regression.pkl')
                joblib.dump(model, model_path)
                print(f"✅ Modelo guardado en {model_path}")
            
            # Almacenar resultados
            self.trained_models["Regresión Lineal"] = model
            self.results["Regresión Lineal"] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ Error entrenando Regresión Lineal: {e}")
    
    def _train_svm(self, X_train, y_train, X_test, y_test, cv_folds, n_iter, scoring, n_jobs, random_state):
        """
        Entrena modelo SVM con búsqueda de hiperparámetros
        """
        print("\n🔄 Entrenando SVM...")
        
        try:
            # Obtener espacio de búsqueda de configuración
            search_space = self.config['models']['svm']['hyperparameters']
            
            # Usar función existente
            model = train_svm_regressor(X_train, y_train, search_space, cv_folds, n_iter, random_state)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            metrics = self._calculate_metrics(y_test, y_pred, "SVM")
            
            # Guardar modelo
            if self.config['output']['save_models']:
                model_path = os.path.join(self.config['output']['models_dir'], 'svm.pkl')
                joblib.dump(model, model_path)
                print(f"✅ Modelo guardado en {model_path}")
            
            # Almacenar resultados
            self.trained_models["SVM"] = model
            self.results["SVM"] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ Error entrenando SVM: {e}")
    
    def _train_decision_tree(self, X_train, y_train, X_test, y_test, cv_folds, n_iter, scoring, n_jobs, random_state):
        """
        Entrena modelo de árbol de decisión con búsqueda de hiperparámetros
        """
        print("\n🔄 Entrenando Árbol de Decisión...")
        
        try:
            # Obtener espacio de búsqueda de configuración
            search_space = self.config['models']['decision_tree']['hyperparameters']
            
            # Usar función existente
            model = train_decision_tree_regressor(X_train, y_train, search_space, cv_folds, n_iter, random_state)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            metrics = self._calculate_metrics(y_test, y_pred, "Árbol de Decisión")
            
            # Guardar modelo
            if self.config['output']['save_models']:
                model_path = os.path.join(self.config['output']['models_dir'], 'decision_tree.pkl')
                joblib.dump(model, model_path)
                print(f"✅ Modelo guardado en {model_path}")
            
            # Almacenar resultados
            self.trained_models["Árbol de Decisión"] = model
            self.results["Árbol de Decisión"] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ Error entrenando Árbol de Decisión: {e}")
    
    def _train_random_forest(self, X_train, y_train, X_test, y_test, cv_folds, n_iter, scoring, n_jobs, random_state):
        """
        Entrena modelo Random Forest con búsqueda de hiperparámetros
        """
        print("\n🔄 Entrenando Random Forest...")
        
        try:
            # Obtener espacio de búsqueda de configuración
            search_space = self.config['models']['random_forest']['hyperparameters']
            
            # Usar función existente
            model = train_random_forest_regressor(X_train, y_train, search_space, cv_folds, n_iter, random_state)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            metrics = self._calculate_metrics(y_test, y_pred, "Random Forest")
            
            # Guardar modelo
            if self.config['output']['save_models']:
                model_path = os.path.join(self.config['output']['models_dir'], 'random_forest.pkl')
                joblib.dump(model, model_path)
                print(f"✅ Modelo guardado en {model_path}")
            
            # Almacenar resultados
            self.trained_models["Random Forest"] = model
            self.results["Random Forest"] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ Error entrenando Random Forest: {e}")
    
    def _train_neural_network(self, X_train, y_train, X_test, y_test, cv_folds, n_iter, scoring, n_jobs, random_state):
        """
        Entrena modelo de red neuronal con búsqueda de hiperparámetros
        """
        print("\n🔄 Entrenando Red Neuronal...")
        
        try:
            # Obtener espacio de búsqueda de configuración
            search_space = self.config['models']['neural_network']['hyperparameters']
            
            # Usar función existente
            model = train_mlp_regressor(X_train, y_train, search_space, cv_folds, n_iter, random_state)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            metrics = self._calculate_metrics(y_test, y_pred, "Red Neuronal")
            
            # Guardar modelo
            if self.config['output']['save_models']:
                model_path = os.path.join(self.config['output']['models_dir'], 'neural_network.pkl')
                joblib.dump(model, model_path)
                print(f"✅ Modelo guardado en {model_path}")
            
            # Almacenar resultados
            self.trained_models["Red Neuronal"] = model
            self.results["Red Neuronal"] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ Error entrenando Red Neuronal: {e}")
    
    def _calculate_metrics(self, y_true, y_pred, model_name):
        """
        Calcula métricas de evaluación
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            model_name: Nombre del modelo
            
        Returns:
            dict: Diccionario con métricas
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'Modelo': model_name,
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R²': round(r2, 4),
            'MAPE (%)': round(mape, 2)
        }
        
        print(f"   • MSE: {mse:,.2f}")
        print(f"   • RMSE: {rmse:,.2f}")
        print(f"   • MAE: {mae:,.2f}")
        print(f"   • R²: {r2:.4f}")
        print(f"   • MAPE: {mape:.2f}%")
        
        return metrics
    
    def generate_comparison_report(self):
        """
        Genera reporte de comparación de modelos
        """
        print("\n" + "="*80)
        print("📊 GENERACIÓN DE REPORTE DE COMPARACIÓN")
        print("="*80)
        
        if not self.results:
            print("❌ No hay resultados para comparar")
            return False
        
        # Crear DataFrame de comparación
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append(result['metrics'])
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Mostrar tabla de comparación
        print("\n📋 TABLA DE COMPARACIÓN DE MODELOS:")
        print(comparison_df.to_string(index=False))
        
        # Identificar mejores modelos
        print(f"\n🏆 MEJORES MODELOS POR MÉTRICA:")
        print(f"   • Mejor MSE (menor): {comparison_df.loc[comparison_df['MSE'].idxmin(), 'Modelo']}")
        print(f"   • Mejor RMSE (menor): {comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Modelo']}")
        print(f"   • Mejor MAE (menor): {comparison_df.loc[comparison_df['MAE'].idxmin(), 'Modelo']}")
        print(f"   • Mejor R² (mayor): {comparison_df.loc[comparison_df['R²'].idxmax(), 'Modelo']}")
        print(f"   • Mejor MAPE (menor): {comparison_df.loc[comparison_df['MAPE (%)'].idxmin(), 'Modelo']}")
        
        # Guardar tabla de comparación
        if self.config['output']['save_comparison']:
            comparison_path = os.path.join(self.config['output']['tables_dir'], 'comparison.csv')
            comparison_df.to_csv(comparison_path, index=False)
            print(f"✅ Tabla de comparación guardada en {comparison_path}")
        
        return comparison_df
    
    def generate_predictions_report(self):
        """
        Genera reporte de predicciones
        """
        print("\n📈 Generando reporte de predicciones...")
        
        if not self.results:
            print("❌ No hay predicciones para reportar")
            return False
        
        # Obtener datos de prueba
        _, _, _, y_test = self.data_handler.get_data()
        
        # Crear DataFrame con predicciones
        predictions_data = {
            'Actual': y_test.values
        }
        
        for model_name, result in self.results.items():
            predictions_data[model_name] = result['predictions']
        
        predictions_df = pd.DataFrame(predictions_data)
        
        # Guardar predicciones
        if self.config['output']['save_predictions']:
            predictions_path = os.path.join(self.config['output']['tables_dir'], 'predictions.csv')
            predictions_df.to_csv(predictions_path, index=False)
            print(f"✅ Predicciones guardadas en {predictions_path}")
        
        return predictions_df
    
    def generate_visualization_plots(self):
        """
        Genera gráficos de comparación y visualización
        """
        print("\n📊 Generando gráficos de comparación...")
        
        if not self.results:
            print("❌ No hay resultados para visualizar")
            return False
        
        # Obtener datos de prueba
        _, _, _, y_test = self.data_handler.get_data()
        
        # Crear gráficos de comparación de métricas
        self._plot_metrics_comparison()
        
        # Crear gráficos de predicciones vs reales
        self._plot_predictions_vs_actual(y_test)
        
        print(f"✅ Gráficos guardados en {self.config['output']['figures_dir']}")
    
    def _plot_metrics_comparison(self):
        """
        Crea gráficos de comparación de métricas
        """
        try:
            # Crear DataFrame de métricas
            metrics_data = []
            for model_name, result in self.results.items():
                metrics_data.append(result['metrics'])
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Configurar subplots
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('📊 Comparación de Métricas de Modelos', fontsize=16, fontweight='bold')
            
            # Colores para cada modelo
            colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_df)))
            
            # 1. MSE
            axes[0, 0].bar(metrics_df['Modelo'], metrics_df['MSE'], color=colors)
            axes[0, 0].set_title('Error Cuadrático Medio (MSE)', fontweight='bold')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. RMSE
            axes[0, 1].bar(metrics_df['Modelo'], metrics_df['RMSE'], color=colors)
            axes[0, 1].set_title('Raíz del Error Cuadrático Medio (RMSE)', fontweight='bold')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. MAE
            axes[0, 2].bar(metrics_df['Modelo'], metrics_df['MAE'], color=colors)
            axes[0, 2].set_title('Error Absoluto Medio (MAE)', fontweight='bold')
            axes[0, 2].set_ylabel('MAE')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. R²
            axes[1, 0].bar(metrics_df['Modelo'], metrics_df['R²'], color=colors)
            axes[1, 0].set_title('Coeficiente de Determinación (R²)', fontweight='bold')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. MAPE
            axes[1, 1].bar(metrics_df['Modelo'], metrics_df['MAPE (%)'], color=colors)
            axes[1, 1].set_title('Error Porcentual Absoluto Medio (MAPE)', fontweight='bold')
            axes[1, 1].set_ylabel('MAPE (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Comparación múltiple (normalizada)
            metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'MAPE (%)']
            normalized_data = metrics_df[metrics_to_plot].copy()
            
            # Normalizar datos (0-1)
            for col in metrics_to_plot:
                normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / (normalized_data[col].max() - normalized_data[col].min())
            
            x = np.arange(len(metrics_df['Modelo']))
            width = 0.2
            
            for i, metric in enumerate(metrics_to_plot):
                axes[1, 2].bar(x + i*width, normalized_data[metric], width, 
                              label=metric, alpha=0.8)
            
            axes[1, 2].set_title('Comparación Normalizada de Métricas', fontweight='bold')
            axes[1, 2].set_ylabel('Valor Normalizado (0-1)')
            axes[1, 2].set_xticks(x + width * 1.5)
            axes[1, 2].set_xticklabels(metrics_df['Modelo'], rotation=45)
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['output']['figures_dir'], 'metrics_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Error al generar gráficos de métricas: {e}")
    
    def _plot_predictions_vs_actual(self, y_test):
        """
        Crea gráficos de predicciones vs valores reales
        """
        try:
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
                
                y_pred = result['predictions']
                metrics = result['metrics']
                
                # Scatter plot
                ax.scatter(y_test, y_pred, alpha=0.6, s=50)
                
                # Línea perfecta (y=x)
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
                
                # Configuración del gráfico
                ax.set_xlabel('Valores Reales', fontweight='bold')
                ax.set_ylabel('Predicciones', fontweight='bold')
                ax.set_title(f'{model_name}\nR² = {metrics["R²"]:.4f}', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Añadir estadísticas
                ax.text(0.05, 0.95, f'RMSE: {metrics["RMSE"]:.2f}\nMAE: {metrics["MAE"]:.2f}', 
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
            plt.savefig(os.path.join(self.config['output']['figures_dir'], 'predictions_vs_actual.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Error al generar gráficos de predicciones: {e}")
    
    def generate_execution_metadata(self):
        """
        Genera metadatos de ejecución para reproducibilidad
        """
        print("\n📋 Generando metadatos de ejecución...")
        
        try:
            # Obtener rutas de archivos
            config_path = "config.json"  # Asumiendo que se usa el config por defecto
            dataset_path = self.config['data']['csv_path']
            output_dir = self.config['output']['reports_dir']
            
            # Generar metadatos
            metadata = generate_run_metadata(config_path, dataset_path, output_dir)
            
            print("✅ Metadatos de ejecución generados exitosamente")
            
        except Exception as e:
            print(f"⚠️  Error al generar metadatos: {e}")
    
    def run_pipeline(self):
        """
        Ejecuta el pipeline completo
        """
        print("🚀 INICIANDO PIPELINE BATCH REPRODUCIBLE")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # 1. Cargar y explorar datos
            if not self.load_and_explore_data():
                return False
            
            # 2. Preparar datos
            if not self.prepare_data():
                return False
            
            # 3. Entrenar modelos
            if not self.train_models():
                return False
            
            # 4. Generar reporte de comparación
            comparison_df = self.generate_comparison_report()
            
            # 5. Generar reporte de predicciones
            predictions_df = self.generate_predictions_report()
            
            # 6. Generar gráficos
            self.generate_visualization_plots()
            
            # 7. Generar metadatos de ejecución
            self.generate_execution_metadata()
            
            # Resumen final
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            print("="*80)
            print(f"⏱️  Tiempo total: {duration}")
            print(f"📊 Modelos entrenados: {len(self.trained_models)}")
            print(f"📁 Reportes generados en: {self.config['output']['reports_dir']}")
            print(f"🤖 Modelos guardados en: {self.config['output']['models_dir']}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error en el pipeline: {e}")
            return False


def run_batch(config_path="config.json"):
    """
    Función principal para ejecutar el pipeline batch
    
    Args:
        config_path (str): Ruta al archivo de configuración
    """
    pipeline = BatchPipeline(config_path)
    return pipeline.run_pipeline()


if __name__ == "__main__":
    # Ejecutar pipeline con configuración por defecto
    success = run_batch()
    
    if success:
        print("\n🎉 Pipeline ejecutado exitosamente!")
        sys.exit(0)
    else:
        print("\n💥 Pipeline falló!")
        sys.exit(1)
