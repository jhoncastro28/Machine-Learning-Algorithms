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
    train_logistic_regression,
    train_svm_classifier,
    train_decision_tree_classifier,
    train_random_forest_classifier,
    train_mlp_classifier,
    evaluate_classification_model,
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
        
        if not self.data_handler.prepare_data(test_size=test_size, random_state=random_state, classification=True):
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
        
        if models_config['logistic_regression']['enabled']:
            self._train_logistic_regression(X_train, y_train, X_test, y_test, cv_folds, n_iter, scoring, n_jobs, random_state)
        
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
    
    def _train_logistic_regression(self, X_train, y_train, X_test, y_test, cv_folds, n_iter, scoring, n_jobs, random_state):
        """
        Entrena modelo de regresión logística
        """
        print("\n🔄 Entrenando Regresión Logística...")
        
        try:
            # Obtener espacio de búsqueda de configuración
            search_space = self.config['models']['logistic_regression']['hyperparameters']
            
            # Usar función existente
            model = train_logistic_regression(X_train, y_train, search_space, cv_folds, n_iter, random_state)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas de clasificación
            metrics, _ = evaluate_classification_model(model, X_test, y_test, "Regresión Logística")
            
            # Guardar modelo
            if self.config['output']['save_models']:
                model_path = os.path.join(self.config['output']['models_dir'], 'logistic_regression.pkl')
                joblib.dump(model, model_path)
                print(f"✅ Modelo guardado en {model_path}")
            
            # Almacenar resultados
            self.trained_models["Regresión Logística"] = model
            self.results["Regresión Logística"] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ Error entrenando Regresión Logística: {e}")
    
    def _train_svm(self, X_train, y_train, X_test, y_test, cv_folds, n_iter, scoring, n_jobs, random_state):
        """
        Entrena modelo SVM con búsqueda de hiperparámetros
        """
        print("\n🔄 Entrenando SVM...")
        
        try:
            # Obtener espacio de búsqueda de configuración
            search_space = self.config['models']['svm']['hyperparameters']
            
            # Usar función existente
            model = train_svm_classifier(X_train, y_train, search_space, cv_folds, n_iter, random_state)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas de clasificación
            metrics, _ = evaluate_classification_model(model, X_test, y_test, "SVM")
            
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
            model = train_decision_tree_classifier(X_train, y_train, search_space, cv_folds, n_iter, random_state)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas de clasificación
            metrics, _ = evaluate_classification_model(model, X_test, y_test, "Árbol de Decisión")
            
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
            model = train_random_forest_classifier(X_train, y_train, search_space, cv_folds, n_iter, random_state)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas de clasificación
            metrics, _ = evaluate_classification_model(model, X_test, y_test, "Random Forest")
            
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
            model = train_mlp_classifier(X_train, y_train, search_space, cv_folds, n_iter, random_state)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas de clasificación
            metrics, _ = evaluate_classification_model(model, X_test, y_test, "Red Neuronal")
            
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
        Calcula métricas de evaluación para clasificación
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            model_name: Nombre del modelo
            
        Returns:
            dict: Diccionario con métricas
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'Modelo': model_name,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4)
        }
        
        print(f"   • Accuracy: {accuracy:.4f}")
        print(f"   • Precision: {precision:.4f}")
        print(f"   • Recall: {recall:.4f}")
        print(f"   • F1-Score: {f1:.4f}")
        
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
        print(f"   • Mejor Accuracy (mayor): {comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Modelo']}")
        print(f"   • Mejor Precision (mayor): {comparison_df.loc[comparison_df['Precision'].idxmax(), 'Modelo']}")
        print(f"   • Mejor Recall (mayor): {comparison_df.loc[comparison_df['Recall'].idxmax(), 'Modelo']}")
        print(f"   • Mejor F1-Score (mayor): {comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Modelo']}")
        
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
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('📊 Comparación de Métricas de Clasificación', fontsize=16, fontweight='bold')
            
            # Colores para cada modelo
            colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_df)))
            
            # 1. Accuracy
            axes[0, 0].bar(metrics_df['Modelo'], metrics_df['Accuracy'], color=colors)
            axes[0, 0].set_title('Accuracy (Precisión)', fontweight='bold')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1)
            
            # 2. Precision
            axes[0, 1].bar(metrics_df['Modelo'], metrics_df['Precision'], color=colors)
            axes[0, 1].set_title('Precision (Precisión)', fontweight='bold')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
            
            # 3. Recall
            axes[1, 0].bar(metrics_df['Modelo'], metrics_df['Recall'], color=colors)
            axes[1, 0].set_title('Recall (Sensibilidad)', fontweight='bold')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(0, 1)
            
            # 4. F1-Score
            axes[1, 1].bar(metrics_df['Modelo'], metrics_df['F1-Score'], color=colors)
            axes[1, 1].set_title('F1-Score (Media Armónica)', fontweight='bold')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['output']['figures_dir'], 'metrics_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Error al generar gráficos de métricas: {e}")
    
    def _plot_predictions_vs_actual(self, y_test):
        """
        Crea gráficos de matriz de confusión para clasificación
        """
        try:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            n_models = len(self.results)
            cols = 2
            rows = (n_models + 1) // 2
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            fig.suptitle('🎯 Matrices de Confusión - Clasificación', fontsize=16, fontweight='bold')
            
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
                
                # Calcular matriz de confusión
                cm = confusion_matrix(y_test, y_pred)
                
                # Crear heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=sorted(y_test.unique()),
                           yticklabels=sorted(y_test.unique()))
                
                # Configuración del gráfico
                ax.set_xlabel('Predicciones', fontweight='bold')
                ax.set_ylabel('Valores Reales', fontweight='bold')
                ax.set_title(f'{model_name}\nAccuracy = {metrics["Accuracy"]:.4f}', fontweight='bold')
                
                # Añadir estadísticas
                ax.text(0.05, 0.95, f'F1-Score: {metrics["F1-Score"]:.4f}\nPrecision: {metrics["Precision"]:.4f}', 
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
            plt.savefig(os.path.join(self.config['output']['figures_dir'], 'confusion_matrices.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Error al generar matrices de confusión: {e}")
    
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
