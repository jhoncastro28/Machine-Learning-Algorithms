"""
Módulo con implementaciones de algoritmos de Machine Learning
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class BaseMLModel:
    """
    Clase base para todos los modelos de Machine Learning
    """
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        
    def train(self, X_train, y_train):
        """
        Entrena el modelo
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
        """
        raise NotImplementedError("Debe implementar el método train")
    
    def predict(self, X_test):
        """
        Realiza predicciones
        
        Args:
            X_test: Datos de prueba
            
        Returns:
            array: Predicciones
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        return self.model.predict(X_test)
    
    def get_feature_importance(self):
        """
        Retorna la importancia de las características
        """
        return self.feature_importance
    
    def cross_validate(self, X, y, cv=5):
        """
        Realiza validación cruzada
        
        Args:
            X: Datos
            y: Etiquetas
            cv: Número de folds
            
        Returns:
            array: Scores de validación cruzada
        """
        if not self.is_trained:
            self.train(X, y)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error')
        return -scores  # Convertir a positivo

class LogisticRegressionModel(BaseMLModel):
    """
    Implementación de Regresión Logística para regresión
    """
    
    def __init__(self):
        super().__init__("Regresión Logística")
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        
    def train(self, X_train, y_train):
        """
        Entrena el modelo de regresión logística
        """
        print(f"🔄 Entrenando {self.model_name}...")
        
        # Para regresión, usamos LogisticRegression con ajuste
        # En este caso, convertiremos el problema a clasificación por rangos
        y_train_categorical = self._convert_to_categories(y_train)
        
        self.model.fit(X_train, y_train_categorical)
        self.is_trained = True
        
        # Calcular importancia de características (coeficientes)
        if hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_[0])
        
        print(f"✅ {self.model_name} entrenado exitosamente")
    
    def _convert_to_categories(self, y):
        """
        Convierte valores continuos a categorías para regresión logística
        """
        # Dividir en 5 categorías basadas en cuartiles
        return pd.cut(y, bins=5, labels=[0, 1, 2, 3, 4])
    
    def predict(self, X_test):
        """
        Realiza predicciones y las convierte de vuelta a valores continuos
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        predictions_categorical = self.model.predict(X_test)
        # Convertir de vuelta a valores continuos (aproximación)
        return predictions_categorical * 500 + 1000  # Ajuste aproximado

class SVMModel(BaseMLModel):
    """
    Implementación de Máquinas de Vector de Soporte para regresión
    """
    
    def __init__(self):
        super().__init__("Máquinas de Vector de Soporte")
        self.model = SVR(kernel='rbf', C=1.0, gamma='scale')
        
    def train(self, X_train, y_train):
        """
        Entrena el modelo SVM
        """
        print(f"🔄 Entrenando {self.model_name}...")
        
        # Optimización de hiperparámetros
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        print(f"   Mejores parámetros: {grid_search.best_params_}")
        print(f"   Mejor score: {-grid_search.best_score_:.4f}")
        
        self.is_trained = True
        
        # Para SVM, la importancia de características no es directa
        # Usaremos los coeficientes si el kernel es lineal
        if hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_[0])
        else:
            self.feature_importance = None
        
        print(f"✅ {self.model_name} entrenado exitosamente")

class DecisionTreeModel(BaseMLModel):
    """
    Implementación de Árboles de Decisión para regresión
    """
    
    def __init__(self):
        super().__init__("Árboles de Decisión")
        self.model = DecisionTreeRegressor(random_state=42)
        
    def train(self, X_train, y_train):
        """
        Entrena el modelo de árbol de decisión
        """
        print(f"🔄 Entrenando {self.model_name}...")
        
        # Optimización de hiperparámetros
        param_grid = {
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        print(f"   Mejores parámetros: {grid_search.best_params_}")
        print(f"   Mejor score: {-grid_search.best_score_:.4f}")
        
        self.is_trained = True
        
        # Importancia de características
        self.feature_importance = self.model.feature_importances_
        
        print(f"✅ {self.model_name} entrenado exitosamente")
    
    def visualize_tree(self, feature_names, max_depth=3):
        """
        Visualiza el árbol de decisión (primeros niveles)
        """
        if not self.is_trained:
            print("❌ El modelo debe ser entrenado primero")
            return
        
        from sklearn.tree import plot_tree
        
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, feature_names=feature_names, 
                 max_depth=max_depth, filled=True, rounded=True, fontsize=10)
        plt.title(f'🌳 Árbol de Decisión - {self.model_name} (Primeros {max_depth} niveles)', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

class RandomForestModel(BaseMLModel):
    """
    Implementación de Random Forest para regresión
    """
    
    def __init__(self):
        super().__init__("Random Forest")
        self.model = RandomForestRegressor(random_state=42, n_estimators=100)
        
    def train(self, X_train, y_train):
        """
        Entrena el modelo Random Forest
        """
        print(f"🔄 Entrenando {self.model_name}...")
        
        # Optimización de hiperparámetros
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        print(f"   Mejores parámetros: {grid_search.best_params_}")
        print(f"   Mejor score: {-grid_search.best_score_:.4f}")
        
        self.is_trained = True
        
        # Importancia de características
        self.feature_importance = self.model.feature_importances_
        
        print(f"✅ {self.model_name} entrenado exitosamente")

class NeuralNetworkModel(BaseMLModel):
    """
    Implementación de Redes Neuronales Artificiales para regresión
    """
    
    def __init__(self):
        super().__init__("Redes Neuronales Artificiales")
        self.model = MLPRegressor(random_state=42, max_iter=1000)
        
    def train(self, X_train, y_train):
        """
        Entrena el modelo de red neuronal
        """
        print(f"🔄 Entrenando {self.model_name}...")
        
        # Optimización de hiperparámetros
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        print(f"   Mejores parámetros: {grid_search.best_params_}")
        print(f"   Mejor score: {-grid_search.best_score_:.4f}")
        
        self.is_trained = True
        
        # Para redes neuronales, la importancia no es directa
        # Usaremos una aproximación basada en la sensibilidad
        self.feature_importance = self._calculate_feature_importance(X_train, y_train)
        
        print(f"✅ {self.model_name} entrenado exitosamente")
    
    def _calculate_feature_importance(self, X_train, y_train):
        """
        Calcula la importancia de características para redes neuronales
        """
        # Método de permutación para calcular importancia
        baseline_score = self.model.score(X_train, y_train)
        importances = []
        
        for i in range(X_train.shape[1]):
            X_permuted = X_train.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = self.model.score(X_permuted, y_train)
            importance = baseline_score - permuted_score
            importances.append(importance)
        
        return np.array(importances)
    
    def plot_training_history(self):
        """
        Grafica el historial de entrenamiento
        """
        if not self.is_trained:
            print("❌ El modelo debe ser entrenado primero")
            return
        
        if hasattr(self.model, 'loss_curve_'):
            plt.figure(figsize=(10, 6))
            plt.plot(self.model.loss_curve_, linewidth=2)
            plt.title(f'📈 Curva de Pérdida - {self.model_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Épocas', fontweight='bold')
            plt.ylabel('Pérdida', fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("ℹ️  No se dispone del historial de entrenamiento para este modelo")

# Importar pandas para la función de conversión
import pandas as pd
