"""
Implementación de Redes Neuronales Artificiales para regresión
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from .base_model import BaseMLModel

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
