"""
Implementación de Árboles de Decisión para regresión
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV
from .base_model import BaseMLModel

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
        
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, feature_names=feature_names, 
                 max_depth=max_depth, filled=True, rounded=True, fontsize=10)
        plt.title(f'🌳 Árbol de Decisión - {self.model_name} (Primeros {max_depth} niveles)', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
