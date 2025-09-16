"""
Implementación de Random Forest para regresión
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from .base_model import BaseMLModel

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
