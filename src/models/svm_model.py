"""
Implementación de Máquinas de Vector de Soporte para regresión
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from .base_model import BaseMLModel

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
