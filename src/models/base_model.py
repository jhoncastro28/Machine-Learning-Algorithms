"""
Clase base para todos los modelos de Machine Learning
"""

import numpy as np
from sklearn.model_selection import cross_val_score
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
