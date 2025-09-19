"""
Implementación de Regresión Lineal
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .base_model import BaseMLModel

class LinearRegressionModel(BaseMLModel):
    """
    Implementación de Regresión Lineal
    """
    
    def __init__(self):
        super().__init__("Regresión Lineal")
        self.model = LinearRegression()
        
    def train(self, X_train, y_train):
        """
        Entrena el modelo de regresión lineal
        """
        print(f"🔄 Entrenando {self.model_name}...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calcular importancia de características (coeficientes)
        if hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_)
        
        print(f"✅ {self.model_name} entrenado exitosamente")
    
    def predict(self, X_test):
        """
        Realiza predicciones
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        return self.model.predict(X_test)
