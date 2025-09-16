"""
Implementación de Regresión Logística para regresión
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .base_model import BaseMLModel

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
