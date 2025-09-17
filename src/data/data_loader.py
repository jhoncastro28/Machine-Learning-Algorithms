"""
Módulo para cargar, preprocesar y dividir datos.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Union, Any


def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga un dataset desde un archivo CSV.
    
    Args:
        path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Dataset cargado
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si hay errores al cargar el archivo
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"El archivo {path} no existe")
        
        df = pd.read_csv(path)
        print(f"Dataset cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    
    except Exception as e:
        raise ValueError(f"Error al cargar el dataset: {str(e)}")


def split_features_target(df: pd.DataFrame, target: str = 'Daily_Revenue') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa las características (X) del objetivo (y) del dataset.
    
    Args:
        df (pd.DataFrame): Dataset completo
        target (str): Nombre de la columna objetivo
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) donde X son las características e y es el objetivo
        
    Raises:
        ValueError: Si la columna objetivo no existe
    """
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el dataset")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    print(f"Características separadas: {X.shape[1]} columnas")
    print(f"Objetivo separado: {target}")
    
    return X, y


def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X (pd.DataFrame): Características
        y (pd.Series): Objetivo
        test_size (float): Proporción de datos para prueba (default: 0.2)
        random_state (int): Semilla para reproducibilidad (default: 42)
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = sklearn_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Datos divididos:")
    print(f"  Entrenamiento: {X_train.shape[0]} muestras")
    print(f"  Prueba: {X_test.shape[0]} muestras")
    print(f"  Proporción de prueba: {test_size}")
    
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """
    Construye un preprocesador que selecciona y estandariza columnas numéricas.
    
    Returns:
        ColumnTransformer: Preprocesador configurado
    """
    # Pipeline para columnas numéricas
    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # ColumnTransformer que aplica el pipeline solo a columnas numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, 'passthrough')  # 'passthrough' selecciona automáticamente columnas numéricas
        ],
        remainder='passthrough'  # Mantiene columnas no numéricas sin cambios
    )
    
    print("Preprocesador creado: estandarización de columnas numéricas")
    return preprocessor


def fit_transform(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> np.ndarray:
    """
    Ajusta el preprocesador con los datos de entrenamiento y los transforma.
    
    Args:
        preprocessor (ColumnTransformer): Preprocesador
        X_train (pd.DataFrame): Datos de entrenamiento
        
    Returns:
        np.ndarray: Datos de entrenamiento transformados
    """
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    print(f"Datos de entrenamiento transformados: {X_train_transformed.shape}")
    return X_train_transformed


def transform(preprocessor: ColumnTransformer, X_test: pd.DataFrame) -> np.ndarray:
    """
    Transforma los datos de prueba usando el preprocesador ya ajustado.
    
    Args:
        preprocessor (ColumnTransformer): Preprocesador ya ajustado
        X_test (pd.DataFrame): Datos de prueba
        
    Returns:
        np.ndarray: Datos de prueba transformados
    """
    X_test_transformed = preprocessor.transform(X_test)
    
    print(f"Datos de prueba transformados: {X_test_transformed.shape}")
    return X_test_transformed


def save_artifact(obj: Any, name: str) -> None:
    """
    Guarda un objeto (scaler, splits, etc.) en un archivo pickle.
    
    Args:
        obj (Any): Objeto a guardar
        name (str): Nombre del archivo (sin extensión)
    """
    # Crear directorio artifacts si no existe
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    file_path = artifacts_dir / f"{name}.pkl"
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    
    print(f"Artifact guardado: {file_path}")


def load_artifact(name: str) -> Any:
    """
    Carga un objeto desde un archivo pickle.
    
    Args:
        name (str): Nombre del archivo (sin extensión)
        
    Returns:
        Any: Objeto cargado
        
    Raises:
        FileNotFoundError: Si el archivo no existe
    """
    file_path = Path("artifacts") / f"{name}.pkl"
    
    if not file_path.exists():
        raise FileNotFoundError(f"El artifact '{name}' no existe en {file_path}")
    
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    
    print(f"Artifact cargado: {file_path}")
    return obj
