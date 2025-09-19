"""
Módulo para generar metadatos de ejecución y reproducibilidad
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional
"""

import json
import hashlib
import os
import sys
import platform
from datetime import datetime
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd


def get_package_versions():
    """
    Obtiene las versiones de los paquetes principales utilizados
    
    Returns:
        dict: Diccionario con las versiones de los paquetes
    """
    packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scikit-learn', 'joblib'
    ]
    
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "No instalado"
    
    return versions


def calculate_dataset_hash(file_path):
    """
    Calcula el hash SHA-256 del archivo de dataset
    
    Args:
        file_path (str): Ruta al archivo de dataset
        
    Returns:
        str: Hash SHA-256 del archivo
    """
    if not os.path.exists(file_path):
        return "Archivo no encontrado"
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def get_system_info():
    """
    Obtiene información del sistema
    
    Returns:
        dict: Información del sistema
    """
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "machine": platform.machine()
    }


def generate_run_metadata(config_path, dataset_path, output_dir="reports"):
    """
    Genera metadatos completos de la ejecución
    
    Args:
        config_path (str): Ruta al archivo de configuración
        dataset_path (str): Ruta al archivo de dataset
        output_dir (str): Directorio de salida para los metadatos
        
    Returns:
        dict: Metadatos de la ejecución
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener timestamp
    timestamp = datetime.now().isoformat()
    
    # Cargar configuración
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # Generar metadatos
    metadata = {
        "execution_info": {
            "timestamp": timestamp,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "config_file": config_path,
            "dataset_file": dataset_path
        },
        "system_info": get_system_info(),
        "package_versions": get_package_versions(),
        "configuration": config,
        "dataset_info": {
            "file_path": dataset_path,
            "file_size_bytes": os.path.getsize(dataset_path) if os.path.exists(dataset_path) else 0,
            "sha256_hash": calculate_dataset_hash(dataset_path),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(dataset_path)).isoformat() if os.path.exists(dataset_path) else None
        },
        "reproducibility": {
            "numpy_random_seed": "Fijo en configuración",
            "python_random_seed": "Fijo en configuración", 
            "sklearn_random_state": config.get('preprocessing', {}).get('random_state', 42),
            "note": "Todos los random_state están fijados para reproducibilidad"
        }
    }
    
    # Guardar metadatos
    metadata_path = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Metadatos de ejecución guardados en: {metadata_path}")
    
    return metadata


def set_global_random_seeds(random_state=42):
    """
    Establece semillas globales para reproducibilidad
    
    Args:
        random_state (int): Semilla para reproducibilidad
    """
    # Establecer semilla de numpy
    np.random.seed(random_state)
    
    # Establecer semilla de Python (para random)
    import random
    random.seed(random_state)
    
    print(f"🔒 Semillas globales establecidas con random_state={random_state}")


def validate_reproducibility_setup(config):
    """
    Valida que la configuración tenga todos los random_state necesarios
    
    Args:
        config (dict): Configuración del proyecto
        
    Returns:
        bool: True si la configuración es reproducible
    """
    required_random_states = [
        'preprocessing.random_state',
        'training.random_state'
    ]
    
    missing_states = []
    for state_path in required_random_states:
        keys = state_path.split('.')
        current = config
        try:
            for key in keys:
                current = current[key]
            if current is None:
                missing_states.append(state_path)
        except (KeyError, TypeError):
            missing_states.append(state_path)
    
    if missing_states:
        print(f"⚠️  Random states faltantes: {missing_states}")
        return False
    
    print("✅ Configuración de reproducibilidad validada")
    return True
