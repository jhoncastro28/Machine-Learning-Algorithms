"""
Módulo de manejo de datos para el proyecto de Machine Learning.
Contiene funciones para cargar, preprocesar y dividir datos.
"""

from .data_loader import (
    load_dataset,
    split_features_target,
    train_test_split,
    build_preprocessor,
    fit_transform,
    transform,
    save_artifact,
    load_artifact
)

__all__ = [
    'load_dataset',
    'split_features_target', 
    'train_test_split',
    'build_preprocessor',
    'fit_transform',
    'transform',
    'save_artifact',
    'load_artifact'
]
