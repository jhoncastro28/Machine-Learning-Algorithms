"""
Módulo de evaluación de modelos
"""

from .metrics import metrics_regression, cv_score, save_results_table, compare_models
from .plotting import plot_metrics_bar, plot_predictions_vs_actual, plot_feature_importance, plot_comprehensive_comparison

__all__ = [
    'metrics_regression', 
    'cv_score', 
    'save_results_table', 
    'compare_models',
    'plot_metrics_bar', 
    'plot_predictions_vs_actual', 
    'plot_feature_importance',
    'plot_comprehensive_comparison'
]
