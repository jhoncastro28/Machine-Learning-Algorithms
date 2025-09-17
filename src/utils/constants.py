"""
Constantes del sistema
"""

# Colores de la interfaz
COLORS = {
    'primary': '#2E86AB',      # Azul principal
    'secondary': '#A23B72',    # Rosa/Magenta
    'accent': '#F18F01',       # Naranja
    'success': '#C73E1D',      # Rojo
    'background': '#F8F9FA',   # Gris claro
    'text': '#212529',         # Negro
    'light': '#FFFFFF',        # Blanco
    'dark': '#343A40'          # Gris oscuro
}

# Estilos de la interfaz
STYLES = {
    'title': 'Title.TLabel',
    'header': 'Header.TLabel', 
    'info': 'Info.TLabel',
    'primary_button': 'Primary.TButton',
    'success_button': 'Success.TButton',
    'accent_button': 'Accent.TButton'
}

# Configuración de matplotlib
MATPLOTLIB_CONFIG = {
    'font_size': 10,
    'figure_size': (12, 8),
    'style': 'whitegrid',
    'palette': 'husl'
}

# Configuración de modelos
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# Nombres de archivos
FILES = {
    'csv_data': 'coffee_shop_revenue.csv',
    'results_csv': 'resultados_comparacion_modelos.csv',
    'predictions_csv': 'predicciones_modelos.csv',
    'models_dir': 'models_store',
    'scaler_pkl': 'models_store/scaler.pkl',
    'model_pkls': {
        'Regresión Logística': 'models_store/model_logistic_regression.pkl',
        'Máquinas de Vector de Soporte': 'models_store/model_svm.pkl',
        'Árboles de Decisión': 'models_store/model_decision_tree.pkl',
        'Random Forest': 'models_store/model_random_forest.pkl',
        'Redes Neuronales': 'models_store/model_neural_network.pkl'
    }
}
