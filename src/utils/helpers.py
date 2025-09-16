"""
Funciones auxiliares
"""

import matplotlib.pyplot as plt
import seaborn as sns
from .constants import MATPLOTLIB_CONFIG

def setup_matplotlib():
    """
    Configura matplotlib para gráficos en español
    """
    plt.rcParams['font.size'] = MATPLOTLIB_CONFIG['font_size']
    plt.rcParams['figure.figsize'] = MATPLOTLIB_CONFIG['figure_size']
    sns.set_style(MATPLOTLIB_CONFIG['style'])
    sns.set_palette(MATPLOTLIB_CONFIG['palette'])

def format_number(number, decimals=2):
    """
    Formatea un número con separadores de miles
    """
    return f"{number:,.{decimals}f}"

def format_percentage(number, decimals=2):
    """
    Formatea un número como porcentaje
    """
    return f"{number:.{decimals}f}%"
