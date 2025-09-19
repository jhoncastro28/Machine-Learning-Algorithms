"""
Funciones auxiliares
"""

import matplotlib.pyplot as plt
import seaborn as sns
from .constants import MATPLOTLIB_CONFIG, FILES
import os
import json
import joblib
from datetime import datetime

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

def ensure_models_dir():
    """
    Crea el directorio de modelos si no existe.
    """
    models_dir = FILES.get('models_dir', 'models_store')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        print(f"[CACHE] Created models dir: {models_dir}")
    return models_dir

def save_artifact(obj, path, metadata=None):
    """
    Guarda un artefacto con joblib y un archivo de metadatos opcional.
    """
    ensure_models_dir()
    joblib.dump(obj, path)
    print(f"[CACHE] Saved artifact: {path}")
    if metadata is None:
        metadata = {}
    metadata_payload = {
        **metadata,
        'saved_at': datetime.utcnow().isoformat() + 'Z'
    }
    meta_path = f"{path}.meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2)
    print(f"[CACHE] Wrote metadata: {meta_path}")
    return path

def load_artifact(path):
    """
    Carga un artefacto si existe; retorna (objeto, metadata) o (None, None).
    """
    if not os.path.exists(path):
        print(f"[CACHE] Artifact not found: {path}")
        return None, None
    obj = joblib.load(path)
    print(f"[CACHE] Loaded artifact: {path}")
    meta_path = f"{path}.meta.json"
    metadata = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception:
            metadata = None
    return obj, metadata

def is_artifact_valid(path, source_files=None):
    """
    Verifica si el artefacto existe y es más reciente que las fuentes.
    """
    if not os.path.exists(path):
        return False
    artifact_mtime = os.path.getmtime(path)
    if not source_files:
        return True
    for src in source_files:
        if os.path.exists(src) and os.path.getmtime(src) > artifact_mtime:
            print(f"[CACHE] Invalidated: {path} older than source {src}")
            return False
    print(f"[CACHE] Valid: {path}")
    return True
