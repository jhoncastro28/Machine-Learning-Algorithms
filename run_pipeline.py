#!/usr/bin/env python3
"""
Script de ejecución del pipeline batch reproducible
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional

Uso:
    python run_pipeline.py [config.json]
"""

import sys
import os
from pathlib import Path

# Configurar codificación UTF-8 para Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from cli.run_batch import run_batch

def main():
    """
    Función principal para ejecutar el pipeline
    """
    # Obtener archivo de configuración (opcional)
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Verificar que el archivo de configuración existe
    if not os.path.exists(config_path):
        print(f"[ERROR] Archivo de configuración no encontrado: {config_path}")
        print("[INFO] Usando configuración por defecto...")
        config_path = "config.json"
    
    print("[INICIO] Iniciando pipeline batch reproducible...")
    print(f"[CONFIG] Configuración: {config_path}")
    
    # Ejecutar pipeline
    success = run_batch(config_path)
    
    if success:
        print("\n[EXITO] Pipeline ejecutado exitosamente!")
        print("[RESULTADOS] Revisa los resultados en:")
        print("   • reports/tables/ - Tablas de comparación y predicciones")
        print("   • reports/figures/ - Gráficos y visualizaciones")
        print("   • models_store/ - Modelos entrenados")
        return 0
    else:
        print("\n[ERROR] Pipeline falló!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
