"""
Aplicación principal para análisis de Machine Learning de ingresos de cafeterías
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional

Esta aplicación ahora utiliza únicamente la interfaz gráfica.
"""

import sys
import os

def main():
    """
    Función principal - Ejecuta la interfaz gráfica
    """
    try:
        # Importar y ejecutar la interfaz gráfica
        from src.gui import CoffeeShopMLGUI
        
        print("🚀 Iniciando aplicación de Machine Learning...")
        print("📱 Abriendo interfaz gráfica...")
        
        app = CoffeeShopMLGUI()
        app.run()
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Asegúrate de tener todas las dependencias instaladas:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()