"""
Lanzador simple para la interfaz gráfica
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional
"""

import sys
import os

def main():
    """
    Lanzador principal para la interfaz gráfica
    """
    print("☕ Iniciando Análisis de Machine Learning - Ingresos de Cafeterías")
    print("=" * 70)
    print("Universidad Pedagógica y Tecnológica de Colombia")
    print("Inteligencia Computacional")
    print("=" * 70)
    print()
    
    try:
        # Verificar que el archivo CSV existe
        if not os.path.exists("coffee_shop_revenue.csv"):
            print("❌ Error: No se encontró el archivo 'coffee_shop_revenue.csv'")
            print("   Asegúrate de que el archivo esté en el directorio actual")
            input("Presiona Enter para salir...")
            return
        
        # Importar y ejecutar la GUI
        from src.gui import CoffeeShopMLGUI
        
        print("🚀 Iniciando interfaz gráfica...")
        print("📱 La ventana se abrirá en unos segundos...")
        print()
        
        app = CoffeeShopMLGUI()
        app.run()
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print()
        print("💡 Soluciones posibles:")
        print("   1. Instala las dependencias: pip install -r requirements.txt")
        print("   2. Verifica que tienes Python 3.7 o superior")
        print("   3. Asegúrate de que todos los archivos están en el mismo directorio")
        input("Presiona Enter para salir...")
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        input("Presiona Enter para salir...")

if __name__ == "__main__":
    main()
