"""
Aplicación principal para análisis de Machine Learning de ingresos de cafeterías
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional

Autor: [Tu nombre]
Fecha: [Fecha actual]
"""

import os
import sys
import time
from data_handler import DataHandler
from ml_models import (
    LogisticRegressionModel, SVMModel, DecisionTreeModel, 
    RandomForestModel, NeuralNetworkModel
)
from model_comparator import ModelComparator

class CoffeeShopMLApp:
    """
    Aplicación principal para el análisis de Machine Learning
    """
    
    def __init__(self):
        self.data_handler = None
        self.models = {}
        self.comparator = ModelComparator()
        self.csv_path = "coffee_shop_revenue.csv"
        
    def print_header(self):
        """
        Imprime el encabezado de la aplicación
        """
        print("\n" + "="*80)
        print("☕ ANÁLISIS DE MACHINE LEARNING - INGRESOS DE CAFETERÍAS ☕")
        print("="*80)
        print("Universidad Pedagógica y Tecnológica de Colombia")
        print("Facultad de Ingeniería - Escuela de Ingeniería de Sistemas y Computación")
        print("Inteligencia Computacional")
        print("="*80)
    
    def print_menu(self):
        """
        Imprime el menú principal
        """
        print("\n📋 MENÚ PRINCIPAL:")
        print("1. 📊 Explorar y visualizar datos")
        print("2. 🔧 Preparar datos para entrenamiento")
        print("3. 🤖 Entrenar todos los modelos")
        print("4. 📈 Comparar modelos")
        print("5. 🎯 Visualizar resultados")
        print("6. 📋 Generar reporte completo")
        print("7. 💾 Guardar resultados")
        print("8. 🚀 Ejecutar análisis completo")
        print("0. ❌ Salir")
        print("-" * 50)
    
    def load_data(self):
        """
        Carga los datos desde el archivo CSV
        """
        print("\n🔄 Cargando datos...")
        
        if not os.path.exists(self.csv_path):
            print(f"❌ Error: No se encontró el archivo {self.csv_path}")
            print("   Asegúrate de que el archivo esté en el directorio actual")
            return False
        
        self.data_handler = DataHandler(self.csv_path)
        success = self.data_handler.load_data()
        
        if success:
            print("✅ Datos cargados exitosamente")
            return True
        else:
            print("❌ Error al cargar los datos")
            return False
    
    def explore_data(self):
        """
        Explora y visualiza los datos
        """
        if self.data_handler is None:
            print("❌ Primero debe cargar los datos")
            return
        
        print("\n🔍 Explorando datos...")
        self.data_handler.explore_data()
        
        print("\n📊 Generando visualizaciones...")
        self.data_handler.visualize_data()
    
    def prepare_data(self):
        """
        Prepara los datos para el entrenamiento
        """
        if self.data_handler is None:
            print("❌ Primero debe cargar los datos")
            return
        
        print("\n🔧 Preparando datos...")
        success = self.data_handler.prepare_data()
        
        if success:
            print("✅ Datos preparados exitosamente")
            return True
        else:
            print("❌ Error al preparar los datos")
            return False
    
    def train_models(self):
        """
        Entrena todos los modelos de Machine Learning
        """
        if self.data_handler is None or self.data_handler.X_train_scaled is None:
            print("❌ Primero debe cargar y preparar los datos")
            return
        
        print("\n🤖 Entrenando modelos de Machine Learning...")
        
        # Obtener datos preparados
        X_train, X_test, y_train, y_test = self.data_handler.get_data()
        
        # Inicializar modelos
        self.models = {
            "Regresión Logística": LogisticRegressionModel(),
            "Máquinas de Vector de Soporte": SVMModel(),
            "Árboles de Decisión": DecisionTreeModel(),
            "Random Forest": RandomForestModel(),
            "Redes Neuronales": NeuralNetworkModel()
        }
        
        # Entrenar cada modelo
        for name, model in self.models.items():
            try:
                print(f"\n🔄 Entrenando {name}...")
                start_time = time.time()
                
                model.train(X_train, y_train)
                
                end_time = time.time()
                training_time = end_time - start_time
                
                print(f"   ⏱️  Tiempo de entrenamiento: {training_time:.2f} segundos")
                
                # Añadir al comparador
                self.comparator.add_model(model, name)
                
            except Exception as e:
                print(f"❌ Error entrenando {name}: {e}")
        
        print(f"\n✅ Entrenamiento completado para {len(self.models)} modelos")
    
    def compare_models(self):
        """
        Compara todos los modelos entrenados
        """
        if not self.models:
            print("❌ Primero debe entrenar los modelos")
            return
        
        print("\n📊 Comparando modelos...")
        
        # Obtener datos de prueba
        _, _, _, y_test = self.data_handler.get_data()
        X_test_scaled = self.data_handler.X_test_scaled
        
        # Evaluar todos los modelos
        self.comparator.evaluate_all_models(X_test_scaled, y_test)
        
        # Mostrar tabla de comparación
        self.comparator.create_comparison_table()
    
    def visualize_results(self):
        """
        Visualiza los resultados de todos los modelos
        """
        if not self.models:
            print("❌ Primero debe entrenar los modelos")
            return
        
        print("\n🎯 Generando visualizaciones...")
        
        # Obtener datos de prueba
        _, _, _, y_test = self.data_handler.get_data()
        
        # Gráficos de comparación de métricas
        self.comparator.plot_comparison_metrics()
        
        # Gráficos de predicciones vs reales
        self.comparator.plot_predictions_vs_actual(y_test)
        
        # Gráficos de importancia de características
        if self.data_handler.feature_names:
            self.comparator.plot_feature_importance(self.data_handler.feature_names)
        
        # Visualización específica del árbol de decisión
        if "Árboles de Decisión" in self.models:
            print("\n🌳 Visualizando árbol de decisión...")
            self.models["Árboles de Decisión"].visualize_tree(
                self.data_handler.feature_names, max_depth=3
            )
        
        # Curva de entrenamiento de la red neuronal
        if "Redes Neuronales" in self.models:
            print("\n📈 Visualizando curva de entrenamiento de red neuronal...")
            self.models["Redes Neuronales"].plot_training_history()
    
    def generate_report(self):
        """
        Genera un reporte completo del análisis
        """
        if not self.models:
            print("❌ Primero debe entrenar los modelos")
            return
        
        print("\n📋 Generando reporte completo...")
        self.comparator.generate_report()
    
    def save_results(self):
        """
        Guarda los resultados en archivos
        """
        if not self.models:
            print("❌ No hay resultados para guardar")
            return
        
        print("\n💾 Guardando resultados...")
        
        # Guardar tabla de comparación
        self.comparator.save_results("resultados_comparacion_modelos.csv")
        
        # Guardar predicciones de cada modelo
        _, _, _, y_test = self.data_handler.get_data()
        X_test_scaled = self.data_handler.X_test_scaled
        
        predictions_df = pd.DataFrame({'Valores_Reales': y_test})
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_test_scaled)
                predictions_df[f'Predicciones_{name.replace(" ", "_")}'] = pred
            except Exception as e:
                print(f"⚠️  Error guardando predicciones de {name}: {e}")
        
        predictions_df.to_csv("predicciones_modelos.csv", index=False)
        print("✅ Predicciones guardadas en 'predicciones_modelos.csv'")
        
        print("✅ Resultados guardados exitosamente")
    
    def run_complete_analysis(self):
        """
        Ejecuta el análisis completo automáticamente
        """
        print("\n🚀 Iniciando análisis completo...")
        
        # Paso 1: Cargar datos
        if not self.load_data():
            return
        
        # Paso 2: Explorar datos
        self.explore_data()
        
        # Paso 3: Preparar datos
        if not self.prepare_data():
            return
        
        # Paso 4: Entrenar modelos
        self.train_models()
        
        # Paso 5: Comparar modelos
        self.compare_models()
        
        # Paso 6: Visualizar resultados
        self.visualize_results()
        
        # Paso 7: Generar reporte
        self.generate_report()
        
        # Paso 8: Guardar resultados
        self.save_results()
        
        print("\n🎉 ¡Análisis completo finalizado exitosamente!")
        print("📁 Revisa los archivos generados:")
        print("   • resultados_comparacion_modelos.csv")
        print("   • predicciones_modelos.csv")
    
    def run(self):
        """
        Ejecuta la aplicación principal
        """
        self.print_header()
        
        while True:
            self.print_menu()
            
            try:
                choice = input("👉 Selecciona una opción (0-8): ").strip()
                
                if choice == "0":
                    print("\n👋 ¡Gracias por usar la aplicación!")
                    print("Universidad Pedagógica y Tecnológica de Colombia")
                    break
                
                elif choice == "1":
                    if self.data_handler is None:
                        if not self.load_data():
                            continue
                    self.explore_data()
                
                elif choice == "2":
                    if self.data_handler is None:
                        if not self.load_data():
                            continue
                    self.prepare_data()
                
                elif choice == "3":
                    if self.data_handler is None:
                        if not self.load_data():
                            continue
                    if self.data_handler.X_train_scaled is None:
                        if not self.prepare_data():
                            continue
                    self.train_models()
                
                elif choice == "4":
                    self.compare_models()
                
                elif choice == "5":
                    self.visualize_results()
                
                elif choice == "6":
                    self.generate_report()
                
                elif choice == "7":
                    self.save_results()
                
                elif choice == "8":
                    self.run_complete_analysis()
                
                else:
                    print("❌ Opción no válida. Por favor, selecciona un número del 0 al 8.")
                
                # Pausa antes de mostrar el menú nuevamente
                input("\n⏸️  Presiona Enter para continuar...")
                
            except KeyboardInterrupt:
                print("\n\n👋 ¡Aplicación interrumpida por el usuario!")
                break
            except Exception as e:
                print(f"\n❌ Error inesperado: {e}")
                input("⏸️  Presiona Enter para continuar...")

def main():
    """
    Función principal
    """
    try:
        app = CoffeeShopMLApp()
        app.run()
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Importar pandas aquí para evitar problemas de importación
    import pandas as pd
    main()
