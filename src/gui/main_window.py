"""
Interfaz Gráfica para Análisis de Machine Learning - Ingresos de Cafeterías
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import sys
import pandas as pd
import numpy as np

# Importaciones locales
from ..core.data_handler import DataHandler
from ..core.model_comparator import ModelComparator
from ..models import (
    LogisticRegressionModel, SVMModel, DecisionTreeModel, 
    RandomForestModel, NeuralNetworkModel
)
from ..utils.constants import COLORS, STYLES, FILES
from ..utils.helpers import ensure_models_dir, load_artifact, save_artifact, is_artifact_valid

class CoffeeShopMLGUI:
    """
    Interfaz gráfica principal para el análisis de Machine Learning
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_main_window()
        self.setup_styles()
        self.setup_variables()
        self.setup_main_interface()
        
        # Inicializar componentes
        self.data_handler = None
        self.models = {}
        self.comparator = ModelComparator()
        
    def setup_main_window(self):
        """
        Configura la ventana principal
        """
        self.root.title("☕ Análisis ML - Ingresos de Cafeterías | UPTC")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Centrar la ventana
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (800 // 2)
        self.root.geometry(f"1200x800+{x}+{y}")
        
        # Configurar el icono (si existe)
        try:
            self.root.iconbitmap("coffee.ico")
        except:
            pass
        
        # Configurar el cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """
        Configura los estilos y colores
        """
        self.colors = COLORS
        
        # Configurar estilos de ttk
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar estilos personalizados
        style.configure(STYLES['title'], 
                       font=('Arial', 16, 'bold'),
                       foreground=self.colors['primary'])
        
        style.configure(STYLES['header'],
                       font=('Arial', 12, 'bold'),
                       foreground=self.colors['dark'])
        
        style.configure(STYLES['info'],
                       font=('Arial', 10),
                       foreground=self.colors['text'])
        
        style.configure(STYLES['primary_button'],
                       font=('Arial', 10, 'bold'),
                       background=self.colors['primary'],
                       foreground='white')
        
        style.configure(STYLES['success_button'],
                       font=('Arial', 10, 'bold'),
                       background=self.colors['success'],
                       foreground='white')
        
        style.configure(STYLES['accent_button'],
                       font=('Arial', 10, 'bold'),
                       background=self.colors['accent'],
                       foreground='white')
    
    def setup_variables(self):
        """
        Configura las variables de control
        """
        self.data_loaded = tk.BooleanVar(value=False)
        self.data_prepared = tk.BooleanVar(value=False)
        self.models_trained = tk.BooleanVar(value=False)
        self.analysis_complete = tk.BooleanVar(value=False)
        
        # Variables para el progreso
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Listo para comenzar")
    
    def setup_main_interface(self):
        """
        Configura la interfaz principal
        """
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Header
        self.create_header(main_frame)
        
        # Panel de estado
        self.create_status_panel(main_frame)
        
        # Panel de botones principales
        self.create_main_buttons(main_frame)
        
        # Panel de visualización
        self.create_visualization_panel(main_frame)
        
        # Panel de resultados
        self.create_results_panel(main_frame)
        
        # Barra de progreso
        self.create_progress_bar(main_frame)
    
    def create_header(self, parent):
        """
        Crea el encabezado de la aplicación
        """
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Título principal
        title_label = ttk.Label(header_frame, 
                               text="☕ Análisis de Machine Learning - Ingresos de Cafeterías",
                               style=STYLES['title'])
        title_label.pack()
        
        # Subtítulo
        subtitle_label = ttk.Label(header_frame,
                                  text="Universidad Pedagógica y Tecnológica de Colombia | Inteligencia Computacional",
                                  style=STYLES['info'])
        subtitle_label.pack()
        
        # Línea separadora
        separator = ttk.Separator(header_frame, orient='horizontal')
        separator.pack(fill='x', pady=(10, 0))
    
    def create_status_panel(self, parent):
        """
        Crea el panel de estado
        """
        status_frame = ttk.LabelFrame(parent, text="📊 Estado del Análisis", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Estados
        states = [
            ("📁 Datos cargados", self.data_loaded),
            ("🔧 Datos preparados", self.data_prepared),
            ("🤖 Modelos entrenados", self.models_trained),
            ("✅ Análisis completo", self.analysis_complete)
        ]
        
        for i, (text, var) in enumerate(states):
            state_label = ttk.Label(status_frame, text=text, style='Info.TLabel')
            state_label.grid(row=0, column=i*2, sticky=tk.W, padx=(0, 5))
            
            status_indicator = ttk.Label(status_frame, text="❌", font=('Arial', 12))
            status_indicator.grid(row=0, column=i*2+1, sticky=tk.W, padx=(0, 20))
            
            # Guardar referencia para actualizar
            setattr(self, f'status_indicator_{i}', status_indicator)
            var.trace('w', lambda *args, idx=i: self.update_status_indicator(idx))
    
    def create_main_buttons(self, parent):
        """
        Crea los botones principales
        """
        buttons_frame = ttk.Frame(parent)
        buttons_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Título del panel
        ttk.Label(buttons_frame, text="🎮 Controles Principales", style=STYLES['header']).pack(anchor='w', pady=(0, 10))
        
        # Botones principales
        buttons = [
            ("📊 Cargar y Explorar Datos", self.load_and_explore_data, STYLES['primary_button']),
            ("🔧 Preparar Datos", self.prepare_data, STYLES['primary_button']),
            ("🤖 Entrenar Modelos", self.train_models, STYLES['success_button']),
            ("📈 Comparar Modelos", self.compare_models, STYLES['accent_button']),
            ("🎯 Visualizar Resultados", self.visualize_results, STYLES['accent_button']),
            ("📋 Generar Reporte", self.generate_report, STYLES['success_button']),
            ("💾 Guardar Resultados", self.save_results, STYLES['primary_button']),
            ("🚀 Análisis Completo", self.run_complete_analysis, STYLES['success_button'])
        ]
        
        for text, command, style in buttons:
            btn = ttk.Button(buttons_frame, text=text, command=command, style=style, width=25)
            btn.pack(pady=5, fill='x')
        
        # Botón de salir
        exit_btn = ttk.Button(buttons_frame, text="❌ Salir", command=self.on_closing, width=25)
        exit_btn.pack(pady=(20, 0), fill='x')
    
    def create_visualization_panel(self, parent):
        """
        Crea el panel de visualización
        """
        viz_frame = ttk.LabelFrame(parent, text="📊 Visualizaciones", padding="10")
        viz_frame.grid(row=2, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Canvas para matplotlib
        self.viz_canvas = None
        self.current_figure = None
        
        # Label inicial
        self.viz_label = ttk.Label(viz_frame, 
                                  text="Las visualizaciones aparecerán aquí\n\nSelecciona una opción del menú para comenzar",
                                  style=STYLES['info'],
                                  anchor='center')
        self.viz_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def create_results_panel(self, parent):
        """
        Crea el panel de resultados
        """
        results_frame = ttk.LabelFrame(parent, text="📋 Resultados y Logs", padding="10")
        results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Text widget con scroll
        self.results_text = scrolledtext.ScrolledText(results_frame, 
                                                     height=15, 
                                                     width=50,
                                                     font=('Consolas', 9))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Botones para el panel de resultados
        results_buttons = ttk.Frame(results_frame)
        results_buttons.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(results_buttons, text="🗑️ Limpiar", command=self.clear_results).pack(side='left', padx=(0, 5))
        ttk.Button(results_buttons, text="💾 Guardar Log", command=self.save_log).pack(side='left')
    
    def create_progress_bar(self, parent):
        """
        Crea la barra de progreso
        """
        progress_frame = ttk.Frame(parent)
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        progress_frame.columnconfigure(1, weight=1)
        
        # Label de estado
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var, style=STYLES['info'])
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Barra de progreso
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                          variable=self.progress_var,
                                          maximum=100,
                                          length=400)
        self.progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
    
    def update_status_indicator(self, index):
        """
        Actualiza los indicadores de estado
        """
        states = [self.data_loaded, self.data_prepared, self.models_trained, self.analysis_complete]
        indicator = getattr(self, f'status_indicator_{index}')
        
        if states[index].get():
            indicator.config(text="✅", foreground='green')
        else:
            indicator.config(text="❌", foreground='red')
    
    def log_message(self, message, level="INFO"):
        """
        Añade un mensaje al log
        """
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if level == "ERROR":
            color = "red"
        elif level == "SUCCESS":
            color = "green"
        elif level == "WARNING":
            color = "orange"
        else:
            color = "black"
        
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_results(self):
        """
        Limpia el panel de resultados
        """
        self.results_text.delete(1.0, tk.END)
    
    def save_log(self):
        """
        Guarda el log en un archivo
        """
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                self.log_message(f"Log guardado en: {filename}", "SUCCESS")
            except Exception as e:
                self.log_message(f"Error al guardar log: {e}", "ERROR")
    
    def show_visualization(self, figure):
        """
        Muestra una visualización en el panel
        """
        # Limpiar visualización anterior
        if self.viz_canvas:
            self.viz_canvas.get_tk_widget().destroy()
        
        # Crear nuevo canvas
        self.viz_canvas = FigureCanvasTkAgg(figure, self.viz_frame)
        self.viz_canvas.draw()
        self.viz_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Ocultar label inicial
        self.viz_label.grid_remove()
    
    def load_and_explore_data(self):
        """
        Carga y explora los datos
        """
        def load_data_thread():
            try:
                self.status_var.set("Cargando datos...")
                self.progress_var.set(10)
                
                # Verificar si el archivo existe
                csv_path = FILES['csv_data']
                if not os.path.exists(csv_path):
                    self.log_message("❌ Error: No se encontró el archivo coffee_shop_revenue.csv", "ERROR")
                    self.log_message("   Asegúrate de que el archivo esté en el directorio actual", "ERROR")
                    return
                
                self.data_handler = DataHandler(csv_path)
                success = self.data_handler.load_data()
                
                if success:
                    self.progress_var.set(50)
                    self.log_message("✅ Datos cargados exitosamente", "SUCCESS")
                    
                    # Explorar datos
                    self.status_var.set("Explorando datos...")
                    self.data_handler.explore_data()
                    
                    self.progress_var.set(80)
                    self.log_message("✅ Exploración de datos completada", "SUCCESS")
                    
                    # Mostrar visualizaciones
                    self.status_var.set("Generando visualizaciones...")
                    self.data_handler.visualize_data()
                    
                    # Mostrar la última figura generada
                    if hasattr(plt, 'gcf'):
                        self.show_visualization(plt.gcf())
                    
                    self.data_loaded.set(True)
                    self.progress_var.set(100)
                    self.status_var.set("Datos cargados y explorados")
                    self.log_message("🎉 Proceso de carga completado exitosamente", "SUCCESS")
                else:
                    self.log_message("❌ Error al cargar los datos", "ERROR")
                    
            except Exception as e:
                self.log_message(f"❌ Error inesperado: {e}", "ERROR")
            finally:
                self.progress_var.set(0)
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=load_data_thread)
        thread.daemon = True
        thread.start()
    
    def prepare_data(self):
        """
        Prepara los datos para entrenamiento
        """
        if not self.data_loaded.get():
            messagebox.showerror("Error", "Primero debe cargar los datos")
            return
        
        def prepare_data_thread():
            try:
                self.status_var.set("Preparando datos...")
                self.progress_var.set(50)
                
                success = self.data_handler.prepare_data()
                
                if success:
                    self.data_prepared.set(True)
                    self.progress_var.set(100)
                    self.status_var.set("Datos preparados")
                    self.log_message("✅ Datos preparados exitosamente", "SUCCESS")
                else:
                    self.log_message("❌ Error al preparar los datos", "ERROR")
                    
            except Exception as e:
                self.log_message(f"❌ Error inesperado: {e}", "ERROR")
            finally:
                self.progress_var.set(0)
        
        thread = threading.Thread(target=prepare_data_thread)
        thread.daemon = True
        thread.start()
    
    def train_models(self):
        """
        Entrena todos los modelos
        """
        if not self.data_prepared.get():
            messagebox.showerror("Error", "Primero debe preparar los datos")
            return
        
        def train_models_thread():
            try:
                self.status_var.set("Entrenando modelos...")
                self.log_message("🤖 Iniciando entrenamiento de modelos...", "INFO")
                
                # Obtener datos
                X_train, X_test, y_train, y_test = self.data_handler.get_data()
                
                # Inicializar modelos
                self.models = {
                    "Regresión Logística": LogisticRegressionModel(),
                    "Máquinas de Vector de Soporte": SVMModel(),
                    "Árboles de Decisión": DecisionTreeModel(),
                    "Random Forest": RandomForestModel(),
                    "Redes Neuronales": NeuralNetworkModel()
                }
                ensure_models_dir()
                model_pkls = FILES.get('model_pkls', {})
                source_files = [FILES.get('csv_data', 'coffee_shop_revenue.csv')]
                
                total_models = len(self.models)
                
                for i, (name, model) in enumerate(self.models.items()):
                    self.status_var.set(f"Entrenando {name}...")
                    self.log_message(f"🔄 Entrenando {name}...", "INFO")
                    
                    try:
                        pkl_path = model_pkls.get(name)
                        loaded = False
                        if pkl_path and is_artifact_valid(pkl_path, source_files):
                            loaded_model, metadata = load_artifact(pkl_path)
                            if loaded_model is not None:
                                model = loaded_model
                                self.models[name] = model
                                loaded = True
                                self.log_message(f"📦 {name}: cargado desde cache", "INFO")
                                print(f"[MODEL] Loaded from cache: {name} -> {pkl_path}")
                        if not loaded:
                            model.train(X_train, y_train)
                            # Guardar modelo
                            if pkl_path:
                                save_artifact(model, pkl_path, metadata={
                                    'model_name': name
                                })
                                print(f"[MODEL] Trained and saved: {name} -> {pkl_path}")
                            self.log_message(f"✅ {name} entrenado y guardado", "SUCCESS")
                        # Añadir al comparador
                        self.comparator.add_model(model, name)
                    except Exception as e:
                        self.log_message(f"❌ Error entrenando/cargando {name}: {e}", "ERROR")
                    
                    # Actualizar progreso
                    progress = ((i + 1) / total_models) * 100
                    self.progress_var.set(progress)
                
                self.models_trained.set(True)
                self.status_var.set("Modelos entrenados")
                self.log_message("🎉 Entrenamiento de todos los modelos completado", "SUCCESS")
                
            except Exception as e:
                self.log_message(f"❌ Error inesperado: {e}", "ERROR")
            finally:
                self.progress_var.set(0)
        
        thread = threading.Thread(target=train_models_thread)
        thread.daemon = True
        thread.start()
    
    def compare_models(self):
        """
        Compara los modelos entrenados
        """
        if not self.models_trained.get():
            messagebox.showerror("Error", "Primero debe entrenar los modelos")
            return
        
        def compare_models_thread():
            try:
                self.status_var.set("Comparando modelos...")
                self.log_message("📊 Iniciando comparación de modelos...", "INFO")
                
                # Obtener datos de prueba
                _, _, _, y_test = self.data_handler.get_data()
                X_test_scaled = self.data_handler.X_test_scaled
                
                # Evaluar modelos
                self.comparator.evaluate_all_models(X_test_scaled, y_test)
                
                # Mostrar tabla de comparación
                self.log_message("📋 Tabla de Comparación de Modelos:", "INFO")
                self.log_message("=" * 80, "INFO")
                
                if self.comparator.comparison_df is not None:
                    for _, row in self.comparator.comparison_df.iterrows():
                        self.log_message(f"Modelo: {row['Modelo']}", "INFO")
                        self.log_message(f"  MSE: {row['MSE']:.4f}", "INFO")
                        self.log_message(f"  RMSE: {row['RMSE']:.4f}", "INFO")
                        self.log_message(f"  MAE: {row['MAE']:.4f}", "INFO")
                        self.log_message(f"  R²: {row['R²']:.4f}", "INFO")
                        self.log_message(f"  MAPE: {row['MAPE (%)']:.2f}%", "INFO")
                        self.log_message("-" * 40, "INFO")
                
                self.status_var.set("Modelos comparados")
                self.log_message("✅ Comparación de modelos completada", "SUCCESS")
                
            except Exception as e:
                self.log_message(f"❌ Error inesperado: {e}", "ERROR")
            finally:
                self.progress_var.set(0)
        
        thread = threading.Thread(target=compare_models_thread)
        thread.daemon = True
        thread.start()
    
    def visualize_results(self):
        """
        Visualiza los resultados
        """
        if not self.models_trained.get():
            messagebox.showerror("Error", "Primero debe entrenar los modelos")
            return
        
        def visualize_results_thread():
            try:
                self.status_var.set("Generando visualizaciones...")
                self.log_message("🎯 Generando visualizaciones de resultados...", "INFO")
                
                # Obtener datos de prueba
                _, _, _, y_test = self.data_handler.get_data()
                
                # Generar gráficos de comparación
                self.comparator.plot_comparison_metrics()
                self.show_visualization(plt.gcf())
                
                self.log_message("✅ Visualizaciones generadas exitosamente", "SUCCESS")
                self.status_var.set("Visualizaciones generadas")
                
            except Exception as e:
                self.log_message(f"❌ Error inesperado: {e}", "ERROR")
            finally:
                self.progress_var.set(0)
        
        thread = threading.Thread(target=visualize_results_thread)
        thread.daemon = True
        thread.start()
    
    def generate_report(self):
        """
        Genera un reporte completo
        """
        if not self.models_trained.get():
            messagebox.showerror("Error", "Primero debe entrenar los modelos")
            return
        
        def generate_report_thread():
            try:
                self.status_var.set("Generando reporte...")
                self.log_message("📋 Generando reporte completo...", "INFO")
                
                # Generar reporte
                self.comparator.generate_report()
                
                self.log_message("✅ Reporte generado exitosamente", "SUCCESS")
                self.status_var.set("Reporte generado")
                
            except Exception as e:
                self.log_message(f"❌ Error inesperado: {e}", "ERROR")
            finally:
                self.progress_var.set(0)
        
        thread = threading.Thread(target=generate_report_thread)
        thread.daemon = True
        thread.start()
    
    def save_results(self):
        """
        Guarda los resultados
        """
        if not self.models_trained.get():
            messagebox.showerror("Error", "Primero debe entrenar los modelos")
            return
        
        def save_results_thread():
            try:
                self.status_var.set("Guardando resultados...")
                self.log_message("💾 Guardando resultados...", "INFO")
                
                # Guardar resultados
                self.comparator.save_results()
                
                # Guardar predicciones
                _, _, _, y_test = self.data_handler.get_data()
                X_test_scaled = self.data_handler.X_test_scaled
                
                predictions_df = pd.DataFrame({'Valores_Reales': y_test})
                
                for name, model in self.models.items():
                    try:
                        pred = model.predict(X_test_scaled)
                        predictions_df[f'Predicciones_{name.replace(" ", "_")}'] = pred
                    except Exception as e:
                        self.log_message(f"⚠️ Error guardando predicciones de {name}: {e}", "WARNING")
                
                predictions_df.to_csv(FILES['predictions_csv'], index=False)
                
                self.log_message("✅ Resultados guardados exitosamente", "SUCCESS")
                self.log_message("📁 Archivos generados:", "INFO")
                self.log_message(f"   • {FILES['results_csv']}", "INFO")
                self.log_message(f"   • {FILES['predictions_csv']}", "INFO")
                self.status_var.set("Resultados guardados")
                
            except Exception as e:
                self.log_message(f"❌ Error inesperado: {e}", "ERROR")
            finally:
                self.progress_var.set(0)
        
        thread = threading.Thread(target=save_results_thread)
        thread.daemon = True
        thread.start()
    
    def run_complete_analysis(self):
        """
        Ejecuta el análisis completo
        """
        def complete_analysis_thread():
            try:
                self.status_var.set("Ejecutando análisis completo...")
                self.log_message("🚀 Iniciando análisis completo...", "INFO")
                
                # Paso 1: Cargar datos
                if not self.data_loaded.get():
                    self.load_and_explore_data()
                    # Esperar a que termine
                    while not self.data_loaded.get():
                        self.root.update()
                        time.sleep(0.1)
                
                # Paso 2: Preparar datos
                if not self.data_prepared.get():
                    self.prepare_data()
                    while not self.data_prepared.get():
                        self.root.update()
                        time.sleep(0.1)
                
                # Paso 3: Entrenar modelos
                if not self.models_trained.get():
                    self.train_models()
                    while not self.models_trained.get():
                        self.root.update()
                        time.sleep(0.1)
                
                # Paso 4: Comparar modelos
                self.compare_models()
                time.sleep(2)  # Dar tiempo para que termine
                
                # Paso 5: Visualizar resultados
                self.visualize_results()
                time.sleep(2)
                
                # Paso 6: Generar reporte
                self.generate_report()
                time.sleep(1)
                
                # Paso 7: Guardar resultados
                self.save_results()
                time.sleep(1)
                
                self.analysis_complete.set(True)
                self.status_var.set("Análisis completo finalizado")
                self.log_message("🎉 ¡Análisis completo finalizado exitosamente!", "SUCCESS")
                
            except Exception as e:
                self.log_message(f"❌ Error inesperado: {e}", "ERROR")
            finally:
                self.progress_var.set(0)
        
        thread = threading.Thread(target=complete_analysis_thread)
        thread.daemon = True
        thread.start()
    
    def on_closing(self):
        """
        Maneja el cierre de la aplicación
        """
        if messagebox.askokcancel("Salir", "¿Estás seguro de que quieres salir?"):
            self.root.destroy()
    
    def run(self):
        """
        Ejecuta la aplicación
        """
        self.log_message("🚀 Aplicación iniciada", "SUCCESS")
        self.log_message("📋 Selecciona una opción del menú para comenzar", "INFO")
        self.root.mainloop()

def main():
    """
    Función principal
    """
    try:
        app = CoffeeShopMLGUI()
        app.run()
    except Exception as e:
        print(f"Error fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import time
    main()
