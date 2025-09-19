"""
Interfaz Gráfica para Análisis de Machine Learning - Ingresos de Cafeterías
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg
import threading
import os
import sys
import pandas as pd
import time
from PIL import Image, ImageTk

# Importaciones locales
from ..utils.constants import COLORS, STYLES, FILES

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
        
        # Rutas de artefactos
        self.reports_figures_path = "reports/figures"
        self.comparison_csv_path = "reports/tables/comparison.csv"
        
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
        main_frame.rowconfigure(2, weight=1)
        
        # Header
        self.create_header(main_frame)
        
        # Panel de botones principales
        self.create_main_buttons(main_frame)
        
        # Panel de visualización con pestañas
        self.create_tabbed_interface(main_frame)
        
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
    
    
    def create_main_buttons(self, parent):
        """
        Crea los botones principales
        """
        buttons_frame = ttk.Frame(parent)
        buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Título del panel
        ttk.Label(buttons_frame, text="🎮 Controles Principales", style=STYLES['header']).pack(anchor='w', pady=(0, 10))
        
        # Botones principales simplificados
        buttons = [
            ("🚀 Generar Análisis Batch", self.run_batch_analysis, STYLES['success_button']),
            ("📊 Ver Resultados", self.load_results, STYLES['primary_button'])
        ]
        
        for text, command, style in buttons:
            btn = ttk.Button(buttons_frame, text=text, command=command, style=style, width=25)
            btn.pack(pady=5, fill='x')
        
        # Botón de salir
        exit_btn = ttk.Button(buttons_frame, text="❌ Salir", command=self.on_closing, width=25)
        exit_btn.pack(pady=(20, 0), fill='x')
    
    def create_tabbed_interface(self, parent):
        """
        Crea la interfaz con pestañas para mostrar resultados
        """
        # Frame principal para pestañas
        tab_frame = ttk.Frame(parent)
        tab_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        tab_frame.columnconfigure(0, weight=1)
        tab_frame.rowconfigure(0, weight=1)
        
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(tab_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Crear pestañas
        self.create_tabs()
    
    def create_tabs(self):
        """
        Crea las pestañas para mostrar diferentes tipos de resultados
        """
        # Pestaña EDA
        self.eda_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.eda_frame, text="📊 EDA")
        
        # Pestaña Métricas
        self.metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_frame, text="📈 Métricas")
        
        # Pestaña Predicciones
        self.predictions_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predictions_frame, text="🎯 Predicciones")
        
        # Pestaña Importancias
        self.importance_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.importance_frame, text="🔍 Importancias")
        
        # Pestaña Comparación
        self.comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_frame, text="📋 Comparación")
        
        # Inicializar contenido de pestañas
        self.initialize_tab_content()
    
    def initialize_tab_content(self):
        """
        Inicializa el contenido de las pestañas
        """
        # Configurar frames para scroll
        for frame in [self.eda_frame, self.metrics_frame, self.predictions_frame, 
                     self.importance_frame, self.comparison_frame]:
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            
            # Label inicial
            label = ttk.Label(frame, 
                            text="Haz clic en 'Ver Resultados' para cargar el contenido",
                            style=STYLES['info'],
                            anchor='center')
            label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def create_progress_bar(self, parent):
        """
        Crea la barra de progreso
        """
        progress_frame = ttk.Frame(parent)
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
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
    
    def log_message(self, message, level="INFO"):
        """
        Muestra un mensaje breve en la barra de estado
        """
        self.status_var.set(message)
        self.root.update_idletasks()
        
        if level == "ERROR":
            messagebox.showerror("Error", message)
        elif level == "SUCCESS":
            # Solo actualizar estado, no mostrar popup
            pass
    
    def run_batch_analysis(self):
        """
        Ejecuta el análisis batch completo
        """
        def batch_thread():
            try:
                self.status_var.set("Ejecutando análisis batch...")
                self.progress_var.set(10)
                
                # Importar y ejecutar el pipeline batch
                import subprocess
                import sys
                
                # Ejecutar el pipeline
                result = subprocess.run([sys.executable, "run_pipeline.py"], 
                                      capture_output=True, text=True, cwd=".")
                
                if result.returncode == 0:
                    self.progress_var.set(100)
                    self.log_message("✅ Análisis batch completado exitosamente", "SUCCESS")
                    self.status_var.set("Análisis completado - Haz clic en 'Ver Resultados'")
                else:
                    self.log_message(f"❌ Error en análisis batch: {result.stderr}", "ERROR")
                    
            except Exception as e:
                self.log_message(f"❌ Error ejecutando análisis: {e}", "ERROR")
            finally:
                self.progress_var.set(0)
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=batch_thread)
        thread.daemon = True
        thread.start()
    
    def load_results(self):
        """
        Carga y muestra los resultados desde disco
        """
        def load_thread():
            try:
                self.status_var.set("Cargando resultados...")
                self.progress_var.set(10)
                
                # Cargar imágenes EDA
                self.load_eda_images()
                self.progress_var.set(25)
                
                # Cargar imágenes de métricas
                self.load_metrics_images()
                self.progress_var.set(50)
                
                # Cargar imágenes de predicciones
                self.load_predictions_images()
                self.progress_var.set(75)
                
                # Cargar imágenes de importancias
                self.load_importance_images()
                self.progress_var.set(90)
                
                # Cargar tabla de comparación
                self.load_comparison_table()
                self.progress_var.set(100)
                
                self.log_message("✅ Resultados cargados exitosamente", "SUCCESS")
                self.status_var.set("Resultados cargados")
                
            except Exception as e:
                self.log_message(f"❌ Error cargando resultados: {e}", "ERROR")
            finally:
                self.progress_var.set(0)
        
        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()
    
    def load_eda_images(self):
        """
        Carga las imágenes de EDA
        """
        try:
            # Limpiar frame
            for widget in self.eda_frame.winfo_children():
                widget.destroy()
            
            # Buscar imágenes EDA
            eda_images = []
            if os.path.exists(self.reports_figures_path):
                for file in os.listdir(self.reports_figures_path):
                    if 'eda' in file.lower() and file.endswith('.png'):
                        eda_images.append(os.path.join(self.reports_figures_path, file))
            
            if eda_images:
                # Mostrar primera imagen EDA
                self.display_image(self.eda_frame, eda_images[0])
            else:
                ttk.Label(self.eda_frame, text="No se encontraron imágenes de EDA", 
                         style=STYLES['info']).pack(expand=True)
                
        except Exception as e:
            ttk.Label(self.eda_frame, text=f"Error cargando EDA: {e}", 
                     style=STYLES['info']).pack(expand=True)
    
    def load_metrics_images(self):
        """
        Carga las imágenes de métricas
        """
        try:
            # Limpiar frame
            for widget in self.metrics_frame.winfo_children():
                widget.destroy()
            
            # Buscar imágenes de métricas
            metrics_images = []
            if os.path.exists(self.reports_figures_path):
                for file in os.listdir(self.reports_figures_path):
                    if any(metric in file.lower() for metric in ['metrics', 'mse', 'rmse', 'mae', 'r2', 'mape']) and file.endswith('.png'):
                        metrics_images.append(os.path.join(self.reports_figures_path, file))
            
            if metrics_images:
                # Mostrar primera imagen de métricas
                self.display_image(self.metrics_frame, metrics_images[0])
            else:
                ttk.Label(self.metrics_frame, text="No se encontraron imágenes de métricas", 
                         style=STYLES['info']).pack(expand=True)
                
        except Exception as e:
            ttk.Label(self.metrics_frame, text=f"Error cargando métricas: {e}", 
                     style=STYLES['info']).pack(expand=True)
    
    def load_predictions_images(self):
        """
        Carga las imágenes de predicciones
        """
        try:
            # Limpiar frame
            for widget in self.predictions_frame.winfo_children():
                widget.destroy()
            
            # Buscar imágenes de predicciones
            pred_images = []
            if os.path.exists(self.reports_figures_path):
                for file in os.listdir(self.reports_figures_path):
                    if 'pred' in file.lower() and file.endswith('.png'):
                        pred_images.append(os.path.join(self.reports_figures_path, file))
            
            if pred_images:
                # Mostrar primera imagen de predicciones
                self.display_image(self.predictions_frame, pred_images[0])
            else:
                ttk.Label(self.predictions_frame, text="No se encontraron imágenes de predicciones", 
                         style=STYLES['info']).pack(expand=True)
                
        except Exception as e:
            ttk.Label(self.predictions_frame, text=f"Error cargando predicciones: {e}", 
                     style=STYLES['info']).pack(expand=True)
    
    def load_importance_images(self):
        """
        Carga las imágenes de importancias
        """
        try:
            # Limpiar frame
            for widget in self.importance_frame.winfo_children():
                widget.destroy()
            
            # Buscar imágenes de importancias
            importance_images = []
            if os.path.exists(self.reports_figures_path):
                for file in os.listdir(self.reports_figures_path):
                    if 'importance' in file.lower() and file.endswith('.png'):
                        importance_images.append(os.path.join(self.reports_figures_path, file))
            
            if importance_images:
                # Mostrar primera imagen de importancias
                self.display_image(self.importance_frame, importance_images[0])
            else:
                ttk.Label(self.importance_frame, text="No se encontraron imágenes de importancias", 
                         style=STYLES['info']).pack(expand=True)
                
        except Exception as e:
            ttk.Label(self.importance_frame, text=f"Error cargando importancias: {e}", 
                     style=STYLES['info']).pack(expand=True)
    
    def load_comparison_table(self):
        """
        Carga la tabla de comparación
        """
        try:
            # Limpiar frame
            for widget in self.comparison_frame.winfo_children():
                widget.destroy()
            
            if os.path.exists(self.comparison_csv_path):
                # Cargar CSV
                df = pd.read_csv(self.comparison_csv_path)
                
                # Crear Treeview para mostrar la tabla
                tree_frame = ttk.Frame(self.comparison_frame)
                tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
                
                # Crear scrollbars
                v_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical')
                h_scrollbar = ttk.Scrollbar(tree_frame, orient='horizontal')
                
                # Crear Treeview
                tree = ttk.Treeview(tree_frame, 
                                  yscrollcommand=v_scrollbar.set,
                                  xscrollcommand=h_scrollbar.set)
                
                # Configurar scrollbars
                v_scrollbar.config(command=tree.yview)
                h_scrollbar.config(command=tree.xview)
                
                # Configurar columnas
                columns = list(df.columns)
                tree['columns'] = columns
                tree['show'] = 'headings'
                
                # Configurar encabezados
                for col in columns:
                    tree.heading(col, text=col)
                    tree.column(col, width=120, anchor='center')
                
                # Insertar datos
                for index, row in df.iterrows():
                    tree.insert('', 'end', values=list(row))
                
                # Posicionar widgets
                tree.grid(row=0, column=0, sticky='nsew')
                v_scrollbar.grid(row=0, column=1, sticky='ns')
                h_scrollbar.grid(row=1, column=0, sticky='ew')
                
                tree_frame.grid_rowconfigure(0, weight=1)
                tree_frame.grid_columnconfigure(0, weight=1)
                
            else:
                ttk.Label(self.comparison_frame, text="No se encontró el archivo de comparación", 
                         style=STYLES['info']).pack(expand=True)
                
        except Exception as e:
            ttk.Label(self.comparison_frame, text=f"Error cargando comparación: {e}", 
                     style=STYLES['info']).pack(expand=True)
    
    def display_image(self, parent_frame, image_path):
        """
        Muestra una imagen en un frame
        """
        try:
            # Cargar imagen con PIL
            image = Image.open(image_path)
            
            # Redimensionar si es muy grande
            max_width, max_height = 600, 400
            if image.width > max_width or image.height > max_height:
                image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Convertir a PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Crear label para mostrar imagen
            label = ttk.Label(parent_frame, image=photo)
            label.image = photo  # Mantener referencia
            label.pack(expand=True)
            
        except Exception as e:
            ttk.Label(parent_frame, text=f"Error mostrando imagen: {e}", 
                     style=STYLES['info']).pack(expand=True)
    
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
        self.status_var.set("Listo - Haz clic en 'Generar Análisis Batch' para comenzar")
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
    main()
