"""
Módulo para el manejo y preprocesamiento de datos de cafeterías
Universidad Pedagógica y Tecnológica de Colombia
Inteligencia Computacional
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ..utils.constants import MODEL_CONFIG, FILES
from ..utils.helpers import setup_matplotlib
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib
setup_matplotlib()

class DataHandler:
    """
    Clase para el manejo y preprocesamiento de datos de cafeterías
    """
    
    def __init__(self, csv_path):
        """
        Inicializa el manejador de datos
        
        Args:
            csv_path (str): Ruta al archivo CSV
        """
        self.csv_path = csv_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """
        Carga los datos desde el archivo CSV
        """
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"✅ Datos cargados exitosamente: {self.data.shape[0]} filas, {self.data.shape[1]} columnas")
            return True
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
            return False
    
    def explore_data(self):
        """
        Realiza exploración básica de los datos
        """
        if self.data is None:
            print("❌ Primero debe cargar los datos")
            return
        
        print("\n" + "="*60)
        print("📊 EXPLORACIÓN DE DATOS")
        print("="*60)
        
        # Información básica
        print(f"\n📋 Información del dataset:")
        print(f"   • Filas: {self.data.shape[0]:,}")
        print(f"   • Columnas: {self.data.shape[1]}")
        print(f"   • Memoria utilizada: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Tipos de datos
        print(f"\n🔍 Tipos de datos:")
        for col, dtype in self.data.dtypes.items():
            print(f"   • {col}: {dtype}")
        
        # Estadísticas descriptivas
        print(f"\n📈 Estadísticas descriptivas:")
        print(self.data.describe().round(2))
        
        # Valores faltantes
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  Valores faltantes:")
            for col, count in missing[missing > 0].items():
                print(f"   • {col}: {count} ({count/len(self.data)*100:.1f}%)")
        else:
            print(f"\n✅ No hay valores faltantes")
        
        # Valores únicos
        print(f"\n🔢 Valores únicos por columna:")
        for col in self.data.columns:
            unique_count = self.data[col].nunique()
            print(f"   • {col}: {unique_count:,} valores únicos")
    
    def visualize_data(self):
        """
        Crea visualizaciones exploratorias de los datos
        """
        if self.data is None:
            print("❌ Primero debe cargar los datos")
            return
        
        # Configurar subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('📊 Análisis Exploratorio de Datos - Cafeterías', fontsize=16, fontweight='bold')
        
        # 1. Distribución de ingresos diarios
        axes[0, 0].hist(self.data['Daily_Revenue'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribución de Ingresos Diarios', fontweight='bold')
        axes[0, 0].set_xlabel('Ingresos Diarios ($)')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Correlación con ingresos
        correlations = self.data.corr()['Daily_Revenue'].drop('Daily_Revenue').sort_values(ascending=True)
        colors = ['red' if x < 0 else 'green' for x in correlations.values]
        axes[0, 1].barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
        axes[0, 1].set_yticks(range(len(correlations)))
        axes[0, 1].set_yticklabels(correlations.index, fontsize=9)
        axes[0, 1].set_title('Correlación con Ingresos Diarios', fontweight='bold')
        axes[0, 1].set_xlabel('Coeficiente de Correlación')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Clientes vs Ingresos
        axes[0, 2].scatter(self.data['Number_of_Customers_Per_Day'], self.data['Daily_Revenue'], 
                          alpha=0.6, color='purple')
        axes[0, 2].set_title('Clientes Diarios vs Ingresos', fontweight='bold')
        axes[0, 2].set_xlabel('Número de Clientes por Día')
        axes[0, 2].set_ylabel('Ingresos Diarios ($)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Valor promedio de orden vs Ingresos
        axes[1, 0].scatter(self.data['Average_Order_Value'], self.data['Daily_Revenue'], 
                          alpha=0.6, color='orange')
        axes[1, 0].set_title('Valor Promedio de Orden vs Ingresos', fontweight='bold')
        axes[1, 0].set_xlabel('Valor Promedio de Orden ($)')
        axes[1, 0].set_ylabel('Ingresos Diarios ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Horas de operación vs Ingresos
        axes[1, 1].scatter(self.data['Operating_Hours_Per_Day'], self.data['Daily_Revenue'], 
                          alpha=0.6, color='red')
        axes[1, 1].set_title('Horas de Operación vs Ingresos', fontweight='bold')
        axes[1, 1].set_xlabel('Horas de Operación por Día')
        axes[1, 1].set_ylabel('Ingresos Diarios ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Matriz de correlación
        corr_matrix = self.data.corr()
        im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 2].set_title('Matriz de Correlación', fontweight='bold')
        axes[1, 2].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 2].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 2].set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
        axes[1, 2].set_yticklabels(corr_matrix.columns, fontsize=8)
        
        # Añadir valores de correlación
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = axes[1, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=7)
        
        plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
        plt.tight_layout()
        plt.show()
        
        # Gráfico adicional: Boxplot de ingresos por rangos de clientes
        plt.figure(figsize=(12, 6))
        
        # Crear rangos de clientes
        self.data['Customer_Range'] = pd.cut(self.data['Number_of_Customers_Per_Day'], 
                                           bins=5, labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])
        
        sns.boxplot(data=self.data, x='Customer_Range', y='Daily_Revenue', palette='Set2')
        plt.title('📊 Distribución de Ingresos por Rango de Clientes', fontsize=14, fontweight='bold')
        plt.xlabel('Rango de Clientes Diarios', fontweight='bold')
        plt.ylabel('Ingresos Diarios ($)', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def prepare_data(self, test_size=None, random_state=None):
        """
        Prepara los datos para el entrenamiento
        
        Args:
            test_size (float): Proporción de datos para testing
            random_state (int): Semilla para reproducibilidad
        """
        if self.data is None:
            print("❌ Primero debe cargar los datos")
            return False
        
        # Usar valores por defecto de configuración si no se proporcionan
        if test_size is None:
            test_size = MODEL_CONFIG['test_size']
        if random_state is None:
            random_state = MODEL_CONFIG['random_state']
        
        print("\n" + "="*60)
        print("🔧 PREPARACIÓN DE DATOS")
        print("="*60)
        
        # Separar características y variable objetivo
        X = self.data.drop('Daily_Revenue', axis=1)
        y = self.data['Daily_Revenue']
        
        self.feature_names = X.columns.tolist()
        
        print(f"📋 Características seleccionadas:")
        for i, feature in enumerate(self.feature_names, 1):
            print(f"   {i}. {feature}")
        
        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n📊 División de datos:")
        print(f"   • Conjunto de entrenamiento: {self.X_train.shape[0]:,} muestras")
        print(f"   • Conjunto de prueba: {self.X_test.shape[0]:,} muestras")
        print(f"   • Características: {self.X_train.shape[1]}")
        
        # Estandarizar las características
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Datos preparados y estandarizados exitosamente")
        return True
    
    def get_data(self):
        """
        Retorna los datos preparados
        
        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        if self.X_train_scaled is None:
            print("❌ Primero debe preparar los datos")
            return None
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Evalúa un modelo y retorna las métricas
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            model_name: Nombre del modelo
            
        Returns:
            dict: Diccionario con las métricas
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'Modelo': model_name,
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R²': round(r2, 4)
        }
        
        print(f"\n📊 Métricas de {model_name}:")
        print(f"   • MSE (Error Cuadrático Medio): {metrics['MSE']:,.2f}")
        print(f"   • RMSE (Raíz del Error Cuadrático Medio): {metrics['RMSE']:,.2f}")
        print(f"   • MAE (Error Absoluto Medio): {metrics['MAE']:,.2f}")
        print(f"   • R² (Coeficiente de Determinación): {metrics['R²']:.4f}")
        
        return metrics
