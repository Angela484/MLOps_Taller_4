import pandas as pd
import mlflow
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score

print("--- Debug: Iniciando validación del modelo ---")

# ========================================
# 1. Ruta del dataset (ajustada a tu estructura)
# ========================================
data_path = os.path.join(os.path.dirname(__file__), '..', 'Historical Product Demand.csv')

# ========================================
# 2. Cargar dataset y usar muestra
# ========================================
data = pd.read_csv(data_path)
print(f"--- Debug: Dataset cargado con {data.shape[0]} filas y {data.shape[1]} columnas ---")

data = data.sample(n=10000, random_state=42)
print(f"--- Debug: Usando muestra de {data.shape[0]} filas ---")

# ========================================
# 3. Limpieza avanzada (igual que en train.py)
# ========================================
data.columns = data.columns.str.strip()
data.dropna(inplace=True)

# Convertir la columna de demanda (valores entre paréntesis a negativos)
data["Order_Demand"] = (
    data["Order_Demand"]
    .astype(str)
    .str.replace('(', '-', regex=False)
    .str.replace(')', '', regex=False)
    .astype(float)
)

# Convertir variables categóricas a numéricas (one-hot encoding)
data = pd.get_dummies(data, drop_first=True)

print(f"--- Debug: Dataset limpio con {data.shape[0]} filas y {data.shape[1]} columnas ---")

# ========================================
# 4. Separar features y target
# ========================================
X = data.drop(columns=["Order_Demand"])
y = data["Order_Demand"]

# ========================================
# 5. Cargar modelo entrenado
# ========================================
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"⚠️ No se encontró el modelo en {model_path}")

model = joblib.load(model_path)
print("--- Debug: Modelo cargado correctamente ---")

# ========================================
# 6. Hacer predicciones
# ========================================
y_pred = model.predict(X)

# ========================================
# 7. Calcular métricas
# ========================================
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"📊 MSE del modelo: {mse:.4f}")
print(f"📈 R² del modelo: {r2:.4f}")

# ========================================
# 8. Evaluar si cumple umbral esperado
# ========================================
THRESHOLD = 3e8


if mse < THRESHOLD:
    print("✅ El modelo cumple con el umbral esperado. Validación exitosa.")
else:
    print("❌ El modelo no cumple el umbral esperado. Deteniendo pipeline.")
    exit(1)

print("✅ Validación completada sin errores.")
