import mlflow
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import os

print("🚀 Iniciando validación del modelo...")

# === 1. Ruta al dataset ===
data_path = os.path.join(os.path.dirname(__file__), '..', 'Historical Product Demand.csv')
data = pd.read_csv(data_path)
print(f"✅ Dataset cargado con {data.shape[0]} filas y {data.shape[1]} columnas.")

# === 2. Limpieza avanzada (idéntica a train.py para coherencia) ===
data.columns = data.columns.str.strip()
data.dropna(inplace=True)
data["Order_Demand"] = (
    data["Order_Demand"]
    .astype(str)
    .str.replace('(', '-', regex=True)
    .str.replace(')', '', regex=True)
    .astype(float)
)
data = pd.get_dummies(data, drop_first=True)
print(f"🧹 Dataset limpio con {data.shape[0]} filas y {data.shape[1]} columnas.")

# === 3. Variables ===
if "Order_Demand" not in data.columns:
    raise ValueError("❌ La columna 'Order_Demand' no existe en el dataset después de la limpieza.")

X = data.drop("Order_Demand", axis=1)
y = data["Order_Demand"]

# === 4. Cargar modelo ===
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ No se encontró el modelo en {model_path}. Ejecuta primero train.py.")

model = joblib.load(model_path)
print("✅ Modelo cargado correctamente.")

# === 5. Evaluación ===
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"📊 Resultados de validación:")
print(f"   - MSE: {mse:.4f}")
print(f"   - R²: {r2:.4f}")

# === 6. Validación con umbral (control de calidad del modelo) ===
THRESHOLD = 150000000.0
if mse < THRESHOLD:
    print("✅ El modelo cumple con el umbral esperado. Pipeline exitoso.")
else:
    print("❌ El modelo no cumple con el umbral esperado. Deteniendo pipeline.")
    exit(1)

# === 7. Registrar validación en MLflow ===
mlflow.set_experiment("MLOps_Historical_Product_Demand")
with mlflow.start_run(run_name="validacion_modelo"):
    mlflow.log_metric("mse_validation", mse)
    mlflow.log_metric("r2_validation", r2)

print("🎯 Validación completada y registrada en MLflow.")
