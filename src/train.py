import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

print("--- Debug: Iniciando entrenamiento del modelo ---")

# Ruta al dataset (ajustada a tu estructura)
data_path = os.path.join(os.path.dirname(__file__), '..', 'Historical Product Demand.csv')

# Cargar dataset
data = pd.read_csv(data_path)
print(f"--- Debug: Dataset cargado con {data.shape[0]} filas y {data.shape[1]} columnas ---")

data = data.sample(n=10000, random_state=42)
print(f"--- Debug: Usando muestra de {data.shape[0]} filas ---")


# ==========================
# 🧹 Limpieza avanzada
# ==========================
data.columns = data.columns.str.strip()
data.dropna(inplace=True)

# Convertir la columna de demanda (valores entre paréntesis a negativos)
data["Order_Demand"] = (
    data["Order_Demand"]
    .astype(str)
    .str.replace("(", "-", regex=False)
    .str.replace(")", "", regex=False)
    .astype(float)
)

# Convertir variables categóricas en numéricas (one-hot encoding)
data = pd.get_dummies(data, drop_first=True)

print(f"--- Debug: Dataset limpio con {data.shape[0]} filas y {data.shape[1]} columnas ---")

# ==========================
# 🔢 Separar variables
# ==========================
X = data.drop("Order_Demand", axis=1)
y = data["Order_Demand"]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# 🤖 Entrenamiento del modelo
# ==========================
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# ==========================
# 📊 Métricas de evaluación
# ==========================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- Debug: MSE={mse:.4f}, R2={r2:.4f} ---")

# ==========================
# 💾 Guardar modelo (compatible con Windows y Linux)
# ==========================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(project_root, "model.pkl")
joblib.dump(model, model_path)
print(f"--- Debug: Modelo guardado en {model_path} ---")

# ==========================
# 🧠 Registrar en MLflow
# ==========================
mlflow.set_tracking_uri("file://" + os.path.join(project_root, "mlruns"))
mlflow.set_experiment("MLOps_Historical_Demand")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Muestra de entrada para registrar la firma del modelo
    input_example = X_test.iloc[:5]
    mlflow.sklearn.log_model(model, name="model", input_example=input_example)

print("--- ✅ Entrenamiento completado y modelo guardado exitosamente ---")
