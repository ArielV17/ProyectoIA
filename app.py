from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# =============================
# Cargar modelo y columnas
# =============================
import joblib
model = joblib.load("modelo_xgboost_final.pkl")

with open("columnas_entrenamiento.pkl", "rb") as f:
    COLUMNAS = pickle.load(f)

# =============================
# VALORES ORIGINALES (LOS DE ANTES)
# =============================

TIPOS = [
    "Departamento",
    "Local comercial",
    "Oficina",
    "Otro"
]

CIUDADES = [
    "Guayaquil",
    "Quito",
    "Samborondon",
    "Playas (General Villamil)"
]

BARRIOS = [
    "Centro De Guayaquil",
    "Centro Histórico",
    "Centro Norte",
    "El Ejido",
    "El Morro",
    "La Puntilla",
    "Norte De Guayaquil",
    "Norte De Quito",
    "Playas",
    "Samborondon",
    "San Sebastian",
    "Sur De Guayaquil",
    "Sur De Quito",
    "Totoracocha",
    "Valle Los Chillos",
    "Valle Tumbaco",
    "Via A La Costa",
    "Yanuncay"
]

# =============================
# Rutas
# =============================
@app.route("/")
def index():
    return render_template(
        "index.html",
        tipos=TIPOS,
        ciudades=CIUDADES,
        barrios=BARRIOS
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # DataFrame con todas las columnas en 0
    X = pd.DataFrame(0, index=[0], columns=COLUMNAS)

    # Variables numéricas
    X["surface_total"] = float(data["surface_total"])
    X["bedrooms"] = int(data["bedrooms"])
    X["bathrooms"] = int(data["bathrooms"])

    # One-hot encoding manual
    X[f"property_type_{data['tipo']}"] = 1
    X[f"l3_{data['ciudad']}"] = 1
    X[f"l4_{data['barrio']}"] = 1

    # Predicción
    precio = model.predict(X)[0]

    return jsonify({
        "precio_estimado": round(float(precio), 2),
        "min": round(float(precio * 0.95), 2),
        "max": round(float(precio * 1.05), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)



