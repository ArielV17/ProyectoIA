from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# =============================
# Cargar modelo y columnas
# =============================
import joblib
model = joblib.load("modelo_xgboost_final.pkl")

with open("columnas_entrenamiento.pkl", "rb") as f:
    COLUMNAS = pickle.load(f)



TIPOS = ["Departamento", "Local comercial", "Oficina", "Casa", "Suite", "Otro"]
CIUDADES = ["Quito", "Guayaquil", "Samborondon", "Playas (General Villamil)", "Cuenca"]

# Diccionario de Barrios por Ciudad
BARRIOS_POR_CIUDAD = {
    "Quito": [
        "Centro Histórico",
        "Centro Norte",
        "El Ejido",
        "Norte De Quito",
        "Sur De Quito",
        "Valle Los Chillos",
        "Valle Tumbaco"
    ],
    "Guayaquil": [
        "Centro De Guayaquil",
        "Norte De Guayaquil",
        "Sur De Guayaquil",
        "Via A La Costa",
        "El Morro" # (Parroquia rural de GYE)
    ],
    "Samborondon": [
        "La Puntilla",
        "Samborondon"
    ],
    "Playas (General Villamil)": [
        "Playas"
    ],
    "Cuenca": [
        "San Sebastian",
        "Totoracocha",
        "Yanuncay"
    ]
}

# =============================
# Rutas
# =============================
@app.route("/")
def index():
    return render_template(
        "index.html",
        tipos=TIPOS,
        ciudades=CIUDADES,
        barrios_por_ciudad=BARRIOS_POR_CIUDAD
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
    precio = np.exp(model.predict(X)[0])

    return jsonify({
        "precio_estimado": round(float(precio), 2),
        "min": round(float(precio-150.81), 2),
        "max": round(float(precio+150.81), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)




