from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# =========================
# 🌐 CONFIGURACIÓN CORS Y ARCHIVOS ESTÁTICOS
# =========================

# Permitir CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# 📥 CARGAR MODELOS Y COLUMNAS
# =========================
modelo_regresion = joblib.load("modelo_regresion.pkl")
cols_regresion = joblib.load("columnas_regresion.pkl")

modelo_clasificacion = joblib.load("modelo_clasificacion.pkl")
cols_clasificacion = joblib.load("columnas_clasificacion.pkl")

# =========================
# 🏠 RUTA PRINCIPAL (FRONTEND)
# =========================
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

# =========================
# 📄 ESQUEMA DE DATOS
# =========================

class DatosRegresion(BaseModel):
    RoundId: float
    TeamStartingEquipmentValue: float
    MatchFlankKills: float
    MatchAssists: float
    MatchHeadshots: float

class DatosClasificacion(BaseModel):
    Survived: int
    TeamStartingEquipmentValue: int
    MatchWinner: int
    RoundKills: int
    RoundAssists: int
    Team: int

# =========================
# 🔮 ENDPOINT REGRESIÓN
# =========================

@app.post("/predecir/regresion")
def predecir_regresion(datos: DatosRegresion):
    try:
        df = pd.DataFrame([datos.dict()])
        df = df[cols_regresion]
        pred = modelo_regresion.predict(df)[0]
        return {"modelo": "regresion", "prediccion": round(pred, 2)}
    except Exception as e:
        return {"error": str(e)}

# =========================
# 🔮 ENDPOINT CLASIFICACIÓN
# =========================

@app.post("/predecir/clasificacion")
def predecir_clasificacion(datos: DatosClasificacion):
    try:
        df = pd.DataFrame([datos.dict()])
        df = df[cols_clasificacion]
        prob = float(modelo_clasificacion.predict_proba(df)[0][1]) 
        clase = int(modelo_clasificacion.predict(df)[0]) 
        return {
            "modelo": "clasificacion",
            "clase_predicha": clase,
            "probabilidad_clase_1": round(prob, 4)
        }
    except Exception as e:
        return {"error": str(e)}

# =========================
# 🔍 ENDPOINT DE INFORMACIÓN
# =========================

@app.get("/info")
def info():
    return {
        "modelos_disponibles": ["regresion", "clasificacion"],
        "columnas_regresion": cols_regresion,
        "columnas_clasificacion": cols_clasificacion
    }