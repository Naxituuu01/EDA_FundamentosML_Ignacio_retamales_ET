# Proyecto de Machine Learning - FastAPI

Este proyecto implementa dos modelos de Machine Learning:

- 🔢 **Regresión Lineal Múltiple**: Predice un valor numérico con base en variables del juego.
- 🔍 **Clasificación (XGBoost)**: Predice la probabilidad de ganar una ronda.

## 📦 Tecnologías usadas

- Python
- FastAPI
- scikit-learn
- xgboost
- joblib
- pandas
- matplotlib / seaborn

## 📁 Estructura del Proyecto

ML_examen_transversal/
│
├── main.py # API con FastAPI
├── modelo_clasificacion.pkl
├── columnas_clasificacion.pkl
├── modelo_regresion.pkl
├── columnas_regresion.pkl
├── static/ # Frontend (HTML/CSS/JS si corresponde)
├── README.md
└── .gitignore

ML_examen_transversal/
│
├── main.py # API con FastAPI
├── modelo_clasificacion.pkl
├── columnas_clasificacion.pkl
├── modelo_regresion.pkl
├── columnas_regresion.pkl
├── static/ # Frontend (HTML/CSS/JS si corresponde)
├── README.md
└── .gitignore


## 🚀 Cómo ejecutar

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt

uvicorn main:app --reload

http://127.0.0.1:8000
