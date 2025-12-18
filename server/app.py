import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# --- Configuration et Chargement du Modèle ---
MODEL_PATH = "./server/model.pkl"
app = FastAPI(title="Penguin Species Predictor API")

# Variable pour stocker le modèle et les noms de classes
model_artifact = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle non trouvé à: {MODEL_PATH}")
        
    model_artifact = joblib.load(MODEL_PATH)
    MODEL = model_artifact['model']
    CLASS_NAMES = model_artifact['class_names']
    print(f"Modèle chargé. Classes: {CLASS_NAMES}")
except FileNotFoundError as e:
    print(f"ERREUR FATALE: {e}")
    


class PenguinFeatures(BaseModel):
    
    # Variables Numériques
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    
    # Variables Catégorielles
    island: str # Torgersen, Biscoe, Dream
    sex: str    # Male, Female, . (NA)

# --- Endpoint de Prédiction ---
@app.post("/predict", tags=["Prediction"])
def predict_penguin_species(features: PenguinFeatures):
    """
    Reçoit les caractéristiques d'un manchot et retourne l'espèce prédite.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Le modèle n'a pas pu être chargé. Service indisponible.")
    
    try:
        # Convertir les données d'entrée Pydantic en un DataFrame Pandas
        input_data = pd.DataFrame([features.model_dump()])
        
        # Le modèle (pipeline) gère automatiquement la standardisation/encodage
        prediction_index = MODEL.predict(input_data)[0]
        
        # Convertir l'index numérique en nom de classe
        predicted_species = CLASS_NAMES[prediction_index]
        
        return {
            "prediction": predicted_species,
            "details": f"Index prédit: {prediction_index}"
        }

    except Exception as e:
        # En cas d'erreur de prédiction (ex: données mal formatées)
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {e}")

# --- Endpoint de Santé (Health Check) ---
@app.get("/health", tags=["Monitoring"])
def health_check():
    """Vérifie si l'API et le modèle sont chargés."""
    status = "ready" if MODEL is not None else "model_missing"
    return {"status": status, "model_loaded": MODEL is not None}