import streamlit as st
import requests
import json
import os

# --- Configuration ---
# Récupérer les variables d'environnement définies dans docker-compose.yml
# Fallback aux valeurs par défaut pour le test local
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}/predict"

st.set_page_config(page_title="Penguin Predictor", layout="wide")

st.title("🐧 Détecteur d'Espèce de Manchot")
st.markdown("---")

# --- Interface Utilisateur (Barre Latérale) ---
with st.sidebar:
    st.header("Caractéristiques du Manchot")
    
    # Variables Numériques
    bill_length = st.slider("Longueur du bec (mm)", 30.0, 60.0, 44.0)
    bill_depth = st.slider("Profondeur du bec (mm)", 13.0, 22.0, 17.0)
    flipper_length = st.slider("Longueur de l'aileron (mm)", 170.0, 240.0, 200.0)
    body_mass = st.slider("Masse corporelle (g)", 2700.0, 6300.0, 4000.0)
    
    st.markdown("---")
    
    # Variables Catégorielles
    island = st.selectbox("Île", ('Torgersen', 'Biscoe', 'Dream'))
    sex = st.selectbox("Sexe", ('Male', 'Female'))

# --- Construction des Données d'Entrée ---

input_data = {
    "bill_length_mm": bill_length,
    "bill_depth_mm": bill_depth,
    "flipper_length_mm": flipper_length,
    "body_mass_g": body_mass,
    "island": island,
    "sex": sex
}

st.subheader("Données Soumises")
st.json(input_data)

st.markdown("---")

# --- Appel à l'API ---
if st.button("Prédire l'Espèce"):
    with st.spinner("Envoi des données au serveur..."):
        try:
            # Envoi de la requête POST au serveur FastAPI
            response = requests.post(API_URL, json=input_data)
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Prédiction Réussie !")
                
                # Affichage du résultat principal
                st.subheader(f"L'espèce de ce manchot est : **{result['prediction']}**")
                
                # Affichage des détails techniques
                with st.expander("Détails de la Réponse API"):
                    st.write(result)
            
            elif response.status_code == 503:
                st.error("❌ ERREUR DE SERVICE : Le serveur est opérationnel mais le modèle (model.pkl) n'a pas pu être chargé.")
                st.write(response.json())
                
            else:
                st.error(f"❌ ERREUR API : Le serveur a retourné le code {response.status_code}")
                st.write(response.json())
                
        except requests.exceptions.ConnectionError:
            st.error(f"❌ ERREUR DE CONNEXION : Impossible de joindre le serveur à {API_URL}.")
            st.warning("Vérifiez que le service 'server' est bien lancé et accessible sur le port 8000.")