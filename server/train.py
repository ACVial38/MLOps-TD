import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib 


MODEL_FILENAME = "model.pkl"
DATA_PATH = "C:/Users/admin/OneDrive/Documents/Documents études/td-mlops/MLOps-td_3/server/penguins.csv" 

def train_and_save_penguins_model():
    """
    Charge le dataset Penguins, effectue le prétraitement,
    entraîne un modèle de Régression Logistique et le sauvegarde au format PKL.
    """
    print(f"--- 1. Chargement des données depuis {DATA_PATH} ---")
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{DATA_PATH}' est introuvable.")
        return
    
    # Prétraitement de base
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
        
    target_column = 'species'
    data.dropna(subset=[target_column], inplace=True)
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Transformation de la cible en valeurs numériques (0, 1, 2) pour l'entraînement
    y_encoded, class_names = pd.factorize(y)

    print(f"Classes cibles encodées: {dict(zip(range(len(class_names)), class_names))}")

    # Identification des colonnes
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Création du préprocesseur (Pipeline)
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    
    # Séparation et entraînement
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42)) 
    ])

    print("--- 2. Entraînement du Modèle ---")
    model_pipeline.fit(X_train, y_train)
    
    # Sauvegarde du Modèle et des noms de classes (nécessaire pour la prédiction)
    print(f"--- 3. Sauvegarde du modèle et des classes dans {MODEL_FILENAME} ---")
    
    # Sauvegarder le modèle et les noms de classes dans un dictionnaire
    artifact = {
        'model': model_pipeline,
        'class_names': class_names.tolist() # Convertir en liste pour joblib
    }
    
    joblib.dump(artifact, MODEL_FILENAME)
    print(f"Le modèle et les noms de classes ont été sauvegardés dans '{MODEL_FILENAME}'.")
    
    return model_pipeline

if __name__ == "__main__":
    train_and_save_penguins_model()