import imports

# Charger le modèle préalablement sauvegardé
def load_model(model_path='model.pkl'):
    model = imports.joblib.load(model_path)
    return model

# Faire une prédiction avec le modèle
def predict_class(model, features):
    prediction = model.predict(features)
    return prediction[0]  # Retourne la classe prédite (pour une prédiction binaire ou multiclass)

