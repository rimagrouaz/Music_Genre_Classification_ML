from flask import Flask, render_template, request
import utils
from models import model

app = Flask(__name__)
modele = model.load_model('models/model1.pkl')

@app.route('/', methods=['GET', 'POST'])
def hello_word():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == "audio":
            # Cas pour fichier audio uploadé
            audio_file = request.files['audioFile']
            audio_path = "./audios/" + audio_file.filename
            audio_file.save(audio_path)
        else:
            # Cas pour lien YouTube
            youtube_link = request.form.get('youtubeInput')
            audio_path = utils.download_and_extract_audio(youtube_link)

        # Calcul des caractéristiques
        all_features = utils.calculeFeautures(audio_path=audio_path)
        features = utils.imports.np.array(all_features, utils.imports.np.float32).reshape(1, -1)
        scaler = utils.imports.joblib.load('models/scaler1.pkl')
        features_scaled = scaler.transform(features)
        print(features_scaled)

        # Prédiction
        prediction = model.predict_class(modele, features=features_scaled)
        probabilites = modele.predict_proba(features_scaled)
        
        # Classes et résultats
        classes = ['Classical', 'Jazz', 'Metal', 'Pop']
        predicted_class = classes[prediction]
        predicted_proba = "{:.2f}".format(probabilites[0][prediction] * 100)

        # Obtenir des recommandations YouTube
        recommendations = utils.search_youtube(predicted_class)
        
        return render_template("page.html", prediction=predicted_class, proba=predicted_proba, recommendations=recommendations)

    # GET request : Affichage initial+
    return render_template("page.html")

if __name__ == '__main__':
    app.run(port=3000, debug=True)
