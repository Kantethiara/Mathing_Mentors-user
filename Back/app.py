from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === Chargement des modèles ===
try:
    # Charger le modèle
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Charger les embeddings et le dataframe
    with open("/Users/thiarakante/Documents/Databeez/Matching-user-mentorat/Back/model/embeddings_mentors.pkl", "rb") as f:
        embeddings_mentors = pickle.load(f)
        

    with open("/Users/thiarakante/Documents/Databeez/Matching-user-mentorat/Back/model/df_mentors.pkl", "rb") as f:
        df_mentors = pickle.load(f)

except Exception as e:
    raise RuntimeError(f"Erreur de chargement des fichiers de modèle: {str(e)}")


# === Accueil ===
@app.route('/')
def home():
    return render_template('index.html')

# === Formulaire individuel ===
@app.route('/formulaire-individuel', methods=['GET', 'POST'])
def formulaire_individuel():
    if request.method == 'GET':
        return render_template('formulaire_individuel.html')
    
    try:
        required_fields = ['email', 'objectif', 'formation', 'format_formation', 'situation', 'budget']
        if not all(field in request.form and request.form[field].strip() for field in required_fields):
            return render_template('formulaire_individuel.html', error="Veuillez remplir tous les champs requis.")

        form_data = {
            'email': request.form['email'],
            'objectif': request.form['objectif'],
            'formation': request.form['formation'],
            'format_formation': request.form['format_formation'],
            'situation': request.form['situation'],
            'remarques': request.form.get('remarques', ''),
            'horodateur': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }

        # Création du profil textuel
        profil_textuel = " ".join([
            form_data['objectif'],
            form_data['formation'],
            form_data['format_formation'],
            form_data['situation'],
            
        ])

        recommandations = get_top_mentors(profil_textuel)

        return render_template('resultats_individuels.html',
                               
                               form_data=form_data,
                               recommandations=recommandations)

    except Exception as e:
        return render_template('formulaire_individuel.html',
                               error=f"Une erreur est survenue: {str(e)}")

# === Upload batch ===
@app.route('/batch-upload', methods=['GET', 'POST'])
def batch_upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('batch_upload.html', error="Veuillez choisir un fichier.")

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            try:
                file.save(filepath)

                df = pd.read_excel(filepath) if filename.lower().endswith('xlsx') else pd.read_csv(filepath)

                expected_columns = [
                    'Horodateur', 'Adresse e-mail', 'Quel est ton principal objectif professionnel ?',
                    'Quelle formation serait idéale pour toi ?',
                    'Quel format de formation vous convient le plus? ',
                    'Quelle est ton métier et ta situation professionnelle actuelle ?',
                    # "Quel est budget que vous seriez prêt à mettre sous garantie d'atteinte de votre objectif principal? ",
                    'Des remarques, suggestions ou questions ? '
                ]

                missing = [col for col in expected_columns if col not in df.columns]
                if missing:
                    return render_template('error.html',
                                           message=f"Colonnes manquantes dans le fichier : {', '.join(missing)}")

                recommandations = get_recommendations_for_df(df)

                return render_template('resultats_batch.html',
                                       filename=filename,
                                       horodateur=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                       recommandations=recommandations)

            except Exception as e:
                return render_template('error.html', message=f"Erreur lors du traitement : {str(e)}")

    return render_template('batch_upload.html')

# === Helpers ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}



def get_top_mentors(profil_textuel, top_n=3):
    try:
        # Vectoriser le profil de l'élève avec le modèle SentenceTransformer
        vect_eleve = model.encode([profil_textuel], convert_to_tensor=True)

        # vect_mentors est déjà un Tensor, pas besoin de le recalculer
        vect_mentors = embeddings_mentors

        # Calculer les similarités cosinus
        similarites = cosine_similarity(vect_eleve.cpu().numpy(), vect_mentors.cpu().numpy())[0]

        # Récupérer les indices des meilleurs mentors
        top_indices = similarites.argsort()[-top_n:][::-1]

        results = []
        for idx in top_indices:
            mentor = df_mentors.iloc[idx]
            results.append({
                'nom': mentor.get("nom_mentor", ""),
                'url': f"/mentor/{mentor.get('nom_mentor', '').replace(' ', '%20')}",
                'score': round(float(similarites[idx]), 3),
                'situation_professionnelle': mentor.get("Quelle est ton métier et ta situation professionnelle actuelle ?", ""),
                'expertise': mentor.get("Domaine principal d'expertise", ""),
                'experience': mentor.get("Combien d'années d'expérience avez vous dans ce domaine ?", ""),
                'objectif': mentor.get("Sélectionne ton principal objectif en tant que mentor", ""),
                'competences': mentor.get("Liste les compétences clés que tu peux transmettre à des apprenants dans ce domaine ou métier", ""),
                'disponibilites_jours': mentor.get("Quels sont les jours où tu serais disponible pour des sessions d'1h30 par semaine", ""),
                'disponibilites_periodes': mentor.get("Quelles sont les périodes de la journée où tu serais disponible?", "")
            })

        return results

    except Exception as e:
        print(f"[ERREUR] get_top_mentors: {e}")
        return []


@app.route("/mentor/<nom>")
def mentor_detail(nom):
    nom_decoded = nom.replace('%20', ' ')
    mentor = df_mentors[df_mentors["nom_mentor"] == nom_decoded]

    if mentor.empty:
        return f"Mentor '{nom_decoded}' non trouvé.", 404

    mentor = mentor.iloc[0]

    infos_mentor = {
        "nom": mentor["nom_mentor"],
        "situation_professionnelle": mentor.get("Quel est ton métier et ta situation professionnelle actuelle ?", ""),
        "expertise": mentor.get("Domaine principal d'expertise", ""),
        "experience": mentor.get("Combien d'années d'expérience avez vous dans ce domaine ?", ""),
        "objectif": mentor.get("Sélectionne ton principal objectif en tant que mentor", ""),
        "competences": mentor.get("Liste les compétences clés que tu peux transmettre à des apprenants dans ce domaine ou métier", ""),
        "disponibilites_jours": mentor.get("Quels sont les jours où tu serais disponible pour des sessions d'1h30 par semaine", ""),
        "disponibilites_periodes": mentor.get("Quelles sont les périodes de la journée où tu serais disponible?", "")
    }

    return render_template("mentor_detail.html", mentor=infos_mentor)

def get_recommendations_for_df(df):
    recommandations = []

    for _, row in df.iterrows():
        try:
            texte = " ".join([
                str(row.get('Quel est ton principal objectif professionnel ?', '')).strip(),
                str(row.get('Quelle formation serait idéale pour toi ?', '')).strip(),
                str(row.get('Quel format de formation vous convient le plus? ', '')).strip(),
                str(row.get('Quelle est ton métier et ta situation professionnelle actuelle ?', '')).strip(),
                str(row.get("Quel est budget que vous seriez prêt à mettre sous garantie d'atteinte de votre objectif principal? ", '')).strip()
            ])

            mentors = get_top_mentors(texte)

            recommandations.append({
                'Etudiant': row.get('Adresse e-mail', 'Inconnu'),
                'situation_Pro': str(row.get('Quelle est ton métier et ta situation professionnelle actuelle ?', '')).strip(),
                'Top Mentors': [
                    {
                        'nom': m['nom'],
                        'url': f"/mentor/{m['nom'].replace(' ', '%20')}"
                    } for m in mentors
                ],
                'Scores': [m['score'] for m in mentors],
                'Objectif': str(row.get('Quel est ton principal objectif professionnel ?', '')).strip(),
                'Formation': str(row.get('Quelle formation serait idéale pour toi ?', '')).strip(),
                'expertise': str(row.get("Combien d'années d'expérience avez vous dans ce domaine ?" , '')).strip(),
                "disponibilites_jours": str(row.get("Quels sont les jours où tu serais disponible pour des sessions d'1h30 par semaine", "")).strip(),
                "disponibilites_periodes": str(row.get("Quelles sont les périodes de la journée où tu serais disponible?", "")).strip()
    
            })
        except Exception as e:
            print(f"Erreur ligne {_}: {e}")
            continue
  
    return recommandations



if __name__ == '__main__':
    app.run(debug=True, port=5001)