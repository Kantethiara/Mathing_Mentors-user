from flask import Flask, render_template, request, redirect
import pandas as pd
import lightgbm as lgb
import os
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from datetime import datetime
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# === Chargement du modèle et des données mentors ===
import pickle

with open("/Users/thiarakante/Documents/Databeez/Matching-user-mentorat/Back/model/lightgbm_ranking_model.pkl", "rb") as f:
    model = pickle.load(f)


df_mentors = pd.read_excel("/Users/thiarakante/Documents/Databeez/Matching-user-mentorat/Back/data/Mentors.xlsx")
df_mentors["mentor_id"] = df_mentors.index

model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# === Fonctions utilitaires ===
def text_similarity(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    emb_a = model_embed.encode(a, convert_to_numpy=True)
    emb_b = model_embed.encode(b, convert_to_numpy=True)
    return float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))

def compute_pair_features(student, mentor):
    return {
        "competence_sim": text_similarity(student["formation"], mentor["Liste les compétences clés que tu peux transmettre à des apprenants dans ce domaine ou métier"]),
        "objectif_sim": text_similarity(student["objectif"], mentor['Domaine principal d\'expertise']),
        "carrer_sim": text_similarity(student["situation"], mentor['Quel est ton métier et ta situation professionnelle actuelle ?']),
        "expertise_sim": text_similarity(student["formation"], mentor['Domaine principal d\'expertise']),
    }

def recommend_mentors(student, top_n=3):
    features = []
    meta = []

    for _, mentor in df_mentors.iterrows():
        feats = compute_pair_features(student, mentor)
        features.append([feats["competence_sim"], feats["objectif_sim"], feats["carrer_sim"], feats["expertise_sim"]])
        meta.append({
            "nom": mentor.get("nom_mentor", ""),
            "mentor_id": mentor["mentor_id"],
            "email": mentor.get("Adresse e-mail", ""),
        })

    df_features = pd.DataFrame(features, columns=["competence_sim", "objectif_sim", "carrer_sim", "expertise_sim"])
    scores = model.predict(df_features)

    result_df = pd.DataFrame(meta)
    result_df["score"] = scores
    top = result_df.sort_values(by="score", ascending=False).head(top_n)
    return top

# === Routes ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/formulaire-individuel', methods=['GET', 'POST'])
def formulaire_individuel():
    if request.method == 'POST':
        try:
            student = {
                "email": request.form.get("email", ""),
                "objectif": request.form.get("objectif", ""),
                "formation": request.form.get("formation", ""),
                "format": request.form.get("format_formation", ""),
                "situation": request.form.get("situation", ""),
                "remarques": request.form.get("remarques", "")
            }

            top_mentors = recommend_mentors(student, top_n=3)

            return render_template('resultats_individuels.html',
                                   form_data=student,
                                   recommandations=top_mentors.to_dict(orient="records"),
                                   horodateur=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        except Exception as e:
            return render_template("formulaire_individuel.html", error=str(e))

    return render_template("formulaire_individuel.html")

@app.route('/batch-upload', methods=['GET', 'POST'])
def batch_upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template("batch_upload.html", error="Aucun fichier sélectionné.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df_students = pd.read_excel(filepath) if filename.endswith("xlsx") else pd.read_csv(filepath)
        recommandations = []

        for _, row in df_students.iterrows():
            student = {
                "email": row.get("Adresse e-mail", ""),
                "objectif": row.get("Quel est ton principal objectif professionnel ?", ""),
                "formation": row.get("Quelle formation serait idéale pour toi ?", ""),
                "format": row.get("Quel format de formation vous convient le plus? ", ""),
                "situation": row.get("Quelle est ton métier et ta situation professionnelle actuelle ?", ""),
                "remarques": row.get("Des remarques, suggestions ou questions ? ", "")
            }

            top_mentors = recommend_mentors(student, top_n=3)
            recommandations.append({
                "etudiant": student["email"],
                "objectif": student["objectif"],
                "formation": student["formation"],
                "mentors": top_mentors.to_dict(orient="records")
            })

        return render_template("resultats_batch.html",
                               recommandations=recommandations,
                               horodateur=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    return render_template("batch_upload.html")

if __name__ == '__main__':
    app.run(debug=True, port=5002)
