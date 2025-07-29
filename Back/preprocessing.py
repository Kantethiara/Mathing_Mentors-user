import numpy as np
from sentence_transformers import SentenceTransformer

# Charger le modèle SentenceTransformer
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity_np(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def safe_text(text):
    if isinstance(text, str):
        return text.strip()
    return ""

def text_similarity(a, b):
    a = safe_text(a)
    b = safe_text(b)
    if not a or not b:
        return 0.0
    emb_a = model_embed.encode(a, convert_to_numpy=True).flatten()
    emb_b = model_embed.encode(b, convert_to_numpy=True).flatten()
    return float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))

def compute_pair_features(student, mentor):
    # Vérification des clés existantes avec valeurs par défaut
    student_formation = student.get("Quelle formation serait idéale pour toi ?", "")
    mentor_competences = mentor.get("Liste les compétences clés que tu peux transmettre à des apprenants dans ce domaine ou métier", "")
    student_objectif = student.get("Quel est ton principal objectif professionnel ?", "")
    mentor_expertise = mentor.get("Domaine principal d'expertise", "")
    student_situation = student.get("Quelle est ton métier et ta situation professionnelle actuelle ?", "")
    mentor_situation = mentor.get("Quel est ton métier et ta situation professionnelle actuelle ?", "")

    return {
        "student_id": student.get('student_id', ''),
        "mentor_id": mentor.get('mentor_id', ''),
        "competence_sim": text_similarity(student_formation, mentor_competences),
        "objectif_sim": text_similarity(student_objectif, mentor_expertise),
        "carrer_sim": text_similarity(student_situation, mentor_situation),
        "expertise_sim": text_similarity(student_formation, mentor_expertise),
    }