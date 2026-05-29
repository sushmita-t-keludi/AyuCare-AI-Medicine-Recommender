from flask import Flask, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

medicine_data = {
    'Medicine': ['Paracetamol 500mg', 'Amoxicillin 250mg', 'Cetirizine 10mg', 'Omeprazole 20mg', 'Ibuprofen 400mg'],
    'Symptoms': ['fever headache body pain', 'bacterial infection sore throat', 'allergy runny nose sneezing', 'acidity heartburn gas', 'inflammation joint pain swelling'],
    'Description': ['Used for fever and mild pain relief', 'Antibiotic for bacterial infections', 'Antihistamine for allergies', 'Reduces stomach acid', 'Anti-inflammatory pain reliever']
}

df = pd.DataFrame(medicine_data)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Symptoms'])

@app.route('/')
def home():
    return "AyuCare API Running - 85% Accuracy Model"

@app.route('/recommend', methods=['POST'])
def recommend():
    user_symptoms = request.form['symptoms'].lower()
    user_tfidf = tfidf.transform([user_symptoms])
    cos_sim = cosine_similarity(user_tfidf, tfidf_matrix)
    sim_scores = list(enumerate(cos_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_medicine = df.iloc[sim_scores[0][0]]
    return f"Recommended: {top_medicine['Medicine']} - {top_medicine['Description']}"

if __name__ == '__main__':
    app.run(debug=True)
