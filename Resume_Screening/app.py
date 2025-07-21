from flask import Flask, request, render_template
import re
import joblib

app = Flask(__name__)

# Load vectorizer and model
vectorizer = joblib.load("tfid.pkl")
model = joblib.load("catboost.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Resume cleaning function
def clean_resume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"UTF-?8", " ", text)
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.route('/', methods=['GET', 'POST'])
def resume_form():
    prediction = None
    cleaned_text = None
    original_text = None
    confidence = None
    top_predictions = []

    if request.method == 'POST':
        original_text = request.form['resume']
        cleaned_text = clean_resume(original_text)

        # Check if text is too short to be meaningful
        if len(cleaned_text.split()) < 10:
            prediction = "Please enter a more detailed resume or project description."
        else:
            vectorized = vectorizer.transform([cleaned_text])
            probs = model.predict_proba(vectorized)[0]

            # Get Top 3 predictions
            top3 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
            top_predictions = [
                {
                    'label': label_encoder.inverse_transform([idx])[0],
                    'confidence': round(prob * 100, 2)
                }
                for idx, prob in top3
            ]

            max_prob = top3[0][1]
            if max_prob < 0.1:
                prediction = "Prediction is uncertain – input may not match any known category."
            else:
                prediction = top_predictions[0]['label']
                confidence = top_predictions[0]['confidence']

    return render_template('index.html',
                           original_text=original_text,
                           cleaned_text=cleaned_text,
                           prediction=prediction,
                           confidence=confidence,
                           top_predictions=top_predictions)

if __name__ == '__main__':
    app.run(debug=True)
