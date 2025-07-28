import pandas as pd 
import joblib
from flask import Flask,render_template,request
from string import punctuation
import unicodedata
import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

vectorizer = joblib.load("tfid.pkl")
model = joblib.load("lr.pkl")


def clean(text):
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'-?\b\d+(\.\d+)?\b|-?\b\.\d+\b', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[^a-zA-Z\s]+\b', ' ', text)  
    text = re.sub(r'\b(\w)\1{2,}\w*\b', ' ', text)
    text = re.sub(r'\b(?=\w*\d)(?=\w*[a-zA-Z])\w+\b', '', text)
    text = text.lower()
    punc = list(punctuation)
    tokens = word_tokenize(text)
    stop = list(stopwords.words("english"))+punc
    words = [words for words in tokens if words not in stop]
    lemm = WordNetLemmatizer()
    cleaned = [lemm.lemmatize(word) for word in words]
    return ' '.join(cleaned)


@app.route('/', methods=['GET', 'POST'])
def hotel_rev():
    prediction = None
    cleaned_text = None
    original_text = None

    if request.method == 'POST':
        original_text = request.form['resume']
        cleaned_text = clean(original_text)

        tokens = cleaned_text.split()
        known_words = set(vectorizer.vocabulary_.keys())
        matched = sum(1 for word in tokens if word in known_words)
        match_ratio = matched / len(tokens) if tokens else 0


        if match_ratio < 0.3:
            prediction = "Your review contains too many unknown or irrelevant words."
        else:
            vect = vectorizer.transform([cleaned_text])
            prediction = model.predict(vect)
            clamped_rating = max(0, min(5, prediction[0]))
            prediction = round(clamped_rating, 1)

            
    return render_template('index.html',
                           original_text=original_text,
                           prediction=prediction
                           )

if __name__ == '__main__':
    app.run(debug=True)

