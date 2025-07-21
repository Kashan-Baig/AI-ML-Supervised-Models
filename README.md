# 🧠 AI/ML Supervised Learning Projects

This repository showcases a collection of supervised machine learning projects developed using Python and Scikit-learn. Each project solves real-world regression or classification problems through complete ML pipelines — from preprocessing to deployment.

---

## 📂 Projects Included

### 🔹 Resume Category Predictor (NLP Classification)
- Classifies resumes or job descriptions into roles (e.g., Data Scientist, Web Developer, HR, etc.).
- Built using **Natural Language Processing (NLP)** techniques:
  - Text cleaning, tokenization, stopword removal, TF-IDF vectorization.
- Trained using **CatBoost & Pycaret** and evaluated on accuracy & F1-score.
- Deployed using **Flask** with:
  - Interactive web form input  
  - `/predict` API for integration
- Includes input validation and feedback for short or incomplete resumes.

---

### 🔹 Credit Risk Prediction (Classification)
- Predicts loan approval risk (High Risk / Low Risk) using borrower features.
- Includes feature engineering (log transformation, encoding, outlier removal).
- Trained and optimized using **Random Forest** and **GridSearchCV**.
- Deployed using **Flask** with:
  - Web form interface  
  - `/predict` POST API endpoint  
- Handles imbalanced data and supports real-time inference.

---

### 🔹 Laptop Price Prediction (Regression)
- Predicts laptop prices based on specs like processor, RAM, SSD, brand, etc.
- Preprocessing includes encoding, skewness correction, and scaling.
- Models: **Linear Regression**, **XGBoost**.
- Deployed using **Streamlit** for an interactive web UI.

---

### 🔹 California Housing Price Prediction (Regression)
- Predicts property prices using the California Housing dataset.
- Applied log transformations and scaling.
- Compared **Ridge Regression** and **Random Forest**.
- Evaluation using **RMSE** and **MAE**.

---

## ⚙️ Technologies Used
- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- Flask, Streamlit
- Matplotlib, Seaborn
- TF-IDF, Regex, NLP preprocessing
- GridSearchCV for hyperparameter tuning

---

## 🚀 How to Run

Each project folder includes:
- Jupyter Notebook (`.ipynb`) for data analysis and model training
- Deployment script (`app.py` or `streamlit_app.py`)
- Example inputs and usage guide

---

## 📌 Note
This repo focuses on **supervised learning workflows**, including both regression and classification tasks. It highlights **model interpretability**, **deployment**, and **real-time predictions**.

---

## 📬 Contact
Feel free to open an issue or reach out if you have questions or suggestions!
