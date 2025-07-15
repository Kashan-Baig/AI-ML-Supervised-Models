# 🧠 AI/ML Supervised Learning Projects

This repository showcases a collection of supervised machine learning projects developed using Python and Scikit-learn. Each project solves real-world regression or classification problems through complete ML pipelines — from preprocessing to deployment.

---

## 📂 Projects Included

### 🔹 Credit Risk Prediction (Classification)
- Predicts loan approval risk (High Risk / Low Risk) using borrower features.
- Includes feature engineering (log transformation, encoding, outlier removal).
- Trained and optimized using **Random Forest** and **GridSearchCV**.
- Deployed using **Flask** with:
  - Web form interface  
  - `/predict` POST API endpoint  
- Handles imbalanced data and supports real-time inference.

### 🔹 Laptop Price Prediction (Regression)
- Predicts laptop prices based on specs like processor, RAM, SSD, brand, etc.
- Preprocessing includes encoding, skewness correction, and scaling.
- Models: **Linear Regression**, **XGBoost**.
- Deployed using **Streamlit** for an interactive web UI.

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
