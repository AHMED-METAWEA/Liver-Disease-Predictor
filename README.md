# 🫀 Liver Disease Prediction — ML Pipeline

A complete machine learning pipeline to predict liver disease from patient lab results, with an interactive Streamlit web app for clinical reference.

---

## 📌 Project Overview

This project builds a binary classifier to detect liver disease using the **Indian Liver Patient Dataset**. It covers the full ML lifecycle: data preprocessing, feature engineering, model training with cross-validation, and a deployable web interface.

| | |
|---|---|
| **Task** | Binary Classification (Disease / Healthy) |
| **Algorithm** | Random Forest Classifier |
| **Imbalance Handling** | SMOTE (Synthetic Minority Oversampling) |
| **Evaluation Metric** | F1-Score (Stratified K-Fold CV) |
| **Web App** | Streamlit |

---

## 📁 Project Structure

```
ML_PROJECTS_PIPELINE/
│
├── data/
│   ├── raw/                    # Original dataset
│   │   └── Liver Patient Dataset.csv
│   └── processed/              # Cleaned dataset
│       └── liver_clean.csv
│
├── models/
│   ├── model.pkl               # Trained model + scaler
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── overfitting_check.png
│
├── notebooks/
│   └── explore.ipynb           # EDA & experimentation
│
├── src/
│   ├── data/
│   │   └── make_dataset.py     # Data cleaning & preprocessing
│   ├── features/
│   │   └── build_features.py   # Feature engineering (Log Transform, Scaling, SMOTE)
│   └── models/
│       ├── train_model.py      # Model training & evaluation
│       └── predict_model.py    # Inference on new patients
│
├── streamlit_app.py            # Interactive web application
├── main.py                     # Pipeline entry point
├── config.py                   # Paths & hyperparameters
└── requirements.txt
```

---

## ⚙️ Features Used

| Feature | Description |
|---|---|
| Age | Patient age |
| Gender | Male / Female |
| Total Bilirubin | Liver function marker |
| Direct Bilirubin | Liver function marker |
| Alkaline Phosphotase | Enzyme level |
| Sgpt Alamine Aminotransferase | Liver enzyme |
| Sgot Aspartate Aminotransferase | Liver enzyme |
| Total Proteins | Protein level |
| Albumin | Protein synthesized by liver |
| Albumin/Globulin Ratio | Protein balance ratio |

> ⚠️ Skewed features (Bilirubin, Phosphotase, Aminotransferases) are Log-transformed before scaling.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ML_Projects_Pipeline.git
cd ML_Projects_Pipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline

```bash
python main.py
```

This will:
- Clean and preprocess the raw data
- Engineer features and apply SMOTE
- Train the Random Forest model with 5-Fold Cross Validation
- Save the model to `models/model.pkl`

### 4. Launch the Web App

```bash
streamlit run streamlit_app.py
```

---

## 🧪 Model Performance

The model is evaluated using **Stratified K-Fold Cross Validation (k=5)** with F1-Score as the primary metric, and a final evaluation on a held-out test set.

```
✅ CV F1: reported after training
📊 Test Results: see classification_report output
```

---

## 🖥️ Web App Demo

The Streamlit app allows clinicians to enter patient lab values and get an instant prediction:

- 🔴 **Liver Disease Detected** — consult a specialist
- 🟢 **No Liver Disease Detected** — continue routine monitoring

> ⚠️ For clinical reference only — not a substitute for professional medical advice.

---

## 📦 Dataset

- **Source:** [Indian Liver Patient Dataset — UCI / Kaggle](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records)
- **Samples:** 583 patients
- **Classes:** 1 = Liver Disease, 2 = Healthy

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
