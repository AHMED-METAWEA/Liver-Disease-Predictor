import pickle
import pandas as pd
import numpy as np
from src.config import MODEL_PATH

# أسماء الأعمدة بالترتيب الصح
FEATURE_COLUMNS = [
    'Age of the patient',
    'Gender of the patient',
    'Total Bilirubin',
    'Direct Bilirubin',
    'Alkphos Alkaline Phosphotase',
    'Sgpt Alamine Aminotransferase',
    'Sgot Aspartate Aminotransferase',
    'Total Protiens',
    'ALB Albumin',
    'A/G Ratio Albumin and Globulin Ratio'
]

# الأعمدة اللي عملنا عليها Log Transform
SKEWED_COLS = [
    'Total Bilirubin',
    'Direct Bilirubin',
    'Alkphos Alkaline Phosphotase',
    'Sgpt Alamine Aminotransferase',
    'Sgot Aspartate Aminotransferase'
]

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        saved = pickle.load(f)
    return saved['model'], saved['scaler']

def predict(patient: dict):
    """
    patient = dict فيه بيانات المريض
    """
    model, scaler = load_model()
    
    # 1. حول الـ dict لـ DataFrame
    df = pd.DataFrame([patient], columns=FEATURE_COLUMNS)
    
    # 2. نفس الـ Log Transform اللي عملناه في التدريب
    for col in SKEWED_COLS:
        df[col] = np.log1p(df[col])
    
    # 3. نفس الـ Scaling
    df_scaled = scaler.transform(df)
    
    # 4. Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    
    result = "🔴Disease" if prediction == 1 else "🟢 Healthy"
    
    print(f"\n{'='*40}")
    print(f"result : {result}")
    print(f"Health : {probability[0]:.2%}")
    print(f"Disease : {probability[1]:.2%}")
    print(f"{'='*40}")
    
    return prediction, probability


if __name__ == "__main__":
    # مريض تجريبي
    # test_patient = {
    #     'Age of the patient': 65.0,
    #     'Gender of the patient': 0,          # 1=Male, 0=Female
    #     'Total Bilirubin': .530628,
    #     'Direct Bilirubin': 0.095310,
    #     'Alkphos Alkaline Phosphotase': 5.236442,
    #     'Sgpt Alamine Aminotransferase': 2.833213,
    #     'Sgot Aspartate Aminotransferase': 2.944439,
    #     'Total Protiens': 6.80,
    #     'ALB Albumin': 3.30,
    #     'A/G Ratio Albumin and Globulin Ratio': 1.0

    # }
    
    test_patient = {
        'Age of the patient': 65.000000,
        'Gender of the patient': 0.000000,
        'Total Bilirubin': 0.7,
        'Direct Bilirubin': 0.1,
        'Alkphos Alkaline Phosphotase': 220,
        'Sgpt Alamine Aminotransferase': 16,
        'Sgot Aspartate Aminotransferase': 18,
        'Total Protiens': 6.800000,
        'ALB Albumin': 3.300000,
        'A/G Ratio Albumin and Globulin Ratio': .900000
    }


    predict(test_patient)
    # 10.0,0.0,0.5877866649021191,0.18232155679395462,5.0689042022202315,3.091042453358316,2.833213344056216,6.1,2.8,0.8