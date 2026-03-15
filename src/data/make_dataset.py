import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_data():
    df = pd.read_csv(RAW_DATA_PATH, encoding='latin-1')
    print(f"✅ Loaded: {df.shape}")
    return df

def clean_data(df):
    # 1. نظف أسماء الأعمدة
    df.columns = df.columns.str.strip()
    
    # 2. امسح المكرر
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"✅ Removed {before - df.shape[0]} duplicates")
    
    # 3. حول Gender لأرقام
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Gender of the patient'] = le.fit_transform(
        df['Gender of the patient'].astype(str)
    )
    
    # 4. KNN Imputer
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(
        imputer.fit_transform(df),
        columns=df.columns
    )
    print(f"✅ Missing after impute: {df.isnull().sum().sum()}")
    
    # 5. Log Transform للأعمدة الـ Skewed
    skewed_cols = [
        'Total Bilirubin',
        'Direct Bilirubin',
        'Alkphos Alkaline Phosphotase',
        'Sgpt Alamine Aminotransferase',
        'Sgot Aspartate Aminotransferase'
    ]
    for col in skewed_cols:
        df[col] = np.log1p(df[col])
    
    # 6. غير الـ Target: 2 → 0
    df['Result'] = df['Result'].replace({2.0: 0.0})
    
    print(f"✅ Clean data shape: {df.shape}")
    return df

def save_data(df):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Saved to {PROCESSED_DATA_PATH}")