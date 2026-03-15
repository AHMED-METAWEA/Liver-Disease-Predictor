import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.config import PROCESSED_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

def get_X_y(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    print(f"✅ Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"✅ After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res