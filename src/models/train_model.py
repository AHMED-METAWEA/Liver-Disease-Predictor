import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.config import PROCESSED_DATA_PATH, MODEL_PATH, RANDOM_STATE
from src.features.build_features import get_X_y, split_and_scale, apply_smote

def train():
    # 1. حمل الداتا
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X, y = get_X_y(df)
    
    # 2. Split + Scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    
    # 3. SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    # 4. Cross Validation
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring='f1')
    print(f"✅ CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # 5. Train على كل الداتا
    model.fit(X_train_res, y_train_res)
    
    # 6. Evaluate
    y_pred = model.predict(X_test)
    print(f"\n📊 Test Results:")
    print(classification_report(y_test, y_pred, 
          target_names=['Healthy', 'Disease']))
    
    # 7. حفظ الموديل والـ scaler
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()