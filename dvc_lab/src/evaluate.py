import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import json

def evaluate(model_path, data_path, metrics_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Close', 'Date'])
    y = df['Close']
    
    model = joblib.load(model_path)
    preds = model.predict(X)
    
    metrics = {
        'mse': mean_squared_error(y, preds),
        'rmse': np.sqrt(mean_squared_error(y, preds)),
        'mae': mean_absolute_error(y, preds),
        'r2': r2_score(y, preds)
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    evaluate('models/rf_model.joblib', 'data/processed/cleaned.csv', 'metrics/eval.json')