from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib

def train_model(input_path, model_path):
    df = pd.read_csv(input_path)
    X = df.drop(columns=['Close', 'Date'])
    y = df['Close']
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    joblib.dump(model, model_path)

if __name__ == "__main__":
    train_model('data/processed/cleaned.csv', 'models/rf_model.joblib')