import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
import joblib

def preprocess_tesla_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop(columns=['Adj Close'])
    X = df.drop(columns=['Close'])
    y = df['Close']
    return X, y

def scale_features(X, y):
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X.drop(columns=['Date']).values)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled, power_trans

if __name__ == "__main__":
    file_path = 'Tesla.csv'
    X, y = preprocess_tesla_data(file_path)
    X_scaled, y_scaled, power_trans = scale_features(X, y)

    #данные для последующего использования
    joblib.dump(X_scaled, 'data/X_scaled.pkl')
    joblib.dump(y_scaled, 'data/y_scaled.pkl')
    joblib.dump(power_trans, 'data/power_trans.pkl')
