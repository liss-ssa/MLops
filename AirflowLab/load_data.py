import pandas as pd

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)
    # Предобработка данных
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop(columns=['Adj Close'])
    X = df.drop(columns=['Close'])
    y = df['Close']
    return X, y