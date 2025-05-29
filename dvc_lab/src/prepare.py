import pandas as pd

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # 1. Очистка данных
    df = df.dropna()
    df = df.drop(columns=['Adj Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Генерация признаков
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['PriceChange'] = df['Close'].pct_change()
    
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_data('data/raw/Tesla.csv', 'data/processed/cleaned.csv')