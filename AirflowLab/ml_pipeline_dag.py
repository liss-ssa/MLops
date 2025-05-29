from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

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

import os

def load_data():
    try:
        dag_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(dag_dir, "Tesla.csv")
        X, y = load_and_preprocess(csv_path)
        
        X['Date'] = X['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return {'X': X.to_dict(), 'y': y.to_dict()}
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def train_model(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='load_data')
    
    X = pd.DataFrame.from_dict(data['X'])
    y = pd.Series(data['y'])
    
    X['Date'] = pd.to_datetime(X['Date'])
    
    from train_model import train_and_evaluate
    rmse, mae, r2 = train_and_evaluate(X, y)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

with DAG(
    'tesla_stock_prediction',
    default_args=default_args,
    description='DAG for training Tesla stock prediction model',
    schedule='@weekly', 
    catchup=False,
) as dag:

    start = EmptyOperator(task_id='start')
    
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )
    
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    end = EmptyOperator(task_id='end')
    
    start >> load_data_task >> train_model_task >> end
