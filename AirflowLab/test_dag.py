from airflow import DAG
from airflow.operators.empty import EmptyOperator
from datetime import datetime

with DAG(
    "test_dag",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
) as dag:
    task = EmptyOperator(task_id="test_task")
