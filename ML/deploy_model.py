import os
import joblib
import mlflow

if __name__ == "__main__":
    # Прочитать путь из файла
    with open('best_model.txt', 'r') as f:
        path_model = f.read().strip()

    # Запуск MLflow сервиса
    os.system(f"mlflow models serve -m {path_model} -p 5003 --no-conda &")
