import mlflow

# Установите URI для MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Запуск модели как сервиса
model_uri = "best_model.pkl" 
mlflow.models.serve(model_uri=model_uri, host='0.0.0.0', port=5000)
