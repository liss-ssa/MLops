import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from mlflow.models import infer_signature
import os

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_model(X_train, y_train, X_val, y_val, power_trans):
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        'penalty': ['l1', 'l2', 'elasticnet'],
    }

    lr = SGDRegressor(random_state=42)
    clf = GridSearchCV(lr, params, cv=5)
    clf.fit(X_train, y_train.ravel())
    best_model = clf.best_estimator_

    y_pred = best_model.predict(X_val)
    y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))

    return best_model, *eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)

if __name__ == "__main__":
    # Убедитесь, что пути соответствуют вашей структуре проекта
    base_path = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(base_path, exist_ok=True)
    
    # Загрузка предобработанных данных
    X_scaled = joblib.load(os.path.join(base_path, 'X_scaled.pkl'))
    y_scaled = joblib.load(os.path.join(base_path, 'y_scaled.pkl'))
    power_trans = joblib.load(os.path.join(base_path, 'power_trans.pkl'))

    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42
    )

    # Настройка MLflow
    mlflow.set_tracking_uri("http://localhost:5000")  # или ваш URL MLflow
    mlflow.set_experiment('SGD_Regression')
    
    # Обучение и логирование модели
    with mlflow.start_run():
        model, rmse, mae, r2, predictions = train_model(
            X_train, y_train, X_val, y_val, power_trans
        )
        
        # Логирование параметров и метрик
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
        
        # Логирование модели
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model, 
            "model", 
            signature=signature,
            input_example=X_train[:5]
        )
        
        # Сохранение модели
        model_path = os.path.join(base_path, 'best_model.pkl')
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        # Сохранение информации о модели
        with open(os.path.join(base_path, 'model_info.txt'), 'w') as f:
            f.write(f"Model: SGDRegressor\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"R2: {r2}\n")
            f.write(f"Saved to: {model_path}\n")

    print(f"Training completed. Model saved to {model_path}")
