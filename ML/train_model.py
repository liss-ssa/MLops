import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from mlflow.models import infer_signature

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def load_data(directory):
    """Загрузка предобработанных данных с проверкой существования файлов"""
    required_files = ['X_scaled.pkl', 'y_scaled.pkl', 'power_trans.pkl']
    for file in required_files:
        if not os.path.exists(os.path.join(directory, file)):
            raise FileNotFoundError(f"Required file {file} not found in {directory}")
    
    return (
        joblib.load(os.path.join(directory, 'X_scaled.pkl')),
        joblib.load(os.path.join(directory, 'y_scaled.pkl')),
        joblib.load(os.path.join(directory, 'power_trans.pkl'))
    )

if __name__ == "__main__":
    try:
        # Путь к директории ML (где лежат .pkl файлы)
        ml_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Loading data from: {ml_dir}")
        
        # Загрузка данных
        X_scaled, y_scaled, power_trans = load_data(ml_dir)
        
        # Разделение данных
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=0.3, random_state=42
        )

        # Параметры для GridSearch
        params = {
            'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
            'l1_ratio': [0.001, 0.05, 0.01, 0.2],
            'penalty': ['l1', 'l2', 'elasticnet'],
        }

        print("Training SGDRegressor model...")
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=5)
        clf.fit(X_train, y_train.ravel())
        best_model = clf.best_estimator_

        # Предсказание и оценка
        y_pred = best_model.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        rmse, mae, r2 = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)

        print(f"Model trained with metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

        # Настройка MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment('SGD_Regression')
        
        with mlflow.start_run():
            # Логирование параметров
            mlflow.log_params({
                'alpha': best_model.alpha,
                'l1_ratio': best_model.l1_ratio,
                'penalty': best_model.penalty
            })
            
            # Логирование метрик
            mlflow.log_metrics({
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
            
            # Логирование модели
            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(
                best_model, 
                "model", 
                signature=signature,
                input_example=X_train[:5]
            )

            # Сохранение модели в ту же директорию
            model_path = os.path.join(ml_dir, 'best_model.pkl')
            joblib.dump(best_model, model_path)
            print(f"Model saved to: {model_path}")

    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise
