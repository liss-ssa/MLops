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

if __name__ == "__main__":
    # Устанавливаем пути в пределах workspace Jenkins
    workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlruns_path = os.path.join(workspace_path, "mlruns")
    os.makedirs(mlruns_path, exist_ok=True, mode=0o777)
    
    # Устанавливаем tracking URI перед всеми другими операциями MLflow
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    mlflow.set_experiment('Tesla_Stock_Price')

    # Загружаем данные
    base_path = os.path.join(workspace_path, "ML")
    X_scaled = joblib.load(os.path.join(base_path, 'X_scaled.pkl'))
    y_scaled = joblib.load(os.path.join(base_path, 'y_scaled.pkl'))
    power_trans = joblib.load(os.path.join(base_path, 'power_trans.pkl'))

    # Разделяем данные
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42)

    # Параметры и обучение модели
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        'penalty': ['l1', 'l2', 'elasticnet'],
    }

    lr = SGDRegressor(random_state=42)
    clf = GridSearchCV(lr, params, cv=5)
    clf.fit(X_train, y_train.ravel())
    best_model = clf.best_estimator_

    # Предсказания и метрики
    y_pred = best_model.predict(X_val)
    y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
    rmse, mae, r2 = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)

    # Логирование в MLflow
    with mlflow.start_run():
        mlflow.log_params({
            "alpha": best_model.alpha,
            "l1_ratio": best_model.l1_ratio,
            "penalty": best_model.penalty
        })
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
        
        # Сохраняем модель
        signature = infer_signature(X_train, best_model.predict(X_train))
        try:
            mlflow.sklearn.log_model(
                best_model, 
                "model", 
                signature=signature, 
                input_example=X_train[:5]
            )
        except Exception as e:
            print(f"Error logging model to MLflow: {e}")
            # Fallback - сохраняем модель локально
            model_path = os.path.join(base_path, "model")
            os.makedirs(model_path, exist_ok=True)
            mlflow.sklearn.save_model(
                best_model,
                path=model_path,
                signature=signature,
                input_example=X_train[:5]
            )

    # Сохраняем модель отдельно
    joblib.dump(best_model, os.path.join(base_path, 'best_model.pkl'))
    
    # Сохраняем информацию о модели
    with open(os.path.join(base_path, 'best_model.txt'), 'w') as f:
        f.write(f'Model: SGDRegressor\n')
        f.write(f'RMSE: {rmse}\n')
        f.write(f'MAE: {mae}\n')
        f.write(f'R2: {r2}\n')
        f.write(f'Model saved to: {base_path}/best_model.pkl\n')
