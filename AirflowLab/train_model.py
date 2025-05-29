from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import mlflow
import joblib

def train_and_evaluate(X, y):
    # Масштабирование данных
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X.drop(columns=['Date']).values)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
    
    # Параметры для GridSearch
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Обучение модели
    rf_model = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf_model, rf_params, cv=5)
    rf_grid.fit(X_train, y_train.ravel())
    best_rf_model = rf_grid.best_estimator_
    
    # Оценка модели
    y_pred = best_rf_model.predict(X_val)
    y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(power_trans.inverse_transform(y_val), y_price_pred))
    mae = mean_absolute_error(power_trans.inverse_transform(y_val), y_price_pred)
    r2 = r2_score(power_trans.inverse_transform(y_val), y_price_pred)
    
    # Логирование в MLflow
    with mlflow.start_run(run_name="Best_RandomForestRegressor"):
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(best_rf_model, "model", input_example=X_train[:5])
    
    # Сохранение модели и преобразователей
    joblib.dump(best_rf_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(power_trans, 'power_trans.pkl')
    
    return rmse, mae, r2