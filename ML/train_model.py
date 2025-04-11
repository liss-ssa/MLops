import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import mlflow
from mlflow.models import infer_signature
import joblib

# Установите URI для MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Загрузка данных
file_path = 'path_to_your_data/Tesla.csv'
df = pd.read_csv(file_path)

def preprocess_tesla_data(df):
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop(columns=['Adj Close'])
    X = df.drop(columns=['Close'])
    y = df['Close']
    return X, y

def scale_features(X, y):
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X.drop(columns=['Date']).values)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled, power_trans

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

X, y = preprocess_tesla_data(df)
X_scaled, y_scaled, power_trans = scale_features(X, y)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

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

rmse, mae, r2 = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)

with mlflow.start_run():
    mlflow.log_param("alpha", best_model.alpha)
    mlflow.log_param("l1_ratio", best_model.l1_ratio)
    mlflow.log_param("penalty", best_model.penalty)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    signature = infer_signature(X_train, best_model.predict(X_train))
    mlflow.sklearn.log_model(best_model, "model", signature=signature, input_example=X_train[:5])

models = {
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "XGBRegressor": XGBRegressor(random_state=42, eval_metric='rmse')
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        rmse, mae, r2 = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model", input_example=X_train[:5])
        print(f"Model: {name}")
        print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}\n")

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf_model = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_params, cv=5)
rf_grid.fit(X_train, y_train.ravel())
best_rf_model = rf_grid.best_estimator_

with mlflow.start_run(run_name="Best_RandomForestRegressor"):
    y_pred = best_rf_model.predict(X_val)
    y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
    rmse, mae, r2 = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(best_rf_model, "model", input_example=X_train[:5])
    print(f"Best RandomForestRegressor - RMSE: {rmse}, MAE: {mae}, R2: {r2}")
