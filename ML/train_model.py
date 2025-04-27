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
    base_path = "\\wsl.localhost\Ubuntu\var\lib\jenkins\workspace\Download\ML"
    X_scaled = joblib.load(os.path.join(base_path, 'X_scaled.pkl'))
    y_scaled = joblib.load(os.path.join(base_path, 'y_scaled.pkl'))
    power_trans = joblib.load(os.path.join(base_path, 'power_trans.pkl'))

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

    mlflow.set_experiment('Default')
    with mlflow.start_run():
        mlflow.log_param("alpha", best_model.alpha)
        mlflow.log_param("l1_ratio", best_model.l1_ratio)
        mlflow.log_param("penalty", best_model.penalty)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature, input_example=X_train[:5])

    joblib.dump(best_model, os.path.join(base_path, 'best_model.pkl'))
    with open(os.path.join(base_path, 'best_model.txt'), 'w') as f:
        f.write('best_model.pkl')
