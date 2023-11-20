import pandas as pd
import numpy as np
import warnings
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import mean_squared_error, r2_score
from mlflow import log_artifacts, log_param, log_metric
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 400)
train = pd.read_csv('./data/raw/train.csv')
test = pd.read_csv('./data/raw/test.csv')
sample = pd.read_csv('./data/raw/sample_submission.csv')

mlflow.set_tracking_uri('http://109.168.187.83:5000/')
mlflow.set_experiment('mlflow_test_kaggle_august')
mlflow.autolog()

with mlflow.start_run():

    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    df = pd.concat([train, test], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(train.drop(['ID_LAT_LON_YEAR_WEEK', 'emission', 'year'], axis=1), train['emission'], random_state=42)

    i = 0

    result_shap = pd.DataFrame()

    feature_after_shap = X_train.columns.to_list()
    result_shap['cols_name'] = X_train.columns.to_list()

    feature_after_shap_old = []

    cat_features = X_train.select_dtypes(include='object').columns.tolist()

    params = {
        'boosting': 'gbdt',
        'objective': 'regression',
        'num_leaves': 10,
        "max_depth": 9,
        'learnnig_rage': 0.01,
        "n_estimators": 5000,
        'verbose': -1
    }

    feature_after_shap_old = feature_after_shap

    model_cb = LGBMRegressor(**params)

    model_cb.fit(X_train[feature_after_shap], y_train,
                 eval_set=(X_test[feature_after_shap], y_test)
                 )

    loss = model_cb.predict(X_test)
    mse = mean_squared_error(y_test, loss, squared=False)
    rmse = mean_squared_error(y_test, loss)
    r2 = r2_score(y_test, loss)


    # log metrics
    mlflow.log_metrics({"mse": mse, "mae": rmse, 'r2': r2})

