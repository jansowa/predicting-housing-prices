from time import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBRegressor

n_estimators_default_range = range(50, 500, 50)
n_jobs_default = 4

def calculate_min_error_for_xgb_regressor(preprocessor, X, y, n_estimators_range = n_estimators_default_range, n_jobs = n_jobs_default):
    min_error = -1
    n_estimators_with_min_error = 0
    executing_tests_time = time()

    for n_estimators_number in n_estimators_range:
        print("Calculating error for xgb with n_estimators = " + n_estimators_number)
        model = XGBRegressor(n_estimators=n_estimators_number, n_jobs=n_jobs)
        error = calculate_error_for_xgb_regressor(model, preprocessor, X, y)
        if (min_error == -1) or (error < min_error):
            min_error = error
            n_estimators_with_min_error = n_estimators_number

    executing_tests_time = time() - executing_tests_time

def calculate_error_for_xgb_regressor(model, preprocessor, X, y):
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                  ])

    scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_squared_log_error')
    return scores.mean()