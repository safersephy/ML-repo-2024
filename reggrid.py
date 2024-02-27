from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_diabetes
import numpy as np


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target




# Custom scorer for RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def grid_search_regression_models(X, y, metric='rmse'):
    regressors = {
        LinearRegression(): {},
        Ridge(): {'alpha': [0.1, 1.0, 10.0]},
        Lasso(): {'alpha': [0.1, 1.0, 10.0]},
        ElasticNet(): {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]},
        RandomForestRegressor(): {'n_estimators': [10, 100], 'max_features': ['sqrt', 'log2']},
        GradientBoostingRegressor(): {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
        SVR(): {'C': [1, 10], 'kernel': ['linear', 'rbf']},
        KNeighborsRegressor(): {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    }

    scoring_functions = {
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'rmse': make_scorer(rmse, greater_is_better=False),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score)
    }

    best_score = float('inf') if metric in ['mse', 'rmse', 'mae'] else float('-inf')
    best_regressor_results = {}

    for i, (reg, params) in enumerate(regressors.items()):
        print(f"Processing {i+1}/{len(regressors)}: {reg.__class__.__name__}...")
        grid_search = GridSearchCV(reg, params, cv=5, scoring=scoring_functions[metric])
        grid_search.fit(X, y)

        print(f"Evaluating {i+1}/{len(regressors)}: {reg.__class__.__name__} - Best Parameters {grid_search.best_params_}, Best Score {grid_search.best_score_}")        



        if (metric in ['mse', 'rmse', 'mae'] and grid_search.best_score_ < best_score) or \
           (metric == 'r2' and grid_search.best_score_ > best_score):
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_regressor_results = {
                'Regressor': str(reg),
                'Best Parameters': grid_search.best_params_,
                'Best Score': best_score
            }

    print("\nBest Regressor Results based on", metric, ":")
    for key, value in best_regressor_results.items():
        print(f"{key}: {value}")

    return best_model,best_regressor_results

# Example usage:
model,results = grid_search_regression_models(X, y, metric='rmse')
