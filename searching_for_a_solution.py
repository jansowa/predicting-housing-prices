import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from random_forest_regressor_comparator import calculate_min_error_for_random_forest_regressor
from adaboost_regressor_comparator import calculate_min_error_for_adaboost_regressor

def get_msle(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    msle = mean_squared_log_error(val_y, preds_val)
    return(msle)

train_file_path = './train.csv'

train_data = pd.read_csv(train_file_path)

y = train_data.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond', 'MSSubClass', 'YearRemodAdd']
categorical_features = ['MSZoning', 'ExterQual', 'CentralAir', 'KitchenQual']

X = train_data[features]


#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
train_X, val_X, train_y, val_y = train_test_split(X, y)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
best_tree_size = 0
best_msle = 0
for leaf_nodes in candidate_max_leaf_nodes:
    current_msle = get_msle(leaf_nodes, train_X, val_X, train_y, val_y)
    print("for leaf_nodes = " + str(leaf_nodes) + "; msle = " + str(current_msle))
    if(best_msle==0 or current_msle<best_msle):
        best_tree_size=leaf_nodes
        best_msle=current_msle

print("Validation error for best DecissionTreeRegressor: " + str(best_msle))

random_forest_model = RandomForestRegressor()
random_forest_model.fit(train_X, train_y)
preds_val = random_forest_model.predict(val_X)
msle = mean_squared_log_error(val_y, preds_val)

print("Validation error for default RandomForestRegressor: " + str(msle))

calculate_min_error_for_random_forest_regressor(train_X, train_y, val_X, val_y, n_estimators_range=range(52, 59, 2), print_parameters=True)
calculate_min_error_for_adaboost_regressor(train_X, train_y, val_X, val_y, decision_tree_max_depth_range= range(19, 22), n_estimators_range= range(23, 26), print_parameters=True)

train_X, val_X, train_y, val_y = train_test_split(X, y)
calculate_min_error_for_random_forest_regressor(train_X, train_y, val_X, val_y, n_estimators_range=range(52, 59, 2), print_parameters=True)
calculate_min_error_for_adaboost_regressor(train_X, train_y, val_X, val_y, decision_tree_max_depth_range= range(19, 22), n_estimators_range= range(23, 26), print_parameters=True)

train_X, val_X, train_y, val_y = train_test_split(X, y)
calculate_min_error_for_random_forest_regressor(train_X, train_y, val_X, val_y, n_estimators_range=range(52, 59, 2), print_parameters=True)
calculate_min_error_for_adaboost_regressor(train_X, train_y, val_X, val_y, decision_tree_max_depth_range= range(19, 22), n_estimators_range= range(23, 26), print_parameters=True)