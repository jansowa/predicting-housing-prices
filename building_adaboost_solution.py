# Best AdaBoostRegressor error: 0.033773077400313094 for n_estimators = 20, desion tree max depth = 7, learning_rate = 1.4

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

train_file_path = './train.csv'
train_data = pd.read_csv(train_file_path)
train_y = train_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond', 'MSSubClass', 'YearRemodAdd']

train_X = train_data[features]

regressor = AdaBoostRegressor(n_estimators=24, base_estimator=DecisionTreeRegressor(max_depth=20), learning_rate=1.55)
print("Fitting regressor...")
regressor.fit(train_X, train_y)

test_data_path = './test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]

print("Predicting labels...")
test_predictions = regressor.predict(test_X)

print("Formatting data...")
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_predictions})
output.to_csv('submission.csv', index=False)
