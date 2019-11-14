# Minimum RandomForestRegressor error: 0.03573183686491808 for n_estimators = 53, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0

from sklearn.ensemble import RandomForestRegressor
import pandas as pd

train_file_path = './train.csv'
train_data = pd.read_csv(train_file_path)
train_y = train_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
train_X = train_data[features]

regressor = RandomForestRegressor(n_estimators=53, min_samples_leaf=1, min_weight_fraction_leaf=0.0)
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