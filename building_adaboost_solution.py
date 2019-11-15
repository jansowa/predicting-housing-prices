from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from preprocessing_data import prepare_categorical_features

train_file_path = './train.csv'
train_X = pd.read_csv(train_file_path, index_col='Id')
train_X.dropna(axis=0, subset=['SalePrice'], inplace=True)
train_y = train_X.SalePrice
train_X.drop(['SalePrice'], axis=1, inplace=True)

test_data_path = './test.csv'
test_X = pd.read_csv(test_data_path, index_col='Id')

numerical_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                      'OverallQual', 'OverallCond', 'MSSubClass', 'YearRemodAdd']
categorical_features = ['MSZoning', 'ExterQual', 'CentralAir', 'KitchenQual']

test_X.fillna({'KitchenQual': 'TA', 'MSZoning': 'RL'}, inplace=True)
train_X, test_X = prepare_categorical_features(train_X, test_X, categorical_features, numerical_features)

regressor = AdaBoostRegressor(n_estimators=26, base_estimator=DecisionTreeRegressor(max_depth=20), learning_rate=1.3)
print("Fitting regressor...")
regressor.fit(train_X, train_y)

print("Predicting labels...")
test_predictions = regressor.predict(test_X)

print("Formatting data...")
output = pd.DataFrame({'Id': test_X.index,
                       'SalePrice': test_predictions})
output.to_csv('submission.csv', index=False)
