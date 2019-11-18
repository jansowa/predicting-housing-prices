from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from preprocessing_data import prepare_categorical_features
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

train_file_path = './train.csv'
train_X = pd.read_csv(train_file_path, index_col='Id')
train_X.dropna(axis=0, subset=['SalePrice'], inplace=True)
train_y = train_X.SalePrice
train_X.drop(['SalePrice'], axis=1, inplace=True)

test_data_path = './test.csv'
test_X = pd.read_csv(test_data_path, index_col='Id')

numerical_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                      'OverallQual', 'OverallCond', 'MSSubClass', 'YearRemodAdd']
categorical_features = ['MSZoning', 'CentralAir', 'KitchenQual', 'Neighborhood', 'Condition1', 'Heating']

numerical_transformer = SimpleImputer(strategy="mean")
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

regressor = AdaBoostRegressor(n_estimators=26, base_estimator=DecisionTreeRegressor(max_depth=20), learning_rate=1.36)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', regressor)
                              ])

# test_X.fillna({'KitchenQual': 'TA', 'MSZoning': 'RL'}, inplace=True)
# train_X, test_X = prepare_categorical_features(train_X, test_X, categorical_features, numerical_features)

print("Fitting regressor...")
# regressor.fit(train_X, train_y)
my_pipeline.fit(train_X, train_y)

print("Predicting labels...")
# test_predictions = regressor.predict(test_X)
test_predictions = my_pipeline.predict(test_X)

print("Formatting data...")
output = pd.DataFrame({'Id': test_X.index,
                       'SalePrice': test_predictions})
output.to_csv('submission.csv', index=False)
