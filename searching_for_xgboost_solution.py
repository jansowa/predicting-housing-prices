import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost.sklearn import XGBRegressor
from xgboost_regressor_comparator import calculate_error_for_xgb_regressor

train_file_path = './train.csv'

X = pd.read_csv(train_file_path, index_col='Id')
X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

numerical_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr',
                      'OverallQual', 'OverallCond', 'MSSubClass', 'YearRemodAdd', 'GrLivArea', 'BsmtFullBath', 'Fireplaces', 'GarageYrBlt',
                      'GarageCars']
categorical_features = ['MSZoning', 'CentralAir', 'KitchenQual', 'Neighborhood', 'Condition1', 'Heating', 'LandContour', 'GarageFinish']
rejected_features = list(set(X.columns) - set(numerical_features) - set(categorical_features))
X.drop(rejected_features, axis=1, inplace=True)

#Define preprocessing
numerical_transformer = SimpleImputer(strategy="mean")

categoricak_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categoricak_transformer, categorical_features)
    ]
)

#Define model
model = XGBRegressor(n_estimators=100, n_jobs=4)

print('Mean error: ' + str(calculate_error_for_xgb_regressor(model, preprocessor, X, y)))