from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def prepare_categorical_features(train_X, val_X, categorical_features, numerical_features):
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[categorical_features]))
    OH_cols_val = pd.DataFrame(OH_encoder.transform(val_X[categorical_features]))
    OH_cols_train.index = train_X.index
    OH_cols_val.index = val_X.index
    numerical_train_X = train_X.drop(list(set(train_X.columns) - set(numerical_features)), axis=1)
    numerical_val_X = val_X.drop(list(set(val_X.columns) - set(numerical_features)), axis=1)
    train_X = pd.concat([OH_cols_train, numerical_train_X], axis=1)
    val_X = pd.concat([OH_cols_val, numerical_val_X], axis=1)
    return train_X, val_X