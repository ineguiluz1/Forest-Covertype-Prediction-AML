import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import optuna
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import json


ALGORITHM = 'RandomForest' # Options: 'xgboost', 'lightgbm', 'RandomForest', 'SVM', 'LogisticRegression', 'AdaBoost'
DATASET = 'equal_undersampled' # file name without extension (e.g. 'equal_undersampled', 'smote_oversampled', 'original')
CV_FOLDS = 5
RANDOM_STATE = 42
N_TRIALS = 50
# LOAD DATA
data = pd.read_parquet(f'data/processed/{DATASET}.parquet')

# SPLIT DATA
X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']

# Encode labels to 0..n_classes-1 (XGBoost expects that)
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
)

train_df = X_train.copy()
train_df['Cover_Type'] = y_train
test_df = X_test.copy()
test_df['Cover_Type'] = y_test

train_df.to_csv(f'data/splits/{DATASET}_train.csv', index=False)
test_df.to_csv(f'data/splits/{DATASET}_test.csv', index=False)


# HYPERPARAMETER TUNING
if ALGORITHM == 'xgboost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
elif ALGORITHM == 'lightgbm':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
elif ALGORITHM == 'RandomForest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif ALGORITHM == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
elif ALGORITHM == 'LogisticRegression':
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
elif ALGORITHM == 'AdaBoost':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
# CV splitter
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# OPTUNA STUDY
def suggest_params(trial, grid):
    params = {}
    for key, values in grid.items():
        # use categorical suggestion for list choices
        params[key] = trial.suggest_categorical(key, values)
    return params

def make_model(algo, params):
    if algo == 'xgboost':
        return XGBClassifier(**params, eval_metric='mlogloss', n_jobs=-1, random_state=RANDOM_STATE)
    if algo == 'lightgbm':
        return LGBMClassifier(**params, n_jobs=-1, random_state=RANDOM_STATE)
    if algo == 'RandomForest':
        return RandomForestClassifier(**params, n_jobs=-1, random_state=RANDOM_STATE)
    if algo == 'SVM':
        return SVC(**params, probability=False, random_state=RANDOM_STATE)
    if algo == 'LogisticRegression':
        return LogisticRegression(**params, random_state=RANDOM_STATE, max_iter=1000)
    if algo == 'AdaBoost':
        return AdaBoostClassifier(**params, random_state=RANDOM_STATE)
    raise ValueError(f"Unsupported algorithm: {algo}")

def objective(trial):
    params = suggest_params(trial, param_grid)
    model = make_model(ALGORITHM, params)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

    # cross-validated score on the training set (scaling happens inside pipeline)
    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1_weighted', n_jobs=-1)
    return float(scores.mean())

study = optuna.create_study(direction='maximize')
print('Starting hyperparameter tuning with cross-validation...')
study.optimize(objective, n_trials=N_TRIALS)

print('Best hyperparameters:', study.best_params)
print('Best CV score (train):', study.best_value)

# Save best hyperparameters to a file
best_params_df = pd.DataFrame([study.best_params])
best_params_df.to_json(f'best_hyperparameters/best_hyperparameters_{ALGORITHM}_{DATASET}.json', orient='records', indent=4)

    