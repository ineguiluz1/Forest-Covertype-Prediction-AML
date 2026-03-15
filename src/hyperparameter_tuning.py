import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import optuna

# PARAMETERS
ALGORITHM = ''
DATASET = ''

# LOAD DATA
data = pd.read_csv(f'../data/processed/{DATASET}.csv')

# SPLIT DATA
X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    
# OPTUNA STUDY
def objective(trial, params):
    if ALGORITHM == 'xgboost':
        model = XGBClassifier(**params)
    elif ALGORITHM == 'lightgbm':
        model = LGBMClassifier(**params)
    elif ALGORITHM == 'RandomForest':
        model = RandomForestClassifier(**params)
    elif ALGORITHM == 'SVM':
        model = SVC(**params)
    elif ALGORITHM == 'LogisticRegression':
        model = LogisticRegression(**params)
    elif ALGORITHM == 'AdaBoost':
        model = AdaBoostClassifier(**params)
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score
    