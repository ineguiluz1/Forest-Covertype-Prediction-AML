import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import optuna
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import json


ALGORITHM = 'Soft-Voting-Clf' # Options: 'xgboost', 'lightgbm', 'RandomForest', 'SVM', 'LogisticRegression', 'AdaBoost'
DATASET = 'NearMiss_equal' # file name without extension (e.g. 'equal_undersampled', 'smote_oversampled', 'original')
CV_FOLDS = 5
RANDOM_STATE = 42
N_TRIALS = 50
# LOAD DATA
data = pd.read_parquet(f'data/processed/{DATASET}.parquet')

selected_features = ['Elevation', 'Horizontal_Distance_To_Hydrology', 
                    'Horizontal_Distance_To_Roadways', 'Hillshade_Noon',
                    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
                    'Wilderness_Area3', 'Wilderness_Area4', 
                    'Soil_Type2', 'Soil_Type4', 'Soil_Type10', 'Soil_Type12',
                    'Soil_Type22', 'Soil_Type23', 'Soil_Type38', 'Soil_Type39',
                    'Euclidean_Distance_To_Hydrology', 'Distance_To_Hydrology_To_Roadways_Ratio',
                    'Total_Distance', 'Aspect_North_South', 'Cover_Type'] 

data = data[selected_features]

print(data.shape)

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
        'n_estimators': [100, 200, 300, 350, 500],
        'max_depth': [3, 5, 7, 9, 12, 15],
        'learning_rate': [0.01, 0.1, 0.2, 0.25, 0.4, 0.5],
        'subsample': [0.8, 1.0, 0.75, 0.9],
        'colsample_bytree': [0.8, 0.9, 0.85,1.0]
    }
elif ALGORITHM == 'lightgbm':
    param_grid = {
        'n_estimators': [100, 200, 300, 250, 400],
        'max_depth': [3, 5, 7, 9, 12, 15, 20],
        'learning_rate': [0.01, 0.1, 0.2, 0.25, 0.4, 0.5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 0.9, 0.85,1.0]
    }
elif ALGORITHM == 'RandomForest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif ALGORITHM == 'AdaBoost-RF':
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 300, 500],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    }
elif ALGORITHM == 'AdaBoost-SVC':
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 300, 500],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    }
elif ALGORITHM == 'Soft-Voting-Clf':
    param_grid = {
        'rf_n_estimators': [100, 200, 300],
        'rf_max_depth': [10, 20, None],
        'rf_min_samples_split': [2, 5],
        'lr_C': [0.001, 0.01, 0.1, 1],
    }
# CV splitter
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# OPTUNA STUDY
def suggest_params(trial, grid):
    params = {}
    for key, values in grid.items():
        if isinstance(values[0], (list, tuple)):  # Para weights
            params[key] = trial.suggest_categorical(key, [tuple(v) if isinstance(v, list) else v for v in values])
        else:
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
    if algo == 'AdaBoost-RF':
        return AdaBoostClassifier(**params, estimator=DecisionTreeClassifier(max_depth=15),random_state=RANDOM_STATE)
    if algo == 'AdaBoost-SVC':
        return AdaBoostClassifier(**params, estimator=SVC(probability=True, kernel='rbf'),random_state=RANDOM_STATE)
    if algo == 'Soft-Voting-Clf':
        rf = RandomForestClassifier(
            n_estimators=params['rf_n_estimators'],
            max_depth=params['rf_max_depth'],
            min_samples_split=params['rf_min_samples_split'],
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        lr = LogisticRegression(
            C=params['lr_C'],
            max_iter=1000,
            random_state=RANDOM_STATE
        )
        return VotingClassifier(
            estimators=[('rf', rf), ('lr', lr)],
            voting='soft'
        )
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

    