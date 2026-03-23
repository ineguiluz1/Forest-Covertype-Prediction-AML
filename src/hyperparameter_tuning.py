import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import NearMiss
import optuna
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import json


ALGORITHM = 'AdaBoost-RF' # Options: 'xgboost', 'lightgbm', 'RandomForest', 'SVM', 'LogisticRegression', 'AdaBoost'
DATASET = 'train_original' # file name without extension (e.g. 'equal_undersampled', 'smote_oversampled', 'original')
CV_FOLDS = 3
RANDOM_STATE = 42
N_TRIALS = 10
# LOAD DATA
data = pd.read_parquet(f'data/splits/{DATASET}.parquet')

selected_features = [
                    'Elevation', 'Aspect', 'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
                    'Hillshade_Noon', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 
                    'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type2', 'Soil_Type4', 'Soil_Type10', 
                    'Soil_Type12', 'Soil_Type22', 'Soil_Type23', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 
                    'Euclidean_Distance_To_Hydrology', 'Distance_To_Hydrology_To_Roadways_Ratio', 'Distance_To_Fire_To_Hydrology_Ratio', 
                    'Total_Distance', 'Hydrology_Slope', 'Aspect_North_South', 'Cover_Type'
                    ] 

data = data[selected_features]

print(data.shape)

# SPLIT DATA
X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']

# Encode labels to 0..n_classes-1 (XGBoost expects that)
le = LabelEncoder()
y = le.fit_transform(y)


# HYPERPARAMETER TUNING
if ALGORITHM == 'xgboost':
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 5, 7, 9, 12, 15],
        'learning_rate': [0.01, 0.1, 0.2, 0.25, 0.4, 0.5],
        'subsample': [0.8, 1.0, 0.75, 0.9],
        'colsample_bytree': [0.8, 0.9, 0.85,1.0]
    }
elif ALGORITHM == 'lightgbm':
    param_grid = {
        'n_estimators': [100, 200, 250],
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
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    }
elif ALGORITHM == 'AdaBoost-SVC':
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    }
elif ALGORITHM == 'Soft-Voting-Clf':
    param_grid = {
        'rf_n_estimators': [100, 200],
        'rf_max_depth': [10, 20, None],
        'rf_min_samples_split': [2, 5],
        'lr_C': [0.001, 0.01, 0.1, 1],
    }
elif ALGORITHM == 'Hard-Voting-Clf':
    param_grid = {
        'rf_n_estimators': [100, 200],
        'rf_max_depth': [10, 20, None],
        'rf_min_samples_split': [2, 5],
        'dt_max_depth': [5, 10, 15],
    }
elif ALGORITHM == 'BaggingClassifier':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': [0.5, 0.7, 0.8, 1.0],
        'max_features': [0.5, 0.7, 0.8, 1.0],
        'bootstrap': [True, False],
        'base_max_depth': [5, 10, 15, 20],
    }
elif ALGORITHM == 'GradientBoostingClassifier':
    param_grid = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None],
    }
elif ALGORITHM == 'StackingClassifier':
    param_grid = {
        'rf_n_estimators': [100, 200],
        'rf_max_depth': [10, 20],
        'gb_n_estimators': [100, 150],
        'gb_learning_rate': [0.01, 0.1],
        'gb_max_depth': [3, 5],
        'meta_C': [0.1, 1, 10],
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
    if algo == 'Hard-Voting-Clf':
        rf = RandomForestClassifier(
            n_estimators=params['rf_n_estimators'],
            max_depth=params['rf_max_depth'],
            min_samples_split=params['rf_min_samples_split'],
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        dt = DecisionTreeClassifier(
            max_depth=params['dt_max_depth'],
            random_state=RANDOM_STATE
        )
        return VotingClassifier(
            estimators=[('rf', rf), ('dt', dt)],
            voting='hard'
        )
    if algo == 'BaggingClassifier':
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=params['base_max_depth'],
                random_state=RANDOM_STATE
            ),
            n_estimators=params['n_estimators'],
            max_samples=params['max_samples'],
            max_features=params['max_features'],
            bootstrap=params['bootstrap'],
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    if algo == 'GradientBoostingClassifier':
        return GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            subsample=params['subsample'],
            max_features=params['max_features'],
            random_state=RANDOM_STATE
        )
    if algo == 'StackingClassifier':
        rf = RandomForestClassifier(
            n_estimators=params['rf_n_estimators'],
            max_depth=params['rf_max_depth'],
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        gb = GradientBoostingClassifier(
            n_estimators=params['gb_n_estimators'],
            learning_rate=params['gb_learning_rate'],
            max_depth=params['gb_max_depth'],
            random_state=RANDOM_STATE
        )
        meta_learner = LogisticRegression(
            C=params['meta_C'],
            max_iter=1000,
            random_state=RANDOM_STATE
        )
        return StackingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            final_estimator=meta_learner,
            cv=5
        )
    raise ValueError(f"Unsupported algorithm: {algo}")

def objective(trial):
    params = suggest_params(trial, param_grid)
    model = make_model(ALGORITHM, params)

    pipeline = ImbPipeline([
        ('undersample', NearMiss(version=1)), 
        ('clf', model)
    ])

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        pipeline.fit(X_fold_train, y_fold_train)
        
        y_pred = pipeline.predict(X_fold_val)
        
        score = f1_score(y_fold_val, y_pred, average='macro', zero_division=0)
        fold_scores.append(score)
        
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return float(sum(fold_scores) / len(fold_scores))

pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
study = optuna.create_study(direction='maximize', pruner=pruner)
print(f'Starting hyperparameter tuning for {ALGORITHM} with NearMiss sampling...')
study.optimize(objective, n_trials=N_TRIALS)

print('Best hyperparameters:', study.best_params)
print('Best CV score (Macro-F1):', study.best_value)

# Save best hyperparameters to a file
best_params_df = pd.DataFrame([study.best_params])
best_params_df.to_json(f'best_hyperparameters/best_hyperparameters_{ALGORITHM}_{DATASET}.json', orient='records', indent=4)

    