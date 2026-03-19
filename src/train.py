import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import joblib

# PARAMETROS
DATASET = "NearMiss_equal"
ALGORITHM = "StackingClassifier"

# Load data
train_data = pd.read_csv(f"data/splits/{DATASET}_train.csv")
test_data = pd.read_csv(f"data/splits/{DATASET}_test.csv")
X_train = train_data.drop('Cover_Type', axis=1)
y_train = train_data['Cover_Type']
X_test = test_data.drop('Cover_Type', axis=1)
y_test = test_data['Cover_Type']

print(X_train.shape)

# Load Hyperparameters
with open(f'best_hyperparameters/best_hyperparameters_{ALGORITHM}_{DATASET}.json', 'r') as f:
    best_hyperparameters = json.load(f)
best_hyperparameters = best_hyperparameters[0]

print(f"Best hyperparameters loaded: {best_hyperparameters}")

# Create modelo with best hyperparameters
def make_model(algo, params):
    if algo == 'xgboost':
        return XGBClassifier(**params, eval_metric='mlogloss', n_jobs=-1, random_state=42)
    if algo == 'RandomForest':
        return RandomForestClassifier(**params, n_jobs=-1, random_state=42)
    if algo == 'SVM':
        return SVC(**params, probability=False, random_state=42)
    if algo == 'AdaBoost-RF':
        return AdaBoostClassifier(**params, estimator=DecisionTreeClassifier(max_depth=15),random_state=42)
    if algo == 'AdaBoost-SVC':
        return AdaBoostClassifier(**params, estimator=SVC(probability=True, kernel='rbf'),random_state=42)
    if algo == 'Soft-Voting-Clf':
        rf = RandomForestClassifier(
            n_estimators=params['rf_n_estimators'],
            max_depth=params['rf_max_depth'],
            min_samples_split=params['rf_min_samples_split'],
            n_jobs=-1,
            random_state=42
        )
        lr = LogisticRegression(
            C=params['lr_C'],
            max_iter=1000,
            random_state=42
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
            random_state=42
        )
        dt = DecisionTreeClassifier(
            max_depth=params['dt_max_depth'],
            random_state=42
        )
        return VotingClassifier(
            estimators=[('rf', rf), ('dt', dt)],
            voting='hard'
        )
    if algo == 'BaggingClassifier':
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=params['base_max_depth'],
                random_state=42
            ),
            n_estimators=params['n_estimators'],
            max_samples=params['max_samples'],
            max_features=params['max_features'],
            bootstrap=params['bootstrap'],
            n_jobs=-1,
            random_state=42
        )
    if algo == 'GradientBoosting-Clf':
        return GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            subsample=params['subsample'],
            max_features=params['max_features'],
            random_state=42
        )
    if algo == 'StackingClassifier':
        rf = RandomForestClassifier(
            n_estimators=params['rf_n_estimators'],
            max_depth=params['rf_max_depth'],
            n_jobs=-1,
            random_state=42
        )
        gb = GradientBoostingClassifier(
            n_estimators=params['gb_n_estimators'],
            learning_rate=params['gb_learning_rate'],
            max_depth=params['gb_max_depth'],
            random_state=42
        )
        meta_learner = LogisticRegression(
            C=params['meta_C'],
            max_iter=1000,
            random_state=42
        )
        return StackingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            final_estimator=meta_learner,
            cv=5
        )
    raise ValueError(f"Unsupported algorithm: {algo}")

model = make_model(ALGORITHM, best_hyperparameters)

# Create pipeline with scaler
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', model)
])

# Train model
print("Training model...")
pipeline.fit(X_train, y_train)

# Save model
model_path = f'models/{ALGORITHM}_{DATASET}_model.pkl'
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")