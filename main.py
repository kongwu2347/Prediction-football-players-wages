import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

def load_data():
    """Load training and testing datasets from remote sources."""
    print("Downloading data from remote server...")
    train_url = "https://www.dropbox.com/s/5rr2ysw6tjcnbpj/train.csv?dl=1"
    test_url = "https://www.dropbox.com/s/395thqjfxf7k4wi/test.csv?dl=1"
    try:
        return pd.read_csv(train_url), pd.read_csv(test_url)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None, None

def feature_engineering(df):
    """
    Advanced Feature Engineering: Capturing football market dynamics.
    """
    if df is None: return None
    df = df.copy()

    # 1. Differential imputation for missing values
    skills = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    gk_skills = ['gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning']
    
    for col in skills + gk_skills:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 2. Construct "Star Power" premium
    if 'international_reputation' in df.columns and 'overall' in df.columns:
        df['star_power_index'] = df['international_reputation'] * (df['overall'] ** 2)

    # 3. Construct "Growth Potential" features
    if 'potential' in df.columns and 'overall' in df.columns:
        df['growth_potential'] = df['potential'] - df['overall']
        if 'age' in df.columns:
            # Avoid division by zero by adding a small constant or base age offset
            df['potential_velocity'] = df['growth_potential'] / (df['age'] - 14 + 1)

    # 4. Aggregate physical and technical attributes
    df['physical_avg'] = df[['pace', 'physic']].mean(axis=1)
    df['technical_avg'] = df[['shooting', 'passing', 'dribbling']].mean(axis=1)

    # 5. Prime age classification
    if 'age' in df.columns:
        df['is_prime'] = df['age'].apply(lambda x: 1 if 24 <= x <= 29 else 0)
        df['is_veteran'] = df['age'].apply(lambda x: 1 if x > 32 else 0)

    return df

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna Optimization Objective Function"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'random_state': 111,
        'tree_method': 'hist',
        'n_jobs': -1
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict(X_val)
    # Return RMSE (log scale) as the optimization metric
    return np.sqrt(mean_squared_error(y_val, preds))

def run_experiment():
    # 1. Data Preparation
    train_raw, test_raw = load_data()
    if train_raw is None or test_raw is None:
        return

    train = feature_engineering(train_raw)
    test = feature_engineering(test_raw)

    target = 'wage_eur'
    drop_cols = [target, 'player_ID', 'id', 'long_name', 'short_name', 'dob']
    features = [col for col in train.columns if col not in drop_cols]

    X = train[features]
    # Log-transform target variable to handle skewness
    y = np.log1p(train[target])

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 2. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # 3. Data Split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=111)
    
    print("Performing feature preprocessing...")
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    # 4. Optuna Automatic Hyperparameter Tuning
    print("\n--- Starting Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)

    print(f"\nBest Log-Scale RMSE: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 5. Train Final Model
    print("\nTraining final model...")
    best_model = xgb.XGBRegressor(**study.best_params, random_state=111, tree_method='hist')
    best_model.fit(X_train, y_train)

    # 6. Evaluation
    log_preds = best_model.predict(X_val)
    preds = np.expm1(log_preds)
    actuals = np.expm1(y_val)

    # Calculate RMSE on the real scale
    rmse_real = np.sqrt(mean_squared_error(actuals, preds))

    print(f"\n--- Evaluation Results (Real Wage Scale) ---")
    print(f"RMSE (Root Mean Squared Error): {rmse_real:.2f} EUR")
    print(f"MAE  (Mean Absolute Error):     {mean_absolute_error(actuals, preds):.2f} EUR")
    print(f"R2   (Goodness of Fit):         {r2_score(actuals, preds):.4f}")

    # 7. Predict and Save
    X_final_test = preprocessor.transform(test[features])
    final_preds = np.expm1(best_model.predict(X_final_test))

    submission_file = "submission_optuna.csv"
    submission = pd.DataFrame({
        'Id': test['player_ID'] if 'player_ID' in test.columns else range(len(final_preds)), 
        'Predicted_Wage': final_preds
    })
    submission.to_csv(submission_file, index=False)
    print(f"\n[Done] Results saved to {submission_file}")

if __name__ == "__main__":
    run_experiment()
