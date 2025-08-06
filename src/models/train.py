import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

def train_model(df):
    # Separate features and target
    X = df.drop(columns=['Recycling Rate (%)'])
    y = df['Recycling Rate (%)']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline model
    baseline_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    baseline_rf.fit(X_train, y_train)

    print("Baseline Random Forest Performance:")
    evaluate_model(baseline_rf, X_test, y_test)
    
    # Hyperparameter tuning
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
        error_score='raise'
    )
    
    random_search.fit(X_train, y_train)
    print("Best Hyperparameters:")
    print(random_search.best_params_)

    tuned_rf = random_search.best_estimator_

    print("Tuned Random Forest Performance:")
    evaluate_model(tuned_rf, X_test, y_test)

    return tuned_rf
