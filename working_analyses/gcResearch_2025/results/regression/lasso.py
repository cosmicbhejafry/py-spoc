#%%
from __future__ import annotations

import pandas as pd
import numpy as np

from collections.abc import Collection
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


###############################################
# Functions
###############################################

#%% CV-based LASSO. Returns model object, scaled predictors and the scaler.
# NOTE: This isn't used currently as 16 observations isn't really sufficient for
# an accurate CV process.
def run_lasso_cv(X, y, cv):
    X = np.asarray(X)
    y = np.asarray(y)
    cv_splits = list(cv.split(X, y))

    # Lasso is sensitive to feature scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LassoCV(
        alphas=100,          # Number of automatically selected alpha values
        cv=cv_splits,        # Reusable observation splits
        max_iter=10_000,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_scaled, y)

    print("Selected alpha:", model.alpha_)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("Selected features:", np.flatnonzero(model.coef_))

    return model, X_scaled, scaler

# %% Standard LASSO. Alpha parameter is provided directly.
# Returns model object, scaled predictors and the scaler.
def run_lasso(X, y, alpha: float = 0.02):
    X = np.asarray(X)
    y = np.asarray(y)

    # Lasso is sensitive to feature scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Lasso(
        alpha = alpha,
        max_iter=10_000,
        random_state=42,
    )

    model.fit(X_scaled, y)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("Selected features:", np.flatnonzero(model.coef_))

    return model, X_scaled, scaler

#%%
def run_lasso_alpha_sweep(X, y, alphas: Collection[float]
    ) -> dict[str, np.ndarray | pd.DataFrame]:

    non_zero_coefs = list()
    selected_features = list()
    n_coefs = list()
    n_alphas = len(alphas)
    maes = np.full(shape=n_alphas, fill_value=-1.0)
    rmses = np.full(shape=n_alphas, fill_value=-1.0)
    r2s = np.full(shape=n_alphas, fill_value=-1.0)
    max_n_coef = 0

    for i, alpha in enumerate(alphas):
        model, X_scaled, scaler = run_lasso(X, y, alpha)
        selected_features.append(np.flatnonzero(model.coef_))
        n_coef = np.flatnonzero(model.coef_).shape[0]
        n_coefs.append(n_coef)
        non_zero_coefs.append(model.coef_[np.flatnonzero(model.coef_)])

        if n_coef > max_n_coef:
            max_n_coef = n_coef

        y_pred = model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        maes[i] = mae

        rmse = root_mean_squared_error(y, y_pred)
        rmses[i] = rmse

        r2 = r2_score(y, y_pred)
        r2s[i] = r2

    non_zero_coef_arr = np.zeros(shape=(n_alphas, max_n_coef))
    selected_features_arr = np.zeros(shape=(n_alphas, max_n_coef))

    for i in range(n_alphas):
        n_coef = n_coefs[i]
        non_zero_coef_arr[i,:n_coef] = non_zero_coefs[i]
        selected_features_arr[i,:n_coef] = selected_features[i]

    metrics = np.vstack((
        maes,
        rmses,
        r2s))

    metrics_df = pd.DataFrame(
        data = metrics,
        columns=pd.Index(data=alphas, name="Alpha"),
        index=pd.Index(data=["MAE", "RMSE", "R2"], name="Metric"))

    return {
        "Metrics": metrics_df,
        "Non-Zero Coefficents": non_zero_coef_arr,
        "Selected Features": selected_features_arr}
