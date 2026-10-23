#%%
from __future__ import annotations

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# %% Standard LASSO. Alpha parameter is provided directly.
# Returns model object, scaled predictors and the scaler.
def run_ols(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    # Lasso is sensitive to feature scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()

    model.fit(X_scaled, y)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("Selected features:", np.flatnonzero(model.coef_))

    return model, X_scaled, scaler
