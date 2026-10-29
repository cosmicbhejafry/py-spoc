"""Internal Orthogonal PCA autoencoder implementation.

The package separates model definition, estimator lifecycle, shared Statistic
parameter handling, and fitted-state containers. Concrete user-facing
Statistics are built in the corresponding ``pyspoc.rstatistics`` package.
"""
