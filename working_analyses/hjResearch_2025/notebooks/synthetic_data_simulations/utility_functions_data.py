def plot_empirical_cov(X,fig_title):
    cov_estimator = EmpiricalCovariance().fit(X)
    cov_mat = cov_estimator.covariance_

    mask = np.triu(np.ones_like(cov_mat, dtype=bool), k=1)  

    # Extract off-diagonal, absolute positive values
    off_diag_vals = np.abs(cov_mat[mask])
    off_diag_vals = off_diag_vals[off_diag_vals > 0]  # filter out zeros
    
    plt.figure(figsize=(8, 4))
    plt.hist(off_diag_vals, bins=np.linspace(0, 1, 50), edgecolor='black')
    plt.title(fig_title)
    plt.xlabel("Covariance value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()