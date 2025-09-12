import numpy as np
from scipy.special import expit
import GPy 
from pyspoc import ReducedStatistic


class GPLVM_Fit(ReducedStatistic):

    def __init__(self, ARD_initial_dimension = 15):

        # Calling base class initialiser.
        super().__init__()

        self.ARD_initial_dimension = ARD_initial_dimension


    @property
    def name(self) -> str:
        return "GPLVM-Fit"

    @property
    def identifier(self) -> str:
        return "my_new_reducer_identifier"

    @property
    def labels(self) -> list[str]:
        return ["my_new_reducer_label_1",
                "my_new_reducer_label_2",
                "my_new_reducer_label_n"]

    def _shuffle_rows_per_column(X: np.ndarray, seed: Optional[Union[int, np.random.Generator]] = None, copy: bool = True) -> np.ndarray:
        """
        Shuffle rows independently within each column of X.
        """
        if X.ndim != 2:
            raise ValueError("Shuffling error: X must be a 2D array of shape (n, p).")
        A = np.array(X, copy=copy)
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        n, p = A.shape
        for j in range(p):
            A[:, j] = A[rng.permutation(n), j]  # shuffle rows within column j

    return A

    def compute(self, data: np.ndarray) -> float:

        # Dimensionally reduce the data
        Y = data
        p = Y.shape[1] 

        # Create GPLVM
        dim_input = min(self.ARD_initial_dimension, p) # if there are too few features, use the number of features
        kern  = gpy.kern.RBF(input_dim=dim_input, variance=1.0, ARD=True)
        gplvm = gpy.models.BayesianGPLVM(Y=Y, input_dim=dim_input, kernel=kern)

        # Optimise
        gplvm.optimize()

        # Extract the posterior means
        Z_pred = np.asarray(gplvm.X.mean)

        # Likelihoods
        log_predictive_density = np.sum(gplvm.log_predictive_density(Z_pred, Y))
        null = []
        for _ in range(5):
            null.append(np.sum(gplvm.log_predictive_density(self._shuffle_rows_per_column(Z_pred), Y)))
        null_log_predictive_density = np.mean(null)

        # GPLVM-Fit definition
        output = expit(log_predictive_density - null_log_predictive_density)

        return output