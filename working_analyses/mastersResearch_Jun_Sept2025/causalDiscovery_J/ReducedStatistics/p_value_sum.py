from pyspoc import ReducedStatistic
import numpy as np
import contextlib
from io import StringIO
import re
from causallearn.search.ConstraintBased.PC import pc

class pValueSum(ReducedStatistic):
    """
    Runs the PC algorithm on the dataset and extracts the p values from the conditional independence tests.
    """

    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha

    @property
    def name(self) -> str:
        return "my_new_reducer_name"

    @property
    def identifier(self) -> str:
        return "my_new_reducer_identifier"

    @property
    def labels(self) -> list[str]:
        return ["causal",
                "pc_algorithm",
                "p_value"]

    def compute(self, data: np.ndarray) -> float:
        # Run the PC algorithm and extract the p-values from the verbose output
        # Use contextlib to not print the full verbose output
        with contextlib.redirect_stdout(StringIO()) as captured:
            cg = pc(data, alpha=self.alpha, indep_test='fisherz', verbose=True)

        verbose_output = captured.getvalue()
        p_values = re.findall(r'p-value\s+([\d\.e-]+)', verbose_output)
        p_values = [float(p) for p in p_values]

        # return np.array(np.sum(p_values))
        return np.sum(p_values)