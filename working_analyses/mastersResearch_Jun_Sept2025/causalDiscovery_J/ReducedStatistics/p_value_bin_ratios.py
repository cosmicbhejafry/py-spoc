from pyspoc import ReducedStatistic
import numpy as np
import contextlib
from io import StringIO
import re
from causallearn.search.ConstraintBased.PC import pc

class pValueBinRatios(ReducedStatistic):
    """
    Runs the PC algorithm on the dataset and extracts the p values from the conditional independence tests.
    """

    def __init__(self, ratio_type, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.ratio_type = ratio_type

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
                "p_value",
                "bins"
                "ratios"]

    def compute(self, data: np.ndarray) -> float:
        # Run the PC algorithm and extract the p-values from the verbose output
        # Use contextlib to not print the full verbose output
        with contextlib.redirect_stdout(StringIO()) as captured:
            cg = pc(data, alpha=self.alpha, indep_test='fisherz', verbose=True)

        verbose_output = captured.getvalue()
        p_values = re.findall(r'p-value\s+([\d\.e-]+)', verbose_output)
        p_values = [float(p) for p in p_values]
        
        bin_counts = {'strong': 0, 'moderate': 0, 'weak': 0}
        for p in p_values:
            if p < self.alpha / 10:
                bin_counts['strong'] += 1
            elif p < self.alpha:
                bin_counts['moderate'] += 1
            else:
                bin_counts['weak'] += 1
        
        if self.ratio_type == 'strong_to_moderate':
            ratio = bin_counts['strong'] / bin_counts['moderate'] if bin_counts['moderate'] > 0 else float('inf')
        elif self.ratio_type == 'strong_to_weak':
            ratio = bin_counts['strong'] / bin_counts['weak'] if bin_counts['weak'] > 0 else float('inf')
        elif self.ratio_type == 'moderate_to_weak':
            ratio = bin_counts['moderate'] / bin_counts['weak'] if bin_counts['weak'] > 0 else float('inf')

        # return ratio
        return np.array([ratio])