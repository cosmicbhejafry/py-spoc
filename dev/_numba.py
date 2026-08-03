# %%
from pyspoc._utils import arrays
from pyspoc.settings import settings
import numpy as np

x = np.array([1,2,np.nan])
y = np.array([1,2,np.nan])

with settings.override(verbose=True):
    arrays.array_equal(x, y, equal_nan=True)
