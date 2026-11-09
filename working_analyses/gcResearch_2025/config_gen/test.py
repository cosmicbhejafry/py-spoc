#%%
from pyspoc.config import Config

cfg = Config.from_internal("test_cfg")
#%%
import numpy as np
from pyspoc.calculator import Calculator

X = np.random.standard_normal(size=(1000,100))
calc = Calculator(X, name="yo")
calc.compute(cfg)
