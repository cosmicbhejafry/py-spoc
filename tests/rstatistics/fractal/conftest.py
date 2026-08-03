import pytest
import numpy as np

from functools import cache
from typing import Iterable
from math import log

from pyspoc.data.generators.fractal import mandelbrot, henon, sierpinski, koch

@cache
def _generate_mandelbrot() -> tuple[np.ndarray, float]:
    data = mandelbrot()
    return data, 2


@pytest.fixture
def mandelbrot_factory():
    return _generate_mandelbrot


@cache
def _generate_fractals() -> Iterable[tuple[np.ndarray, float]]:
    test_set = list()

    data = mandelbrot()
    test_set.append(("mandelbrot", data, 2))

    data = henon()
    test_set.append(("henon", data, 1.26))

    data = sierpinski()
    test_set.append(("sierpinskin", data, log(3) / log(2)))

    data = koch()
    test_set.append(("koch", data, log(4) / log(3)))

    return test_set

@pytest.fixture
def fractal_factory():
    return _generate_fractals
