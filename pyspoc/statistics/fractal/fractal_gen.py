from random import randrange

import numpy as np

def __midpoint(
    first: tuple[int | float, int | float],
    second: tuple[int | float, int | float]
) -> tuple[int | float, int | float]:
    """Return the midpoint between two points.
    Args:
        first (tuple[int | float, int | float]): First point.
        second (tuple[int | float, int | float]): Second point.
    Returns:
        tuple[int | float, int | float]: Midpoint between `first` and `second`.
    """
    return (
        (first[0] + second[0]) / 2,
        (first[1] + second[1]) / 2
    )

def sierpinski(iters: int = 25000) -> np.ndarray:
    """Generate a Sierpinski triangle fractal using the midpoint method.
    Args:
        iters (int, optional): Number of iterations to compute the fractal.
            Defaults to 25_000.
        dot_color (str, optional): Color of the plotted points. Defaults to
            "black".
    """
    # Triangle definition and plot:

    data = np.empty(shape=((iters + 1) * 3, 2))
    a = (0,0)
    b = (1,0)
    c = (0.5, 0.8660254037844386)
    vertices = (a,b,c)
    data[0] = a
    data[1] = b
    data[2] = c

    # Initial coordinate setting and plot:
    initial = ((a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3)

    # Fractal computation and plot:
    current = initial

    for i in range(iters):
        pivot = vertices[randrange(0, 3)]
        current = __midpoint(current, pivot)
        j = 3 * (i + 1)
        data[j:j+3] = current

    # Return fractal
    return data

def henon(iters: int = 25000, 
          a: float = 1, 
          b: float = 1, 
          x_start: float = 0.05, 
          y_start: float = 0.1) -> np.ndarray:
    
    # Initialisation
    data = np.empty(shape=((iters + 1), 2))
    x = x_start
    y = y_start
    data[0] = (x,y)

    # Map generation
    for i in range(iters):
        x = 1 - a * x ** 2 + y
        y = b * x
        data[i+1] = (x,y)

    #Return fractal
    return data