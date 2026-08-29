import numpy as np
import pandas as pd

from random import randrange


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
        current = _sierpinski_midpoint(current, pivot)
        j = 3 * (i + 1)
        data[j:j+3] = current

    # Return fractal
    return data

def _sierpinski_midpoint(
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
        data[i+1,0] = 1 - a * x ** 2 + y
        data[i+1,1] = b * x
        x,y = data[i+1]

    # Return fractal
    return data


def mandelbrot(iters: int = 250,
               res: int = 500,
               p: int = 2,
               c: complex = complex(0.1,0.1)):
    
    # Setting parameters (these values can be changed)
    res = 500
    x_domain, y_domain = np.linspace(-2, 2, res), np.linspace(-2, 2, res)
    bound = 2

    def func(z, p, c):
        return z**p + c
    
    # Computing 2D array to represent the Mandelbrot set
    iteration_array = np.empty(shape=(res, res))

    for i, x in enumerate(x_domain):
        
        for j, y in enumerate(y_domain):
            z = 0
            c = complex(x, y)
        
            for iteration in range(iters):
                
                if abs(z) >= bound:
                    iteration_array[i,j] = iteration
                    break
                else:
                    try:
                        z = func(z, p, c)
                    except (ValueError, ZeroDivisionError):
                        z = c
            else:
                iteration_array[i+1,j] = 0
                
    return x_domain, y_domain, iteration_array.T


def mandelbrot_grid(iters: int = 250,
                    res: int = 500,
                    p: int = 2,
                    bound: int = 2,
                    c: complex = complex(0.1,0.1)):
    
    # Setting parameters (these values can be changed)
    res = 500
    x_domain, y_domain = np.linspace(-2, 2, res), np.linspace(-2, 2, res)
    
    def func(z, p, c):
        return z**p + c

    # Computing 2D array to represent the Mandelbrot set
    iteration_array = np.empty(shape=(res, res))

    for i, x in enumerate(x_domain):
        
        for j, y in enumerate(y_domain):
            z = 0
            c = complex(x, y)
        
            for iteration in range(iters):
                
                if abs(z) >= bound:
                    iteration_array[i,j] = iteration
                    break
                else:
                    try:
                        z = func(z, p, c)
                    except (ValueError, ZeroDivisionError):
                        z = c
            else:
                iteration_array[i+1,j] = 0
                
    return x_domain, y_domain, iteration_array.T


def koch(steps=4, shape="snowflake"):
    """Generate a Koch snowflake or curve."""
    initial_vectors = (
        [
            np.array([0, 0]),
            np.array([0.5, np.sqrt(3) / 2]),
            np.array([1, 0]),
            np.array([0, 0]),
        ]
        if shape == "snowflake"
        else [np.array([0, 0]), np.array([1, 0])]
    )
    
    for _ in range(steps):
        vectors = _koch_iteration_step(initial_vectors)

    df = pd.DataFrame(vectors, columns=["x", "y"])
    md = max(df.x.max() - df.x.min(), df.y.max() - df.y.min())
    eps = 1e-10
    df["nx"] = (df.x - df.x.min()) / md * (1 - eps) + 0.5 * eps
    df["ny"] = (df.y - df.y.min()) / md * (1 - eps) + 0.5 * eps
    return df[["nx", "ny"]].values

def _koch_rotate(vector, angle_deg):
    """Rotate a vector by an angle."""
    theta = np.radians(angle_deg)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rotation_matrix @ vector

def _koch_iteration_step(vectors):
    """Perform a single Koch snowflake iteration."""
    new_vectors = []
    for i, start in enumerate(vectors[:-1]):
        end = vectors[i + 1]
        diff = (end - start) / 3
        new_vectors += [
            start,
            start + diff,
            start + diff + _koch_rotate(diff, 60),
            start + 2 * diff,
        ]
    new_vectors.append(vectors[-1])
    return new_vectors


def circle(n=5000):
    v = 2 * np.pi / n
    t = np.array(range(n))
    x = np.sin(v * t) + 1
    y = np.cos(v * t) + 1
    eps = 1e-10

    df = pd.DataFrame([x, y]).T
    df.columns = ["x", "y"]
    md = max(df.x.max() - df.x.min(), df.y.max() - df.y.min())
    df["nx"] = (df.x - df.x.min()) / md * (1 - eps) + 0.5 * eps
    df["ny"] = (df.y - df.y.min()) / md * (1 - eps) + 0.5 * eps

    xs = df[["nx", "ny"]].values
    return xs


def line(xs0, n=5000, normalization=True):
    xs = []
    for t in np.array(xs0).T:
        xs.append(np.linspace(t[0], t[1], n))
    eps = 1e-10

    df = pd.DataFrame(xs).T
    df.columns = ["x", "y"]
    md = max(df.x.max() - df.x.min(), df.y.max() - df.y.min())
    df["nx"] = (df.x - df.x.min()) / md * (1 - eps) + 0.5 * eps
    df["ny"] = (df.y - df.y.min()) / md * (1 - eps) + 0.5 * eps

    if normalization is True:
        xs = df[["nx", "ny"]].values
    else:
        xs = df[["x", "y"]].values
    return xs
