import numpy as np
import pandas as pd

from random import randrange


__all__ = ["sierpinski", "henon", "mandelbrot", "koch", "circle", "line"]

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
          a: float = 1.4,
          b: float = 0.3,
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

def mandelbrot(
        iters: int = 25,
        res: int = 1000,
        p: int = 2,
        bound: int = 2) -> np.ndarray:
    
    # Setting parameters (these values can be changed)
    x_domain, y_domain = np.linspace(-2, 2, res), np.linspace(-2, 2, res)
    grid_x, grid_y = np.meshgrid(x_domain, y_domain, indexing='ij')
    xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    def func(z, p, c):
        return z**p + c
    
    # Computing 2D array to represent the Mandelbrot set
    iteration_array = np.full(shape=res**2, fill_value=False)

    for i in range(res**2):
    
        x, y = xy[i]
        z = 0
        c = complex(x, y)
    
        for j in range(iters):

            try:
                z = func(z, p, c)

                if abs(z) >= bound:
                    break

            except (ValueError, ZeroDivisionError):
                break
        
        if j == iters - 1:
            iteration_array[i] = True
                
    return xy[iteration_array]


def koch(depth: int = 5, interp_rate: int = 10) -> np.ndarray:
    """Return vertices of a Koch snowflake boundary."""
    h = np.sqrt(3) / 2

    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.0, 0.0],
    ])

    for _ in range(depth):
        points = _koch_segment(points)

    points = _sample_koch_segments(points, interp_rate)
    return points

def _koch_segment(points: np.ndarray) -> np.ndarray:
    """Replace each line segment by the four-segment Koch rule."""
    new_points = []

    angle = np.pi / 3
    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)],
    ])

    for p0, p1 in zip(points[:-1], points[1:]):
        v = p1 - p0

        a = p0
        b = p0 + v / 3
        d = p0 + 2 * v / 3
        c = b + rot @ (v / 3)

        new_points.extend([a, b, c, d])

    new_points.append(points[-1])
    return np.asarray(new_points)

def _sample_koch_segments(points: np.ndarray, points_per_segment: int = 10) -> np.ndarray:
    """Generate additional points along each of the Koch segments."""
    samples = []

    for p0, p1 in zip(points[:-1], points[1:]):
        t = np.linspace(0.0, 1.0, points_per_segment, endpoint=False)
        seg = (1.0 - t[:, None]) * p0 + t[:, None] * p1
        samples.append(seg)

    return np.vstack(samples)


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
