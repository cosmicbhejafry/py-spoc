import numpy as np
import pandas as pd
from scipy import odr, stats
import matplotlib.pyplot as plt


def longest_consecutive_sequence(nums):
    """
    Find the longest consecutive sequence in a sorted list of numbers.

    Parameters:
        nums (list): A list of sorted numbers.

    Returns:
        list: Longest consecutive sequence.
    """
    nums = list(nums)
    if not nums:
        return []

    longest_sequence = []
    current_sequence = [nums[0]]

    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:  # Consecutive number
            current_sequence.append(nums[i])
        else:  # Sequence breaks
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence
            current_sequence = [nums[i]]  # Start a new sequence

    # Final check after the loop
    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence

    return longest_sequence


def find_ranges_pct(cs, ss, pct=0.01):
    """
    Find ranges using a threshold-based method.

    Parameters:
        cs (ndarray): Array of box counts.
        ss (ndarray): Array of scales.
        pct (float): Percentage threshold.

    Returns:
        tuple: Filtered counts, scales, and their indices.
    """
    max_box = np.max(cs)
    th0 = pct
    th1 = 1.00 - pct

    idx0 = (
        np.where(cs <= th0 * max_box)[0][-1] + 1 if np.any(cs <= th0 * max_box) else 0
    )
    idx1 = np.where(cs >= th1 * max_box)[0][0]

    xt = cs[idx0 : idx1 + 1]
    st = ss[idx0 : idx1 + 1]
    return xt, st, idx0, idx1


def find_ranges_ls(cs, ss, use_log=True):
    """
    Find ranges using a least-squares regression method.

    Parameters:
        cs (ndarray): Array of box counts.
        ss (ndarray): Array of scales.
        use_log (bool): Whether to use log-transformed values. Defaults to True.

    Returns:
        tuple: Filtered counts, scales, and their indices.
    """
    max_box = np.max(cs)
    idx0 = 0
    idx1 = np.where(cs == max_box)[0][0]

    xt = cs[idx0 : idx1 + 1]
    st = ss[idx0 : idx1 + 1]
    idxs = np.arange(idx0, idx1 + 1)

    if use_log:
        zt = np.log10(xt)
    else:
        zt = xt

    slope, intercept, _, _, _ = stats.linregress(-np.log10(st), zt)

    yt = (
        np.power(10, slope * (-np.log10(st)) + intercept)
        if use_log
        else slope * (-np.log10(st)) + intercept
    )
    residuals = xt - yt

    valid_indices = np.where(residuals > 0)[0]
    if len(valid_indices) == 0:
        return xt, st, idx0, idx1

    sequence = longest_consecutive_sequence(valid_indices)
    idx0, idx1 = sequence[0], sequence[-1]

    return xt[idx0 : idx1 + 1], st[idx0 : idx1 + 1], idxs[idx0], idxs[idx1]


def odr_line(B, x):
    """
    Define a line for orthogonal distance regression.

    Parameters:
        B (array-like): Coefficients [intercept, slope].
        x (ndarray): Input x-values.

    Returns:
        ndarray: Computed y-values.
    """
    return B[1] * x + B[0]


def perform_odr(x, y):
    """
    Perform orthogonal distance regression (ODR) for a line.

    Parameters:
        x (ndarray): Input x-values.
        y (ndarray): Input y-values.

    Returns:
        ODR output: Regression results.
    """
    model = odr.Model(odr_line)
    data = odr.Data(x, y)
    odr_instance = odr.ODR(data, model, beta0=[0.0, 0.0])
    return odr_instance.run()


def get_reflex(cs_new, return_cost=False):
    """
    Detect reflex point based on least-squares regression.

    Parameters:
        cs_new (ndarray): Log-transformed counts and scales.
        return_cost (bool): Whether to return regression costs.

    Returns:
        dict: Reflex point index and optionally costs.
    """
    x = cs_new[:, 0]
    y = cs_new[:, 1]

    costs = []
    indices = range(2, len(x) - 2)

    for i in indices:
        segments = [(x[:i], y[:i]), (x[i:], y[i:])]
        total_cost = 0

        for segment_x, segment_y in segments:
            regression = perform_odr(segment_x, segment_y)
            total_cost += regression.sum_square

        costs.append(total_cost)

    min_idx = np.argmin(costs)
    reflex_idx = indices[min_idx]

    result = {"min_idx": reflex_idx}
    if return_cost:
        result["cost"] = costs

    return result


def plot_grid(xt, idx, s, use_lim=True, figsize=(9, 8)):
    m = int(1 / s) + 1

    fig = plt.figure(figsize=figsize)
    plt.plot(xt[:, 0], xt[:, 1], linewidth=1, c="r")
    for i in range(m):
        plt.plot([i * s, i * s], [0, 1], linewidth=0.5, c="b")
        plt.plot([0, 1], [i * s, i * s], linewidth=0.5, c="b")

    plt.scatter(idx[:, 0] * s + 0.5 * s, idx[:, 1] * s + 0.5 * s, marker=".")
    if use_lim:
        plt.xlim([0, 1])
        plt.ylim([0, 1])

    plt.title("{}x{} boxes".format(m, m), fontdict={"fontsize": 20})

    return fig


def gen_circle(n=5000):
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


def gen_line(xs0, n=5000, normalization=True):
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


# https://github.com/TheAlgorithms/Python/blob/master/fractals/koch_snowflake.py
# ===================================
# Koch Snowflake Generator
# ===================================


def rotate(vector, angle_deg):
    """Rotate a vector by an angle."""
    theta = np.radians(angle_deg)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rotation_matrix @ vector


def iteration_step(vectors):
    """Perform a single Koch snowflake iteration."""
    new_vectors = []
    for i, start in enumerate(vectors[:-1]):
        end = vectors[i + 1]
        diff = (end - start) / 3
        new_vectors += [
            start,
            start + diff,
            start + diff + rotate(diff, 60),
            start + 2 * diff,
        ]
    new_vectors.append(vectors[-1])
    return new_vectors


def iterate(vectors, steps):
    """Iterate Koch snowflake generation."""
    for _ in range(steps):
        vectors = iteration_step(vectors)
    return vectors


def gen_koch(steps=4, shape="snowflake"):
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

    vectors = iterate(initial_vectors, steps)
    df = pd.DataFrame(vectors, columns=["x", "y"])
    md = max(df.x.max() - df.x.min(), df.y.max() - df.y.min())
    eps = 1e-10
    df["nx"] = (df.x - df.x.min()) / md * (1 - eps) + 0.5 * eps
    df["ny"] = (df.y - df.y.min()) / md * (1 - eps) + 0.5 * eps
    return df[["nx", "ny"]].values