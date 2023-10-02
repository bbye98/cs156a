from typing import Callable

import numpy as np

### HOMEWORK 1 ################################################################

def target_function_random_line(
        *, rng: np.random.Generator = None, seed: int = None
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:

    """
    Implements the target function f(x_1, x_2) = m * x_1 + b, where m
    and b are the m and y-intersect, respectively, of a random line, 
    that classifies the outputs of a two-dimensional data set D as 
    either -1 or 1, depending on which side of the line they fall on.

    Parameters
    ----------
    rng : `numpy.random.Generator`, keyword-only, optional
        A NumPy pseudo-random number generator.

    seed : `int`, keyword-only, optional
        Random seed used to initialize a pseudo-random number generator.
        Only used if :code:`rng=None`.

    Returns
    -------
    f : `function`
        Target function f as a function of the inputs x, which contain
        the constant bias and the x- and y-coordinates.
    """
    
    if rng is None:
        rng = np.random.default_rng(seed)
    line = rng.uniform(-1, 1, (2, 2))
    return lambda x: np.sign(
        x[:, 2] - line[0, 1] 
        - np.divide(*(line[1] - line[0])[::-1]) * (x[:, 1] - line[0, 0])
    )

def generate_data(
        N: int, f: Callable, d: int = 2, lb: float = -1.0, ub: float = 1.0, *,
        rng: np.random.Generator = None, seed: int = None
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:

    """
    Randomly generate a d-dimensional data set D in X = [lb, ub]^d,
    append a vector of ones as the first column for the constant bias
    term, and determine its outputs using the target function f.

    Parameters
    ----------
    N : `int`
        Number of random data points.

    f : `function`
        Target function f as a function of the inputs x.

    d : `int`, default: 2
        Dimensionality of the data.

    lb : `float`, default: -1.0
        Data lower bound.
    
    ub : `float`, default: 1.0
        Data upper bound.

    rng : `numpy.random.Generator`, keyword-only, optional
        A NumPy pseudo-random number generator.

    seed : `int`, keyword-only, optional
        Random seed used to initialize a pseudo-random number generator.
        Only used if :code:`rng=None`.

    Returns
    -------
    inputs : `numpy.ndarray`
        Inputs x_n.

    outputs : `numpy.ndarray`
        Outputs y_n.
    """

    if rng is None:
        rng = np.random.default_rng(seed)
    x = np.hstack((np.ones((N, 1)), rng.uniform(lb, ub, (N, d))))
    return x, f(x)

def validate_binary(
        N: int, w: np.ndarray[float], x: np.ndarray[float], 
        y: np.ndarray[float]) -> float:
    
    """
    Calculates the in-sample or out-of-sample error for an algorithm
    trained on binary data.

    Parameters
    ----------
    N : `int`
        Number of random data points.

    w : `numpy.ndarray`
        Hypothesis w.

    x : `numpy.ndarray`
        Inputs x_n.

    y : `numpy.ndarray`
        Outputs y_n.

    Returns
    -------
    error : `float`
        In-sample or out-of-sample error.
    """
    
    return np.count_nonzero(np.sign(x @ w) != y, axis=0) / N

def perceptron(
        N: int, f: Callable, *, N_test: int = 1_000, 
        rng: np.random.Generator = None, seed: int = None) -> tuple[int, float]:
    
    """
    Implements the perceptron learning algorithm (PLA) for a target 
    function operating on a two-dimensional data set D.

    Parameters
    ----------
    N : `int`
        Number of random data points.

    f : `function`
        Target function f as a function of the inputs x.

    N_test : `int`, keyword-only, default: 1_000
        Number of random test data points.

    rng : `numpy.random.Generator`, keyword-only, optional
        A NumPy pseudo-random number generator.

    seed : `int`, keyword-only, optional
        Random seed used to initialize a pseudo-random number generator.
        Only used if :code:`rng=None`.

    Returns
    -------
    iters : `int`
        Number of iterations required for the PLA to converge to a
        formula g.

    prob : `float`
        Estimated misclassification rate P[f(x) != g(x)].
    """

    if rng is None:
        rng = np.random.default_rng(seed)
    x, y = generate_data(N, f, rng=rng)
    w = np.zeros(3, dtype=float)
    iters = 0
    while True:
        wrong = np.argwhere(np.sign(x @ w) != y)[:, 0]
        if wrong.size == 0:
            break
        i = np.random.choice(wrong)
        w += y[i] * x[i]
        iters += 1
    return iters, validate_binary(N_test, w, *generate_data(N_test, f, rng=rng))