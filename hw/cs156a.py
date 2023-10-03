from typing import Callable

import numpy as np

### HOMEWORK 1 ################################################################

def target_function_random_line(
        *, rng: np.random.Generator = None, seed: int = None
    ) -> Callable[[float | np.ndarray[float]], float | np.ndarray[float]]:

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
        w: np.ndarray[float], x: np.ndarray[float], y: np.ndarray[float]
    ) -> float:
    
    """
    Calculates the in-sample or out-of-sample error for an algorithm
    trained on binary data.

    Parameters
    ----------
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
    
    return np.count_nonzero(np.sign(x @ w) != y, axis=0) / x.shape[0]

def perceptron(
        N: int, f: Callable, *, w: np.ndarray[float] = None, 
        N_test: int = 1_000, rng: np.random.Generator = None, seed: int = None
    ) -> tuple[int, float]:
    
    """
    Implements the perceptron learning algorithm (PLA) for a target 
    function operating on a d-dimensional data set D.

    Parameters
    ----------
    N : `int`
        Number of random data points.

    f : `function`
        Target function f as a function of the inputs x.

    w : `numpy.ndarray`, keyword-only, optional
        Hypothesis w. If not provided, a vector is initialized with all
        zeros.

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
    if w is None:
        w = np.zeros(x.shape[1], dtype=float)
    iters = 0
    while True:
        wrong = np.argwhere(np.sign(x @ w) != y)[:, 0]
        if wrong.size == 0:
            break
        i = np.random.choice(wrong)
        w += y[i] * x[i]
        iters += 1
    return iters, validate_binary(w, *generate_data(N_test, f, rng=rng))

### HOMEWORK 2 ################################################################

def coin_flip(
        n_trials: int = 1, n_coins: int = 1_000, n_flips: int = 10, *,
        rng: np.random.Generator = None, seed: int = None
    ) -> np.ndarray[float]:
    
    """
    Simulates coin flipping.

    Parameters
    ----------
    n_trials : `int`, default: 1
        Number of trials.

    n_coins : `int`, default: 1_000
        Number of coins to flip.

    n_flips : `int`, default: 10
        Number of times to flip each coin.

    rng : `numpy.random.Generator`, keyword-only, optional
        A NumPy pseudo-random number generator.

    seed : `int`, keyword-only, optional
        Random seed used to initialize a pseudo-random number generator.
        Only used if :code:`rng=None`.

    Returns
    -------
    nus : `numpy.ndarray`
        Fractions of heads obtained for the first coin flipped, a coin
        chosen randomly, and the coin that had the minimum frequency of
        heads for all trials.
    """

    if rng is None:
        rng = np.random.default_rng(seed)
    heads = np.count_nonzero(
        rng.uniform(size=(n_trials, n_coins, n_flips)) < 0.5, axis=2
    )  # [0.0, 0.5) is heads, [0.5, 1.0) is tails
    return np.stack((
        heads[:, 0],
        heads[np.arange(n_trials), rng.integers(n_coins, size=n_trials)],
        heads[np.arange(n_trials), np.argmin(heads, axis=1)],
    )) / n_flips

def hoeffding_inequality(
        N: int, eps: float | np.ndarray[float], *, M: int = 1) -> float:

    """
    Computes the probability bound using the Hoeffding inequality.

    Parameters
    ----------
    N : `int`
        Sample size.

    eps : `float` or `numpy.ndarray`
        Tolerance for |nu - mu|.

    M : `int`, keyword-only, default: 1
        Number of hypotheses.
    """

    return 2 * M * np.exp(-2 * eps ** 2 * N)

def linear_regression(
        N: int, f: Callable, *, transform: Callable = None, noise: float = 0.0,
        N_test: int = 1_000, rng: np.random.Generator = None, seed: int = None, 
        hyp: bool = False) -> tuple[np.ndarray[float], float, float]:

    """
    Implements the linear regression algorithm for a target function
    operating on a two-dimensional data set D.

    Parameters
    ----------
    N : `int`
        Number of random data points.

    f : `function`
        Target function f as a function of the inputs x.

    noise : `float`, keyword-only, default: 0.0
        Fraction of the training set to introduce noise to.

    transform : `function`, keyword-only, default: None
        Nonlinear transformation function for the inputs x.

    N_test : `int`, keyword-only, default: 1_000
        Number of random test data points.

    rng : `numpy.random.Generator`, keyword-only, optional
        A NumPy pseudo-random number generator.

    seed : `int`, keyword-only, optional
        Random seed used to initialize a pseudo-random number generator.
        Only used if :code:`rng=None`.

    hyp : `bool`, keyword-only, default: False
        Determines whether the hypothesis w is returned.

    Returns
    -------
    w : `numpy.ndarray`
        Hypothesis w. Only available if :code:`hyp=True`.

    E_in : `float`
        In-sample error E_in.

    E_out : `float`
        Out-of-sample error E_out.
    """

    if rng is None:
        rng = np.random.default_rng(seed)
    x, y = generate_data(N, f, rng=rng)
    if transform:
        x = transform(x)
    if noise:
        y[rng.choice(N, round(noise * N), False)] *= -1
    w = np.linalg.pinv(x) @ y
    
    x_test, y_test = generate_data(N_test, f, rng=rng)
    if transform:
        x_test = transform(x_test)
    if noise:
        y_test[rng.choice(N_test, round(noise * N_test), False)] *= -1
    return (w, validate_binary(w, x, y), 
            validate_binary(w, x_test, y_test))[1 - hyp:]

def target_function_hw2(
    ) -> Callable[[float | np.ndarray[float]], float | np.ndarray[float]]:

    """
    Implements the target function 
    f(x_1, x_2) = sgn(x_1^2 + x_2^2 - 0.6).

    Returns
    -------
    f : `function`
        Target function f as a function of the inputs x, which contain
        the constant bias and the x- and y-coordinates.
    """
    
    return lambda x: np.sign((x[:, 1:] ** 2).sum(axis=1) - 0.6)