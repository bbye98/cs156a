from typing import Callable

import numpy as np
from scipy import optimize
from sklearn import svm

### HOMEWORK 1 ################################################################

def target_function_random_line(
        *, rng: np.random.Generator = None, seed: int = None
    ) -> Callable[[np.ndarray[float]], np.ndarray[float]]:

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
        Only used if `rng=None`.

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
        x[:, -1] - line[0, 1] 
        - np.divide(*(line[1] - line[0])[::-1]) * (x[:, -2] - line[0, 0])
    )

def generate_data(
        N: int, f: Callable[[np.ndarray[float]], np.ndarray[float]], 
        d: int = 2, lb: float = -1.0, ub: float = 1.0, *, 
        bias: bool = False, rng: np.random.Generator = None, seed: int = None
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:

    """
    Randomly generate a d-dimensional data set D in X = [lb, ub]^d and
    determine its outputs using the target function f.

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

    bias : `bool`, default: False
        Determines whether a vector of ones is appended as the first
        column for the constant bias term.

    rng : `numpy.random.Generator`, keyword-only, optional
        A NumPy pseudo-random number generator.

    seed : `int`, keyword-only, optional
        Random seed used to initialize a pseudo-random number generator.
        Only used if `rng=None`.

    Returns
    -------
    inputs : `numpy.ndarray`
        Inputs x_n.

    outputs : `numpy.ndarray`
        Outputs y_n.
    """

    if rng is None:
        rng = np.random.default_rng(seed)
    x = rng.uniform(lb, ub, (N, d))
    if bias:
        x = np.hstack((np.ones((N, 1)), x))
    return x, f(x)

def validate_binary(
        w: np.ndarray[float], x: np.ndarray[float], y: np.ndarray[float]
    ) -> float:
    
    """
    Calculates the in-sample or out-of-sample error for the weights of a
    regression model trained on binary data.

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
        N: int = None,
        f: Callable[[np.ndarray[float]], np.ndarray[float]] = None, 
        vf: Callable[[np.ndarray[float], np.ndarray[float], np.ndarray[float]],
                     np.ndarray[float]] = None,
        *, w: np.ndarray[float] = None, x: np.ndarray[float] = None, 
        y: np.ndarray[float] = None, N_test: int = 1_000, 
        x_test: np.ndarray[float] = None, y_test: np.ndarray[float] = None,
        rng: np.random.Generator = None, seed: int = None, hyp: bool = False
    ) -> tuple[int, float]:
    
    """
    Implements the perceptron learning algorithm (PLA) for a target 
    function operating on a d-dimensional data set D.

    Parameters
    ----------
    N : `int`, optional
        Number of random data points.

    f : `function`, optional
        Target function f as a function of the inputs x.

    vf : `function`, optional
        Validation function as a function of w, x, and y.

    w : `numpy.ndarray`, keyword-only, optional
        Hypothesis w. If not provided, a vector is initialized with all
        zeros.

    x : `numpy.ndarray`, keyword-only, optional
        Inputs x_n.

    y : `numpy.ndarray`, keyword-only, optional
        Outputs y_n.

    N_test : `int`, keyword-only, default: 1_000
        Number of random test data points.

    x_test : `numpy.ndarray`, keyword-only, optional
        Test inputs x_n.

    y_test : `numpy.ndarray`, keyword-only, optional
        Test outputs y_n.

    rng : `numpy.random.Generator`, keyword-only, optional
        A NumPy pseudo-random number generator.

    seed : `int`, keyword-only, optional
        Random seed used to initialize a pseudo-random number generator.
        Only used if `rng=None`.

    hyp : `bool`, keyword-only, default: False
        Determines whether the hypothesis w is returned.
        
    Returns
    -------
    iters : `int`
        Number of iterations required for the PLA to converge to a
        formula g.

    prob : `float`
        Estimated misclassification rate P[f(x) != g(x)]. Only returned
        if `vf` is provided.
    """

    if rng is None:
        rng = np.random.default_rng(seed)
    if x is None:
        x, y = generate_data(N, f, bias=True, rng=rng)
    elif y is None:
        y = f(x)
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

    if vf is None:
        return (w, iters)[1 - hyp:]

    if x_test is None:
        x_test, y_test = generate_data(N_test, f, bias=True, rng=rng)
    elif y_test is None:
        y_test = f(x_test)
    return (w, iters, vf(w, x_test, y_test))[1 - hyp:]

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
        Only used if `rng=None`.

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
    ) # [0.0, 0.5) is heads, [0.5, 1.0) is tails
    i = np.arange(n_trials)
    return np.stack((
        heads[:, 0],
        heads[i, rng.integers(n_coins, size=n_trials)],
        heads[i, np.argmin(heads, axis=1)],
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
        N: int = None, 
        f: Callable[[np.ndarray[float]], np.ndarray[float]] = None,
        vf: Callable[[np.ndarray[float], np.ndarray[float], np.ndarray[float]],
                     np.ndarray[float]] = None,
        *, x: np.ndarray[float] = None, y: np.ndarray[float] = None,
        transform: Callable[[np.ndarray[float]], np.ndarray[float]] = None, 
        noise: tuple[float, 
                     Callable[[np.ndarray[float]], np.ndarray[float]]] = None,
        regularization: str = None, N_test: int = 1_000, 
        x_test: np.ndarray[float] = None, y_test: np.ndarray[float] = None,
        x_validate: np.ndarray[float] = None, 
        y_validate: np.ndarray[float] = None, rng: np.random.Generator = None,
        seed: int = None, hyp: bool = False, **kwargs
    ) -> tuple[np.ndarray[float], float | tuple[float], float]:

    """
    Implements the linear regression algorithm for a target function
    operating on a d-dimensional data set D.

    Parameters
    ----------
    N : `int`, optional
        Number of random data points.

    f : `function`, optional
        Target function f as a function of the inputs x.

    vf : `function`, optional
        Validation function as a function of w, x, and y.

    x : `numpy.ndarray`, keyword-only, optional
        Inputs x_n. If a nonlinear transformation is expected, the
        original inputs should be provided here and the transformation
        function in `transform`.

    y : `numpy.ndarray`, keyword-only, optional
        Outputs y_n. If noise is to be introduced, the original outputs
        should be provided here, and the noise fraction and function in
        `noise`.

    transform : `function`, keyword-only, default: None
        Nonlinear transformation function for the inputs x.

    noise : `tuple`, keyword-only, default: 0.0
        Fraction of outputs y to introduce noise to and the
        transformation function.

    regularization : `str`, optional
        Regularization function for the hypothesis w.
        
        **Valid values**: 
        
        * :code:`"weight_decay"`: Weight decay regularization. Specify
          lambda in keyword argument "wd_lambda".

    N_test : `int`, keyword-only, default: 1_000
        Number of random test data points.

    x_test : `numpy.ndarray`, keyword-only, optional
        Test inputs x_n. If a nonlinear transformation is expected, the
        original inputs should be provided here and the transformation
        function in `transform`.

    y_test : `numpy.ndarray`, keyword-only, optional
        Test outputs y_n. If noise is to be introduced, the original 
        outputs should be provided here, and the noise fraction and 
        function in `noise`.

    x_validate : `numpy.ndarray`, keyword-only, optional
        Validation inputs x_n. If a nonlinear transformation is 
        expected, the original inputs should be provided here and the
        transformation function in `transform`. If not specified, no
        validation is performed.

    y_validate : `numpy.ndarray`, keyword-only, optional
        Validation outputs y_n. If noise is to be introduced, the 
        original outputs should be provided here, and the noise fraction
        and function in `noise`. If not specified, no validation is
        performed.

    rng : `numpy.random.Generator`, keyword-only, optional
        A NumPy pseudo-random number generator.

    seed : `int`, keyword-only, optional
        Random seed used to initialize a pseudo-random number generator.
        Only used if `rng=None`.

    hyp : `bool`, keyword-only, default: False
        Determines whether the hypothesis w is returned.

    Returns
    -------
    w : `numpy.ndarray`
        Hypothesis w. Only available if `hyp=True`.

    E_in : `float` or `tuple`
        In-sample error E_in. If `x_validate` and `y_validate` are
        provided, the errors for the test and validation sets are
        returned here. Only returned if `vf` is provided.

    E_out : `float`
        Out-of-sample error E_out. Only returned if `vf` is provided.
    """

    if rng is None:
        rng = np.random.default_rng(seed)
    if x is None:
        x, y = generate_data(N, f, bias=True, rng=rng)
    elif y is None:
        N = x.shape[0]
        y = f(x)
    else:
        N = x.shape[0]
    if transform:
        x = transform(x)
    if noise:
        i = rng.choice(N, round(noise[0] * N), False)
        y[i] = noise[1](y[i])

    if regularization is None:
        w = np.linalg.pinv(x) @ y
    elif regularization == "weight_decay":
        w = np.linalg.inv(
            x.T @ x + kwargs["wd_lambda"] * np.eye(x.shape[1], dtype=float)
        ) @ x.T @ y
    
    if vf is None:
        return w
    
    if x_test is None or y_test is None:
        if f is None:
            return (w, vf(w, x, y))[1 - hyp:]
        if x_test is None:
            x_test, y_test = generate_data(N_test, f, bias=True, rng=rng)
        elif y_test is None:
            N_test = x_test.shape[0]
            y_test = f(x_test)
    else:
        N_test = x_test.shape[0]
    if transform:
        x_test = transform(x_test)
    if noise:
        i = rng.choice(N_test, round(noise[0] * N_test), False)
        y_test[i] = noise[1](y_test[i])

    if x_validate is None or y_validate is None:
        return (w, vf(w, x, y), vf(w, x_test, y_test))[1 - hyp:]
    
    N_validate = len(y_validate)
    if transform:
        x_validate = transform(x_validate)
    if noise:
        i = rng.choice(N_validate, round(noise[0] * N_validate), False)
        y_validate[i] = noise[1](y_validate[i])
    return (w, (vf(w, x, y), vf(w, x_validate, y_validate)), 
            vf(w, x_test, y_test))[1 - hyp:]

def target_function_hw2() -> Callable[[np.ndarray[float]], np.ndarray[float]]:

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

### HOMEWORK 4 ################################################################

def vapnik_chervonenkis_bound(
        m_H: Callable[[np.ndarray[float]], np.ndarray[float]],
        N: np.ndarray[float], delta: float) -> np.ndarray[float]:

    """
    Computes the generalization error bound(s) using the
    Vapnik–Chervonenkis (VC) bound.

    Parameters
    ----------
    m_H : `function`
        Growth function m_H as a function of N.
    
    N : `numpy.ndarray`
        Sample size(s). Must be provided as floating-point numbers to
        prevent integer overflow.

    delta : `float`
        Confidence parameter delta.

    Returns
    -------
    eps : `numpy.ndarray`
        Generalization error bound(s) |E_out - E_in|.
    """

    return np.sqrt(8 * np.log(4 * m_H(2 * N) / delta) / N)

def rademacher_bound(
        m_H: Callable[[np.ndarray[float]], np.ndarray[float]],
        N: np.ndarray[float], delta: float) -> np.ndarray[float]:
    
    """
    Computes the generalization error bound(s) using the 
    Rademacher penalty bound.

    Parameters
    ----------
    m_H : `function`
        Growth function m_H as a function of N.
    
    N : `numpy.ndarray`
        Sample size(s). Must be provided as floating-point numbers to
        prevent integer overflow.

    delta : `float`
        Confidence parameter delta.

    Returns
    -------
    eps : `numpy.ndarray`
        Generalization error bound(s) |E_out - E_in|.
    """

    return (np.sqrt(2 * np.log(2 * N * m_H(N)) / N) 
            + np.sqrt(2 * np.log(1 / delta) / N) + 1 / N)

def parrondo_van_den_broek_bound(
        m_H: Callable[[np.ndarray[float]], np.ndarray[float]],
        N: np.ndarray[float], delta: float, *, ub: float = 10.0
    ) -> np.ndarray[float]:
    
    """
    Computes the generalization error bound(s) using the 
    Parrondo and Van den Broek bound.

    Parameters
    ----------
    m_H : `function`
        Growth function m_H as a function of N.
    
    N : `numpy.ndarray`
        Sample size(s). Must be provided as floating-point numbers to
        prevent integer overflow.

    delta : `float`
        Confidence parameter delta.

    ub : `float`, keyword-only, default: 10
        Upper bound for TOM Algorithm 748 bracket. A greater value may
        be needed to evaluate the error bounds at smaller sample sizes.

    Returns
    -------
    eps : `numpy.ndarray`
        Generalization error bound(s) |E_out - E_in|.
    """

    return np.vectorize(
        lambda N: optimize.root_scalar(
            lambda eps: np.sqrt((2 * eps + np.log(6 * m_H(2 * N) / delta)) / N)
                        - eps, 
            bracket=(0.0, ub), method="toms748"
        ).root
    )(N)

def devroye_bound(
        m_H: Callable[[np.ndarray[float]], np.ndarray[float]],
        N: np.ndarray[float], delta: float, *, ub: float = 10.0,
        log: bool = False) -> np.ndarray[float]:

    """
    Computes the generalization error bound(s) using the 
    Devroye bound.

    Parameters
    ----------
    m_H : `function`
        Growth function m_H as a function of N. If `log=True`, the 
        logarithm of the growth function, i.e., log(m_H), should be
        provided here instead.
    
    N : `numpy.ndarray`
        Sample size(s). Must be provided as floating-point numbers to
        prevent integer overflow.

    delta : `float`
        Confidence parameter delta.

    ub : `float`, keyword-only, default: 10
        Upper bound for TOM Algorithm 748 bracket. A greater value may
        be needed to evaluate the error bounds at smaller sample sizes.

    log : `bool`, keyword-only, default: False
        Specifies whether the logarithm of the growth function is
        provided instead in `m_H`.

    Returns
    -------
    eps : `numpy.ndarray`
        Generalization error bound(s) |E_out - E_in|.
    """

    func = lambda eps, N: np.sqrt(
        (4 * eps * (1 + eps) + np.log(4 / delta) 
         + (m_H(N ** 2) if log else np.log(m_H(N ** 2)))) / (2 * N)
    ) - eps
    return np.vectorize(
        lambda N: optimize.root_scalar(func, args=N, bracket=(0.0, ub), 
                                       method="toms748").root
    )(N)

### HOMEWORK 5 ################################################################

def gradient_descent(
        E: Callable[[np.ndarray[float]], float],
        dE: Callable[[np.ndarray[float]], np.ndarray[float]],
        x: np.ndarray[float], *, eta: float = 0.1, tol: float = 1e-14,
        max_iters: int = 100_000) -> tuple[np.ndarray[float], int]:

    """
    Implements the gradient descent algorithm.

    Parameters
    ----------
    E : `function`
        Error function E(x).

    dE : `function`
        Gradient of the error function dE/dx.

    x : `numpy.ndarray`
        Initial guess x_0.

    eta : `float`, keyword-only, default: 0.1
        Learning rate eta.

    tol : `float`, keyword-only, default: 1e-14
        Tolerance for E(x).

    max_iters : `int`, keyword-only, default: 100_000
        Maximum number of iterations.

    Returns
    -------
    x : `numpy.ndarray`
        Optimal x.

    iters : `int`
        Number of iterations required for the gradient descent algorithm
        to converge to x.
    """

    iters = 0
    while E(x) > tol and iters < max_iters:
        x -= eta * dE(x)
        iters += 1
    return x, iters

def coordinate_descent(
        E: Callable[[np.ndarray[float]], float],
        dE_dx: tuple[Callable[[np.ndarray[float]], float]],
        x: np.ndarray[float], *, eta: float = 0.1, tol: float = 1e-14,
        max_iters: int = 100_000) -> tuple[np.ndarray[float], int]:
    
    """
    Implements the coordinate descent algorithm.

    Parameters
    ----------
    E : `function`
        Error function E(x).

    dE_dx : `tuple`
        Partial derivatives of the error function dE/dx[0], dE/dx[1], 
        ..., etc.

    x : `numpy.ndarray`
        Initial guess x_0.

    eta : `float`, keyword-only, default: 0.1
        Learning rate eta.

    tol : `float`, keyword-only, default: 1e-14
        Tolerance for E(x).

    max_iters : `int`, keyword-only, default: 100_000
        Maximum number of iterations.

    Returns
    -------
    x : `numpy.ndarray`
        Optimal x.

    iters : `int`
        Number of iterations required for the coordinate descent 
        algorithm to converge to x.
    """

    iters = 0
    while E(x) > tol and iters < max_iters:
        for i in range(len(x)):
            x[i] -= eta * dE_dx[i](x)
        iters += 1
    return x, iters

def stochastic_gradient_descent(
        N: int, f: Callable[[np.ndarray[float]], np.ndarray[float]], 
        eta: float = 0.01, tol: float = 0.01, *, N_test: int = 1_000,
        rng: np.random.Generator = None, seed: int = None, hyp: bool = False
    ) -> tuple[np.ndarray[float], float, float]:
    
    """
    Implements the stochastic gradient descent algorithm for a target 
    function operating on a d-dimensional data set D.

    Parameters
    ----------
    N : `int`
        Number of random data points.

    f : `function`
        Target function f as a function of the inputs x.

    eta : `float`, default: 0.01
        Learning rate eta.

    tol : `float`, default: 0.01
        Tolerance for the norm of the change in the hypothesis w.

    N_test : `int`, keyword-only, default: 1_000
        Number of random test data points.

    rng : `numpy.random.Generator`, keyword-only, optional
        A NumPy pseudo-random number generator.

    seed : `int`, keyword-only, optional
        Random seed used to initialize a pseudo-random number generator.
        Only used if `rng=None`.

    hyp : `bool`, keyword-only, default: False
        Determines whether the hypothesis w is returned.

    Returns
    -------
    w : `numpy.ndarray`
        Hypothesis w.

    epoch : `int`
        Number of epochs required for the stochastic gradient descent
        algorithm to converge to w.

    E_out : `float`
        Out-of-sample error E_out.
    """

    if rng is None:
        rng = np.random.default_rng(seed)
    xs, ys = generate_data(N, f, bias=True, rng=rng)
    w = np.zeros(xs.shape[1], dtype=float)
    epoch = 0
    while True:
        _w = w.copy()
        ri = rng.permutation(np.arange(N))
        for x, y in zip(xs[ri], ys[ri]):
            _w += eta * y * x / (1 + np.exp(y * x @ _w))
        dw = _w - w
        w = _w
        epoch += 1
        if np.linalg.norm(dw) < tol:
            break
    
    x_test, y_test = generate_data(N_test, f, bias=True, rng=rng)
    E_out = np.log(1 + np.exp(-y_test[:, None] * x_test @ w)).mean()
    return (w, epoch, E_out)[1 - hyp:]

### HOMEWORK 7 ################################################################

def support_vector_machine(
        N: int = None, 
        f: Callable[[np.ndarray[float]], np.ndarray[float]] = None,
        vf: Callable[[np.ndarray[float], np.ndarray[float], np.ndarray[float]],
                     np.ndarray[float]] = None,
        *, x: np.ndarray[float] = None, y: np.ndarray[float] = None,
        N_test: int = 1_000, x_test: np.ndarray[float] = None, 
        y_test: np.ndarray[float] = None, clf: svm.SVC = None,
        rng: np.random.Generator = None, seed: int = None, hyp: bool = False,
        **kwargs) -> tuple[int, float]: 

    """
    Provides a wrapper around `sklearn.svm.SVC` to implement the
    support vector machine (SVM) algorithm for a target function
    operating on a d-dimensional data set D.

    Parameters
    ----------
    N : `int`, optional
        Number of random data points.

    f : `function`, optional
        Target function f as a function of the inputs x.

    vf : `function`, optional
        Validation function as a function of w, x, and y.

    x : `numpy.ndarray`, keyword-only, optional
        Inputs x_n. If :code:`kernel="linear"`, the first
        column should be an array of ones for the constant bias term.

    y : `numpy.ndarray`, keyword-only, optional
        Outputs y_n.

    N_test : `int`, keyword-only, default: 1_000
        Number of random test data points.

    x_test : `numpy.ndarray`, keyword-only, optional
        Test inputs x_n.

    y_test : `numpy.ndarray`, keyword-only, optional
        Test outputs y_n.

    clf : `sklearn.svm.SVC`, keyword-only, optional
        Support vector machine classifier.

    rng : `numpy.random.Generator`, keyword-only, optional
        A NumPy pseudo-random number generator.

    seed : `int`, keyword-only, optional
        Random seed used to initialize a pseudo-random number generator.
        Only used if `rng=None`.

    hyp : `bool`, keyword-only, default: False
        Determines whether the hypothesis w is returned.

    Returns
    -------
    w : `numpy.ndarray`
        Hypothesis w. Only available if `hyp=True` and 
        :code:`kernel="linear"`.

    N_sv : `int`
        Number of support vectors.

    E_out : `float`
        Out-of-sample error E_out.
    """

    if clf is None:
        clf = svm.SVC(**kwargs)
    is_linear_kernel = clf.kernel == "linear"

    if rng is None:
        rng = np.random.default_rng(seed)
    if y is None:
        if x is None:
            x, y = generate_data(N, f, bias=is_linear_kernel, rng=rng)
        else:
            y = f(x)

    clf.fit(x[:, is_linear_kernel:], y)
    N_sv = clf.n_support_.sum()

    if x_test is None or y_test is None:
        if x_test is None:
            x_test, y_test = generate_data(N_test, f, bias=is_linear_kernel,
                                           rng=rng)
        else:
            y_test = f(x_test)

    if is_linear_kernel:
        w = np.concatenate((clf.intercept_, clf.coef_[0]))
        return (
            w, N_sv, 
            vf(w, x_test, y_test) if vf 
            else (1 - clf.score(x_test[:, is_linear_kernel:], y_test))
        )[1 - hyp:]
    return (N_sv, 1 - clf.score(x_test, y_test))