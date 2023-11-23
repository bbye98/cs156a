from typing import Callable, Union

import numpy as np
from scipy import optimize
from sklearn import svm
from sklearn.cluster import k_means

### Homework 1 ################################################################

class Perceptron:

    """
    Perceptron.

    Parameters
    ----------
    w : `numpy.ndarray`, optional
        Hypothesis w. If not provided, a vector with all zeros will be
        initialized during training.

    vf : `function`, optional
        Validation function as a function of w, x, and y.

    Attributes
    ----------
    iters : `int`
        Number of iterations required for the perceptron to converge.

    w : `numpy.ndarray`
        Hypothesis w.

    vf : `function`
        Validation function as a function of w, x, and y.
    """

    def __init__(
            self, w: np.ndarray[float] = None, *, 
            vf: Callable[[np.ndarray[float], np.ndarray[float],
                          np.ndarray[float]], 
                         np.ndarray[float]] = None
        ) -> None:
        
        self.set_parameters(w, vf=vf)

    def get_error(self, x: np.ndarray[float], y: np.ndarray[float]) -> float:
        
        """
        Get in-sample or out-of-sample error using a validation or test
        data set, respectively.

        Parameters
        ----------
        x : `numpy.ndarray`
            Inputs x.

        y : `numpy.ndarray`
            Outputs y.

        Returns
        -------
        E : `float`
            In-sample error E_in or out-of-sample error E_out.
        """
        
        if self.vf is not None and self.w is not None:
            return self.vf(self.w, x, y)

    def set_parameters(
            self, w: np.ndarray[float] = None, *,
            vf: Callable[[np.ndarray[float], np.ndarray[float],
                          np.ndarray[float]], 
                         np.ndarray[float]] = None,
            update: bool = False) -> None:

        """
        Set perceptron parameters.

        Parameters
        ----------
        w : `numpy.ndarray`, optional
            Hypothesis w. If not provided, a vector with all zeros will
            be initialized during training.

        vf : `function`, keyword-only, optional
            Validation function as a function of w, x, and y.

        update : `bool`, keyword-only, default: False
            If True, only arguments that are not None will be updated.
            If False, all parameters will be overwritten.
        """

        if update:
            self.vf = vf or self.vf
            self._w = self._w if w is None else w
        else:
            self.vf = vf
            self._w = w

    def train(self, x: np.ndarray[float], y: np.ndarray[float]) -> float:

        """
        Train the perceptron using a training data set.

        Parameters
        ----------
        x : `numpy.ndarray`
            Inputs x.

        y : `numpy.ndarray`
            Outputs y.

        Returns
        -------
        iters : `int`
            Number of iterations required for the perceptron to 
            converge.
        """
        
        self.iters = 0
        self.w = (np.zeros(x.shape[1], dtype=float) if self._w is None 
                  else self._w)
        while True:
            wrong = np.argwhere(np.sign(x @ self.w) != y)[:, 0]
            if wrong.size == 0:
                break
            index = np.random.choice(wrong)
            self.w += y[index] * x[index]
            self.iters += 1

        if self.vf:
            return self.vf(self.w, x, y)

def target_function_random_line(
        x: np.ndarray[float] = None, *, rng: np.random.Generator = None, 
        seed: int = None
    ) -> Union[Callable[[np.ndarray[float]], np.ndarray[float]],
               np.ndarray[float]]:

    """
    Implements the target function f(x_1, x_2) = m * x_1 + b, where m
    and b are the m and y-intersect, respectively, of a random line, 
    that classifies the outputs of a two-dimensional data set D as 
    either -1 or 1, depending on which side of the line they fall on.

    Parameters
    ----------
    x : `numpy.ndarray`, optional
        Inputs x, which contain the x_1- and x_2-coordinates in the last
        two columns.

    rng : `numpy.random.Generator`, keyword-only, optional
        A NumPy pseudo-random number generator.

    seed : `int`, keyword-only, optional
        Random seed used to initialize the pseudo-random number
        generator. Only used if `rng=None`.

    Returns
    -------
    f : `function` or `numpy.ndarray`
        If `x=None`, a target function f as a function of the inputs x, 
        which contain the x_1- and x_2-coordinates in the last two 
        columns, is returned. If x is provided, the outputs y is 
        returned.
    """
    
    if rng is None:
        rng = np.random.default_rng(seed)

    line = rng.uniform(-1, 1, (2, 2))
    f = lambda x: np.sign(
        x[:, -1] - line[0, 1] 
        - np.divide(*(line[1] - line[0])[::-1]) * (x[:, -2] - line[0, 0])
    )
    return f if x is None else f(x)

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
        Random seed used to initialize the pseudo-random number 
        generator. Only used if `rng=None`.

    Returns
    -------
    inputs : `numpy.ndarray`
        Inputs x.

    outputs : `numpy.ndarray`
        Outputs y.
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
        Inputs x.

    y : `numpy.ndarray`
        Outputs y.

    Returns
    -------
    E : `float`
        In-sample error E_in or out-of-sample error E_out.
    """
    
    return np.count_nonzero(np.sign(x @ w) != y, axis=0) / x.shape[0]

### Homework 2 ################################################################

class LinearRegression:

    """
    Linear regression model.

    Parameters
    ----------
    vf : `function`, optional
        Validation function as a function of w, x, and y.

    regularization : `str`, optional
        Regularization function for the hypothesis w.
        
        **Valid values**: 
        
        * :code:`"weight_decay"`: Weight decay regularization. Specify
          lambda in keyword argument `weight_decay_lambda`.

    transform : `function`, optional
        Nonlinear transformation function for the inputs x.

    noise : `tuple`, optional
        Fraction of outputs y to introduce noise to and the
        noise function.

    rng : `numpy.random.Generator`, optional
        A NumPy pseudo-random number generator.

    seed : `int`, optional
        Random seed used to initialize a pseudo-random number generator.
        Only used if `rng=None`.

    Attributes
    ----------
    w : `numpy.ndarray`
        Hypothesis w.

    vf : `function`
        Validation function as a function of w, x, and y.

    regularization : `str`
        Regularization function for the hypothesis w.

    transform : `function`
        Nonlinear transformation function for the inputs x.

    noise : `tuple`
        Fraction of outputs y to introduce noise to and the
        noise function.

    rng : `numpy.random.Generator`
        A NumPy pseudo-random number generator.
    """

    def __init__(
            self, *,
            vf: Callable[[np.ndarray[float], np.ndarray[float], 
                          np.ndarray[float]], 
                         np.ndarray[float]] = None,
            regularization: str = None,
            transform: Callable[[np.ndarray[float]], np.ndarray[float]] = None,
            noise: tuple[float, Callable[[np.ndarray[float]], 
                                         np.ndarray[float]]] = None,
            rng: np.random.Generator = None, seed: int = None, **kwargs
        ) -> None:

        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.set_parameters(vf=vf, regularization=regularization, 
                            transform=transform, noise=noise, **kwargs)

    def get_error(self, x: np.ndarray[float], y: np.ndarray[float]) -> float:

        """
        Get in-sample or out-of-sample error using a validation or test
        data set, respectively.

        Parameters
        ----------
        x : `numpy.ndarray`
            Inputs x.

        y : `numpy.ndarray`
            Outputs y.

        Returns
        -------
        E : `float`
            In-sample error E_in or out-of-sample error E_out.
        """

        if self.transform:
            x = self.transform(x)
        if self.noise:
            N = x.shape[0]
            index = self.rng.choice(N, round(self.noise[0] * N), False)
            y[index] = self.noise[1](y[index])

        if self.vf is not None and self.w is not None:
            return self.vf(self.w, x, y)

    def set_parameters(
            self, *,
            vf: Callable[[np.ndarray[float], np.ndarray[float], 
                          np.ndarray[float]], 
                         np.ndarray[float]] = None,
            regularization: str = None, 
            transform: Callable[[np.ndarray[float]], np.ndarray[float]] = None, 
            noise: tuple[float, Callable[[np.ndarray[float]], 
                                         np.ndarray[float]]] = None,
            update: bool = False, **kwargs) -> None:
        
        """
        Set linear regression model parameters.

        Parameters
        ----------
        vf : `function`, keyword-only, optional
            Validation function as a function of w, x, and y.

        regularization : `str`, keyword-only, optional
            Regularization function for the hypothesis w.
            
            **Valid values**: 
            
            * :code:`"weight_decay"`: Weight decay regularization. Specify
              lambda in keyword argument `weight_decay_lambda`.

        transform : `function`, keyword-only, optional
            Nonlinear transformation function for the inputs x.

        noise : `tuple`, keyword-only, optional
            Fraction of outputs y to introduce noise to and the
            noise function.

        update : `bool`, keyword-only, default: False
            If True, only arguments that are not None will be updated.
            If False, all parameters will be overwritten.
        """

        self._reg_params = {}       
        self.w = None

        if update:
            self.noise = noise or self.noise
            self.regularization = regularization or self.regularization
            if self.regularization == "weight_decay" \
                    and "weight_decay_lambda" in kwargs:
                self._reg_params["lambda"] = kwargs["weight_decay_lambda"]
            self.transform = transform or self.transform
            self.vf = vf or self.vf
        else:
            self.noise = noise
            self.regularization = regularization
            if regularization == "weight_decay":
                self._reg_params["lambda"] = kwargs["weight_decay_lambda"]
            self.transform = transform
            self.vf = vf

    def train(self, x: np.ndarray[float], y: np.ndarray[float]) -> float:

        """
        Train the linear regression model using a training data set.

        Parameters
        ----------
        x : `numpy.ndarray`
            Inputs x.

        y : `numpy.ndarray`
            Outputs y.

        Returns
        -------
        E : `float`
            In-sample error E_in.
        """

        if self.transform:
            x = self.transform(x)
        if self.noise:
            N = x.shape[0]
            index = self.rng.choice(N, round(self.noise[0] * N), False)
            y[index] = self.noise[1](y[index])
        
        if self.regularization is None:
            self.w = np.linalg.pinv(x) @ y
        elif self.regularization == "weight_decay":
            self.w = np.linalg.inv(
                x.T @ x 
                + self._reg_params["lambda"] * np.eye(x.shape[1], dtype=float)
            ) @ x.T @ y
        
        if self.vf is not None:
            return self.vf(self.w, x, y)

def coin_flip(
        N_trials: int = 1, N_coins: int = 1_000, N_flips: int = 10, *,
        rng: np.random.Generator = None, seed: int = None
    ) -> np.ndarray[float]:
    
    """
    Simulates coin flipping.

    Parameters
    ----------
    N_trials : `int`, default: 1
        Number of trials.

    N_coins : `int`, default: 1_000
        Number of coins to flip.

    N_flips : `int`, default: 10
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
        rng.uniform(size=(N_trials, N_coins, N_flips)) < 0.5, 
        axis=2
    ) # [0.0, 0.5) is heads, [0.5, 1.0) is tails
    indices = np.arange(N_trials)
    return np.stack((
        heads[:, 0],
        heads[indices, rng.integers(N_coins, size=N_trials)],
        heads[indices, np.argmin(heads, axis=1)],
    )) / N_flips

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

def target_function_homework_2(
        x: np.ndarray[float] = None
    ) -> Union[Callable[[np.ndarray[float]], np.ndarray[float]],
               np.ndarray[float]]:

    """
    Implements the target function 
    f(x_1, x_2) = sgn(x_1^2 + x_2^2 - 0.6).

    Parameters
    ----------
    x : `numpy.ndarray`, optional
        Inputs x, which contain the x_1- and x_2-coordinates in the 
        last two columns.

    Returns
    -------
    f : `function` or `numpy.ndarray`
        If `x=None`, a target function f as a function of the inputs x, 
        which contain the x_1- and x_2-coordinates in the last two 
        columns, is returned. If x is provided, the outputs y is 
        returned.
    """
    
    f = lambda x: np.sign((x[:, -2:] ** 2).sum(axis=1) - 0.6)
    return f if x is None else f(x)

### Homework 4 ################################################################

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
        N: np.ndarray[np.longdouble], delta: float, *, ub: float = 10.0
    ) -> np.ndarray[np.longdouble]:

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
        Sample size(s). Must be provided as 128-bit floating-point 
        numbers to prevent integer overflow.

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

    return np.vectorize(
        lambda N: optimize.root_scalar(
            lambda eps, N: np.sqrt(
                (4 * eps * (1 + eps) + np.log(4 / delta) + np.log(m_H(N ** 2)))
                / (2 * N)
            ) - eps,
            args=N, 
            bracket=(0.0, ub), 
            method="toms748"
        ).root
    )(N)

### Homework 5 ################################################################

def gradient_descent( #
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

def coordinate_descent( #
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

def stochastic_gradient_descent( #
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

### Homework 7 ################################################################

def support_vector_machine( #
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

    x_validate : `numpy.ndarray`, keyword-only, optional
        Validation inputs x_n. If not specified, no validation is
        performed.

    y_validate : `numpy.ndarray`, keyword-only, optional
        Validation outputs y_n. If not specified, no validation is
        performed.

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

### Final Exam ################################################################

class RBFRegular:

    """
    Radial basis function (RBF) network in regular form (Lloyd + 
    pseudo-inverse) with K centers.

    Parameters
    ----------
    gamma : `float`
        Kernel coefficient gamma.

    K : `int`
        Number of clusters to form or number of centroids to generate.

    vf : `function`, keyword-only, optional
        Validation function as a function of w, phi, and y.

    Attributes
    ----------
    centers : `numpy.ndarray`
        Centers.

    gamma : `float`
        Kernel coefficient gamma.

    K : `int`
        Number of clusters to form or number of centroids to generate.

    vf : `function`
        Validation function as a function of w, phi, and y.

    w : `numpy.ndarray`
        Hypothesis w.
    """

    def __init__(
            self, gamma: float, K: int, *,
            vf: Callable[[np.ndarray[float], np.ndarray[float], 
                          np.ndarray[float]],
                         np.ndarray[float]] = None) -> None:
        
        self.set_parameters(gamma, K, vf=vf)

    def get_error(self, x: np.ndarray[float], y: np.ndarray[float]) -> float:
        
        """
        Get in-sample or out-of-sample error using a validation or test
        data set, respectively.

        Parameters
        ----------
        x : `numpy.ndarray`
            Inputs x.

        y : `numpy.ndarray`
            Outputs y.

        Returns
        -------
        E : `float`
            In-sample error E_in or out-of-sample error E_out.
        """
        
        if self.vf is not None and self.w is not None:
            return self.vf(self.w, self.get_phi(x, self.centers), y)

    def get_phi(
            self, x: np.ndarray[float], centers: np.ndarray[float]
        ) -> np.ndarray[float]:

        """
        Get the input vector phi for a given input x.

        Parameters
        ----------
        x : `numpy.ndarray`
            Inputs x.

        centers : `numpy.ndarray`
            Centers.

        Returns
        -------
        phi : `numpy.ndarray`
            Input vector phi.
        """
        
        return np.hstack((
            np.ones((x.shape[0], 1), dtype=float), 
            np.exp(-self.gamma 
                   * np.linalg.norm((x[:, None] - centers), axis=2) ** 2)
        ))

    def set_parameters(
            self, gamma: float, K: int, *, 
            vf: Callable[[np.ndarray[float], np.ndarray[float],
                          np.ndarray[float]],
                         np.ndarray[float]] = None,
            update: bool = False) -> None:

        """
        Set RBF network parameters and reset centers and weights from
        previous training.

        Parameters
        ----------
        gamma : `float`
            Kernel coefficient gamma.

        K : `int`
            Number of clusters to form or number of centroids to
            generate.

        vf : `function`, keyword-only, optional
            Validation function as a function of w, phi, and y.

        update : `bool`, keyword-only, default: False
            If True, only arguments that are not None will be updated.
            If False, all parameters will be overwritten.
        """
        
        self.centers = None
        self.w = None
        if update:
            self.gamma = gamma or self.gamma
            self.K = K or self.K
            self.vf = vf or self.vf
        else:
            self.gamma = gamma
            self.K = K
            self.vf = vf

    def train(self, x: np.ndarray[float], y: np.ndarray[float]) -> float:

        """
        Train the RBF network using a training data set.

        Parameters
        ----------
        x : `numpy.ndarray`
            Training inputs x.

        y : `numpy.ndarray`
            Training outputs y.

        Returns
        -------
        E_in : `float`
            In-sample error E_in.
        """
        
        self.centers = k_means(x, self.K, n_init="auto")[0]
        phi = self.get_phi(x, self.centers)
        self.w = np.linalg.pinv(phi) @ y

        if self.vf is not None:
            return self.vf(self.w, phi, y)

def target_function_final_exam(
        x: np.ndarray[float]
    ) -> Union[Callable[[np.ndarray[float]], np.ndarray[float]],
               np.ndarray[float]]:

    """
    Implements the target function 
    f(x_1, x_2) = sgn(x_2 - x_1 + 0.25 * sin(pi * x_1)).

    Parameters
    ----------
    x : `numpy.ndarray`
        Inputs x, which contain the x_1- and x_2-coordinates in the last
        two columns.

    Returns
    -------
    f : `function` or `numpy.ndarray`
        If `x=None`, a target function f as a function of the inputs x, 
        which contain the x_1- and x_2-coordinates in the last two 
        columns, is returned. If x is provided, the outputs y is 
        returned.
    """

    f = lambda x: np.sign(np.diff(x[:, -2:], axis=1)[:, 0]
                          + 0.25 * np.sin(np.pi * x[:, -1]))
    return f if x is None else f(x)