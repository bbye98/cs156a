import numpy as np
from sklearn.utils import shuffle

class StochasticGradientDescent:
    def __init__(self, eta=0.01, tol=0.01, *, rng=None, seed=None):
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.vf = lambda w, x, y: np.log(1 + np.exp(-y[:, None] * x @ w)).mean()
        self.set_parameters(eta, tol)

    def get_error(self, x, y):
        if self.w is not None:
            return self.vf(self.w, x, y)

    def set_parameters(self, eta=None, tol=None, update=False):
        if update:
            self.eta = eta or self.eta
            self.tol = tol or self.tol
        else:
            self.eta = eta
            self.tol = tol

    def train(self, x, y) -> float:
        self.w = np.zeros(x.shape[1], dtype=float)
        self.epochs = 0
        while True:
            w = self.w.copy()
            for x_, y_ in zip(*shuffle(x, y)):
                w += self.eta * y_ * x_ / (1 + np.exp(y_ * x_ @ w))
            dw = w - self.w
            self.w = w.copy()
            self.epochs += 1
            if np.linalg.norm(dw) < self.tol:
                break
        return self.vf(self.w, x, y)

def target_function_random_line(x=None, *, rng=None, seed=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    line = rng.uniform(-1, 1, (2, 2))
    f = lambda x: np.sign(
        x[:, -1] - line[0, 1] 
        - np.divide(*(line[1] - line[0])[::-1]) * (x[:, -2] - line[0, 0])
    )
    return f if x is None else f(x)

def generate_data(
        N, f, d=2, lb=-1.0, ub=1.0, *, bias=False, rng=None, seed=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    x = rng.uniform(lb, ub, (N, d))
    if bias:
        x = np.hstack((np.ones((N, 1)), x))
    return x, f(x)

if __name__ == "__main__":
    rng = np.random.default_rng()

    eta = 0.01
    N_runs = N_train = 100
    N_test = 9 * N_train
    sgd = StochasticGradientDescent(eta, rng=rng)
    counters = np.zeros(2, dtype=float)
    for _ in range(N_runs):
        f = target_function_random_line(rng=rng)
        sgd.train(*generate_data(N_train, f, bias=True, rng=rng))
        counters += (
            sgd.epochs, 
            sgd.get_error(*generate_data(N_test, f, bias=True, rng=rng))
        )
    counters /= N_runs
    print("\n[Homework 5 Problems 8–9]\n"
          f"Using stochastic gradient descent with {eta=}, the average "
          f"number of epochs and out-of-sample error over {N_runs} runs "
          f"are {counters[0]:.0f} and {counters[1]:.6f}, respectively.")