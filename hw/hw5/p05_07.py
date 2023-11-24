import numpy as np
import pandas as pd

def gradient_descent(E, dE, x, *, eta=0.1, tol=1e-14, max_iters=100_000):
    iters = 0
    while E(x) > tol and iters < max_iters:
        x -= eta * dE(x)
        iters += 1
    return x, iters

def coordinate_descent(E, dE_dx, x, *, eta=0.1, tol=1e-14, max_iters=100_000):
    iters = 0
    while E(x) > tol and iters < max_iters:
        for i in range(len(x)):
            x[i] -= eta * dE_dx[i](x)
        iters += 1
    return x, iters

if __name__ == "__main__":
    rng = np.random.default_rng()

    eta = 0.1
    E = lambda x: (x[0] * np.exp(x[1]) - 2 * x[1] * np.exp(-x[0])) ** 2
    dE_du = lambda x: (2 * (x[0] * np.exp(x[1]) - 2 * x[1] * np.exp(-x[0])) 
                       * (np.exp(x[1]) + 2 * x[1] * np.exp(-x[0])))
    dE_dv = lambda x: (2 * (x[0] * np.exp(x[1]) - 2 * x[1] * np.exp(-x[0]))
                       * (x[0] * np.exp(x[1]) - 2 * np.exp(-x[0])))
    df = pd.DataFrame(columns=["method", "iterations", "x", "E(x)"])
    x, iters = gradient_descent(E, lambda x: np.array((dE_du(x), dE_dv(x))), 
                                np.array((1, 1), dtype=float), eta=eta)
    df.loc[len(df)] = "gradient descent", iters, np.round(x, 6), E(x)
    x, iters = coordinate_descent(E, (dE_du, dE_dv), 
                                  np.array((1, 1), dtype=float), eta=eta,
                                  max_iters=15)
    df.loc[len(df)] = "coordinate descent", iters, np.round(x, 6), E(x)
    print(f"\n[Homework 5 Problems 5–7]\nDescent methods ({eta=}):\n",
          df.to_string(index=False), sep="")