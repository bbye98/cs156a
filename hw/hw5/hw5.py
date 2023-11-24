#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
October 30, 2023
Homework 5
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cs156a import (StochasticGradientDescent, gradient_descent, 
                    coordinate_descent, target_function_random_line, 
                    generate_data)

if __name__ == "__main__":
    rng = np.random.default_rng()
    
    ### Problems 5–7 ##########################################################

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
    
    ### Problems 8–9 ##########################################################

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