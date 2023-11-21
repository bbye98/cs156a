#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
October 30, 2023
Homework 5
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from cs156a import (
    gradient_descent, coordinate_descent,
    target_function_random_line, stochastic_gradient_descent
)

if __name__ == "__main__":
    # Problems 5–7
    E = lambda x: (x[0] * np.exp(x[1]) - 2 * x[1] * np.exp(-x[0])) ** 2
    dE_du = lambda x: (2 * (x[0] * np.exp(x[1]) - 2 * x[1] * np.exp(-x[0])) 
                    * (np.exp(x[1]) + 2 * x[1] * np.exp(-x[0])))
    dE_dv = lambda x: (2 * (x[0] * np.exp(x[1]) - 2 * x[1] * np.exp(-x[0]))
                    * (x[0] * np.exp(x[1]) - 2 * np.exp(-x[0])))
    print(f"\n[HW5 P5–7]\nPerformance of descent methods for eta=0.1:")
    x, iters = gradient_descent(E, lambda x: np.array((dE_du(x), dE_dv(x))), 
                                np.array((1, 1), dtype=float))
    print(f"  Gradient descent: {iters=}, x=({x[0]:.3f}, {x[1]:.3f})")
    x, iters = coordinate_descent(E, (dE_du, dE_dv), 
                                  np.array((1, 1), dtype=float), max_iters=15)
    print(f"  Coordinate descent: {iters=}, x=({x[0]:.3f}, {x[1]:.3f}), "
          f"{E(x)=:.3e}")
    
    # Problems 8–9
    rng = np.random.default_rng()
    N = 100
    n_runs = 100
    print("\n[HW5 P8–9]\nStochastic gradient descent statistics over "
          f"{n_runs:,} runs:")
    epochs, E_out = np.mean(
        [stochastic_gradient_descent(N, target_function_random_line(rng=rng), 
                                     rng=rng) for _ in range(n_runs)],
        axis=0
    )
    print(f"  {N=}, {epochs=:.0f}, {E_out=:.3f}")