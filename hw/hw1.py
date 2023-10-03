#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
October 2, 2023
Homework 1
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, pathlib.Path(__file__).resolve().parent)
from cs156a import target_function_random_line, perceptron
    
if __name__ == "__main__":
    # problems 7–10
    rng = np.random.default_rng()
    n_runs = 1_000
    print(f"\n[HW1 P7–10]\nPLA statistics over {n_runs:,} runs:")
    for N in (10, 100):
        iters, prob = np.mean(
            [perceptron(N, target_function_random_line(rng=rng), rng=rng)
             for _ in range(n_runs)], 
            axis=0
        )
        print(f"  {N=:,}, {iters=:,.0f}, {prob=:.3f}")