#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
October 2, 2023
Homework 1
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cs156a import (Perceptron, target_function_random_line, generate_data,
                    validate_binary)
    
if __name__ == "__main__":

    rng = np.random.default_rng()

    ### Problems 7–10 #########################################################

    N_runs = 1_000
    pla = Perceptron(vf=validate_binary)
    columns = ["number of points", "number of iterations", 
               "misclassification rate"]
    df = pd.DataFrame(columns=columns)
    for N_train in (10, 100):
        N_test = 9 * N_train
        counters = np.zeros(2, dtype=float)
        for _ in range(N_runs):
            f = target_function_random_line(rng=rng)
            pla.train(*generate_data(N_train, f, bias=True, rng=rng))
            counters += (
                pla.iters, 
                pla.get_error(*generate_data(N_test, f, bias=True, rng=rng))
            )
        df.loc[len(df)] = N_train, *(counters / N_runs)
    print("\n[Homework 1 Problems 7–10]\n"
          f"Perceptron learning algorithm ({N_runs:,} runs):\n",
          df.to_string(index=False, 
                       formatters={c: "{:.0f}".format for c in columns[:2]}),
          sep="")