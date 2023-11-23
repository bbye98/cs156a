#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
October 9, 2023
Homework 2
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from cs156a import (Perceptron, LinearRegression,
                    coin_flip, hoeffding_inequality, 
                    target_function_random_line, target_function_homework_2,
                    generate_data, validate_binary)

if __name__ == "__main__":

    rng = np.random.default_rng()

    ### Problems 1–2 ##########################################################

    N_trials = 100_000
    N_coins = 1_000
    N_flips = 10
    nus = coin_flip(N_trials, N_coins, N_flips, rng=rng)
    coins = ("first coin", "random coin", "min. frequency of heads")
    df = pd.DataFrame({"coin": coins, "fraction of heads": nus.mean(axis=1)})
    print(f"\n[Homework 2 Problem 1]\n{df.to_string(index=False)}")

    epsilons = np.linspace(0, 0.5, 6)
    histograms = np.apply_along_axis(
        lambda x: np.histogram(x, bins=np.linspace(-0.05, 1.05, 12))[0], 1, nus
    ) # requires at least 8 GB RAM
    probabilities = np.hstack((
        histograms[:, (5,)], 
        histograms[:, 4::-1] + histograms[:, 6:]
    )) / N_trials
    bounds = hoeffding_inequality(N_flips, epsilons)
    satisfies = probabilities < bounds
    data = {"epsilon": epsilons, "bound": bounds}
    for i in range(nus.shape[0]):
        data[coins[i]] = probabilities[i]
        data[i * " "] = satisfies[i]
    print("\n[Homework 2 Problem 2]\n"
          f"{pd.DataFrame(data).to_string(index=False)}")

    ### Problems 5–7 ##########################################################

    N_train = 100
    N_test = 9 * N_train
    N_runs = 1_000
    f = target_function_random_line(rng=rng)
    reg = LinearRegression(vf=validate_binary, rng=rng)
    errors = np.zeros(2, dtype=float)
    for _ in range(N_runs):
        x_train, y_train = generate_data(N_train, f, bias=True, rng=rng)
        x_test, y_test = generate_data(N_test, f, bias=True, rng=rng)
        E_in = reg.train(x_train, y_train)
        errors += (E_in, reg.get_error(x_test, y_test))
    errors /= N_runs
    print("\n[Homework 2 Problems 5–6]\n"
          "For the linear regression model, the average in-sample and "
          f"out-of-sample errors over {N_runs:,} runs are "
          f"{errors[0]:.6f} and {errors[1]:.6f}, respectively.")

    N_train = 10
    pla = Perceptron(vf=validate_binary)
    iters = 0
    for _ in range(N_runs):
        f = target_function_random_line(rng=rng)
        x_train, y_train = generate_data(N_train, f, bias=True, rng=rng)
        reg.train(x_train, y_train)
        pla.set_parameters(w=reg.w, update=True)
        pla.train(x_train, y_train)
        iters += pla.iters
    print("\n[Homework 2 Problem 7]\n"
          "With initial weights from linear regression, the perceptron "
          f"takes an average of {iters / N_runs:.0f} iterations to "
          "converge.")

    ### Problems 8–10 #########################################################

    N_train = N_runs = 1_000
    N_test = 9 * N_train
    noise = (0.1, lambda y: -y)
    reg = LinearRegression(vf=validate_binary, noise=noise, rng=rng)
    E_in = 0
    for _ in range(N_runs):
        x_train, y_train = generate_data(N_train, target_function_homework_2,
                                         bias=True, rng=rng)
        x_test, y_test = generate_data(N_test, target_function_homework_2,
                                       bias=True, rng=rng)
        E_in += reg.train(x_train, y_train)
    print("\n[Homework 2 Problem 8]\n"
          f"For the linear regression model with {noise[0]:.0%} noise, "
          f"the average in-sample error over {N_runs:,} runs is "
          f"{E_in / N_runs:.6f}.")

    transform = lambda x: np.hstack((x, x[:, 1:2] * x[:, 2:], x[:, 1:2] ** 2,
                                     x[:, 2:] ** 2))
    gs = np.array(((-1, -0.05, 0.08, 0.13, 1.5, 1.5), 
                   (-1, -0.05, 0.08, 0.13, 1.5, 15),
                   (-1, -0.05, 0.08, 0.13, 15, 1.5),
                   (-1, -1.5, 0.08, 0.13, 0.05, 0.05),
                   (-1, -0.05, 0.08, 1.5, 0.15, 0.15)))
    w = np.zeros_like(gs[0])
    reg.set_parameters(vf=validate_binary, transform=transform, noise=noise, 
                       update=True)
    for _ in range(N_runs):
        x_train, y_train = generate_data(N_train, target_function_homework_2,
                                         bias=True, rng=rng)
        reg.train(x_train, y_train)
        w += reg.w
    w /= N_runs
    counters = np.zeros(6, dtype=float)
    for _ in range(N_runs):
        x_test, y_test = generate_data(N_test, target_function_homework_2,
                                    bias=True, rng=rng)
        x_test = transform(x_test)
        y_test[rng.choice(N_test, round(noise[0] * N_test), False)] *= -1
        h_test = np.sign(x_test @ w)
        counters += (*validate_binary(gs.T, x_test, h_test[:, None]),
                    np.count_nonzero(h_test != y_test) / N_test)
    counters /= N_runs
    df = pd.DataFrame({
        "choice": [f"[{chr(97 + i)}]" for i in range(5)],
        "g": [f"[{', '.join(f'{c:.2g}' for c in g)}]" for g in gs],
        "probability": 1 - counters[:5]
    })
    print("\n[Homework 2 Problem 9]\n"
          f"The average weight vector over {N_runs:,} runs is "
          "w = [", ", ".join(f"{v:.6f}" for v in w), "].\n", 
          df.to_string(index=False), sep="")

    print("\n[Homework 2 Problem 10]\n"
          f"The average out-of-sample error over {N_runs:,} runs is "
          f"{counters[5]:.6f}.")