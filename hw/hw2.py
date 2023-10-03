#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
October 9, 2023
Homework 2
"""

import copy
import pathlib
import sys

import numpy as np

sys.path.insert(0, pathlib.Path(__file__).resolve().parent)
from cs156a import (
    coin_flip, hoeffding_inequality, 
    target_function_random_line, perceptron, linear_regression,
    target_function_hw2, generate_data, validate_binary
)

if __name__ == "__main__":
    # problems 1–2
    n_trials = 100_000
    n_flips = 10
    labels = ("first", "random", "minimum")
    print(f"\n[HW2 P1–2]\nCoin flip statistics over {n_trials:,} trials:")
    nus = coin_flip(n_trials)
    for label, nu in zip(labels, nus.mean(axis=1)):
        print(f"  {label} coin: {nu=:.5f}")

    print("\nHoeffding's inequality:")
    epss = np.linspace(0, 0.5, 6)
    hist = np.apply_along_axis(
        lambda x: np.histogram(x, bins=np.linspace(-0.05, 1.05, 12))[0], 1, nus
    ) # requires at least 8 GB RAM
    probs = np.hstack((hist[:, (5,)], hist[:, 4::-1] + hist[:, 6:])) / n_trials
    bounds = hoeffding_inequality(n_flips, epss)
    print("   eps |  bound  | ", " | ".join(l.center(15) for l in labels),
          "\n  -----+---------+", "+".join(3 * [17 * "-"]), sep="")
    for eps, bound, prob, satisfy in zip(
            epss, bounds, probs.T, (probs <= bounds).T):
        print(
            f"   {eps:.1f} | {bound:.5f} |",
            " | ".join(
                f"{p:.5f} ({s})".ljust(15) for p, s in zip(prob, satisfy)
            )
        )

    # problems 5–7
    rng = np.random.default_rng()
    N = 100
    n_runs = 1_000
    print(f"\n[HW2 P5–7]\nLinear regression statistics over {n_runs:,} runs:")
    E_in, E_out = np.mean(
        [linear_regression(N, target_function_random_line(rng=rng), rng=rng)
         for _ in range(n_runs)], 
        axis=0
    )
    print(f"  {N=:,}, {E_in=:.3f}, {E_out=:.3f}")

    N = 10
    print(
        "\nPLA (with linear regression hypothesis) statistics over",
        f"{n_runs:,} runs:"
    )
    iters = np.empty(n_runs, dtype=float)
    for i in range(n_runs):
        f = target_function_random_line(rng=rng)
        iters[i] = perceptron(
            N, f,
            w=linear_regression(N, f, rng=copy.copy(rng), hyp=True)[0], 
            rng=copy.copy(rng) # ensures same RNG state for PLA and LRA
        )[0]
    print(f"  {N=:,}, iters={iters.mean():,.0f}")

    # problems 8–10
    f = target_function_hw2()
    N = N_test = n_runs = 1_000
    noise = 0.1
    print("\n[HW2 P8–10]\nLinear regression (with linear feature vector)",
          f"statistics over {n_runs:,} runs:")
    E_in = np.mean(
        [linear_regression(N, f, noise=noise)[0] for _ in range(n_runs)]
    )
    print(f"  {N=:,}, {noise=:.3f}, {E_in=:.3f}")

    transform = lambda x: np.hstack((x, x[:, 1:2] * x[:, 2:], x[:, 1:2] ** 2,
                                     x[:, 2:] ** 2))
    gs = np.array(((-1, -0.05, 0.08, 0.13, 1.5, 1.5), 
                   (-1, -0.05, 0.08, 0.13, 1.5, 15),
                   (-1, -0.05, 0.08, 0.13, 15, 1.5),
                   (-1, -1.5, 0.08, 0.13, 0.05, 0.05),
                   (-1, -0.05, 0.08, 1.5, 0.15, 0.15)))
    print("\nLinear regression (with nonlinear feature vector) hypothesis",
          f"over {n_runs:,} runs:")
    w = np.mean(
        [linear_regression(N, f, transform=transform, rng=rng, noise=noise,
                           hyp=True)[0]
         for _ in range(n_runs)],
        axis=0
    )
    print("  w=[", ", ".join(f"{v:.5f}" for v in w), "]", sep="")
    probs = np.zeros((N_test, 5))
    Es_out = np.zeros(N_test)
    for i in range(n_runs):
        x_test, y_test = generate_data(N_test, f, rng=rng)
        x_test = transform(x_test)
        y_test[rng.choice(N_test, round(noise * N_test), False)] *= -1
        h_test = np.sign(x_test @ w)
        probs[i] = validate_binary(gs.T, x_test, h_test[:, None])
        Es_out[i] = np.count_nonzero(h_test != y_test) / N_test
    for i, (g, p) in enumerate(zip(gs, probs.mean(axis=0))):
        print(f"  g{i + 1}=[", ", ".join(f"{v:.2g}" for v in g),
              f"] (prob={p:.5f})", sep="")
    print(f"  {N=:,}, {noise=:.3f}, E_out={Es_out.mean():.3f}")