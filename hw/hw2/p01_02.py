import numpy as np
import pandas as pd

def coin_flip(N_trials=1, N_coins=1_000, N_flips=10, *,rng=None, seed=None):
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

def hoeffding_inequality(N, eps, *, M=1):
    return 2 * M * np.exp(-2 * eps ** 2 * N)

if __name__ == "__main__":
    rng = np.random.default_rng()

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