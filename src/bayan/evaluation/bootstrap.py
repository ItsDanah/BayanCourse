"""Lab 6: bootstrap confidence intervals."""

import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        raise ValueError("values must not be empty")

    rng = np.random.default_rng(seed)

    point = float(values.mean())

    samples = rng.choice(
        values,
        size=(n_boot, len(values)),
        replace=True,
    )

    boot_means = samples.mean(axis=1)

    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))

    return point, lo, hi


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b):
        raise ValueError("a and b must have the same length")

    if len(a) == 0:
        raise ValueError("inputs must not be empty")

    differences = a - b

    rng = np.random.default_rng(seed)

    delta = float(differences.mean())

    samples = rng.choice(
        differences,
        size=(n_boot, len(differences)),
        replace=True,
    )

    boot_deltas = samples.mean(axis=1)

    lo = float(np.quantile(boot_deltas, alpha / 2))
    hi = float(np.quantile(boot_deltas, 1 - alpha / 2))

    return delta, lo, hi