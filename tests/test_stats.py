import random

import pytest

from ragbench.stats import bootstrap_ci, paired_bootstrap, percentile


def test_bootstrap_is_seeded_and_brackets_the_mean():
    rng = random.Random(0)
    values = [rng.random() for _ in range(300)]
    a = bootstrap_ci(values, rounds=500, seed=3)
    b = bootstrap_ci(values, rounds=500, seed=3)
    assert a == b
    assert a.low < a.mean < a.high
    assert a.mean == pytest.approx(sum(values) / len(values))


def test_constant_values_give_zero_width():
    ci = bootstrap_ci([0.5] * 50, rounds=200)
    assert ci.low == ci.high == ci.mean == 0.5


def test_bootstrap_covers_true_mean_most_of_the_time():
    # 95% intervals from samples of a known distribution should cover its mean ~95% of
    # the time; allow a wide band so the test is not flaky.
    rng = random.Random(1)
    covered = 0
    trials = 200
    for t in range(trials):
        sample = [rng.gauss(0.3, 0.2) for _ in range(80)]
        ci = bootstrap_ci(sample, rounds=400, seed=t)
        covered += ci.low <= 0.3 <= ci.high
    assert 0.88 <= covered / trials <= 0.99


def test_paired_bootstrap_detects_a_real_difference():
    rng = random.Random(2)
    a = [rng.random() for _ in range(200)]
    b = [x + 0.05 + rng.gauss(0, 0.02) for x in a]
    diff = paired_bootstrap(a, b, rounds=1000)
    assert diff.low > 0
    assert diff.mean == pytest.approx(0.05, abs=0.01)


def test_paired_bootstrap_identical_runs():
    a = [0.1, 0.4, 0.9]
    d = paired_bootstrap(a, list(a))
    assert (d.mean, d.low, d.high) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_length_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 2.0])


def test_percentile():
    assert percentile([5, 1, 3], 50) == 3
    assert percentile(list(range(101)), 95) == 95
    with pytest.raises(ValueError):
        percentile([], 50)


def test_empty_bootstrap_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])
