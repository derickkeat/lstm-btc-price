import numpy as np
import pandas as pd

from src.data.preprocess import normalize, denormalize, build_train_data, split_data


def test_normalize_basic():
    df = pd.DataFrame({
        "a": [0.0, 5.0, 10.0],
        "b": [10.0, 10.0, 10.0],  # constant column -> yields NaN; ensure function doesn't crash
    })

    norm = normalize(df)
    print(norm)
    assert np.isclose(norm["a"].iloc[0], 0.0)
    assert np.isclose(norm["a"].iloc[-1], 1.0)


def test_denormalize_roundtrip():
    original = np.array([100.0, 110.0, 120.0, 140.0])
    # Normalize manually
    min_v, max_v = original.min(), original.max()
    norm = (original - min_v) / (max_v - min_v)

    restored = denormalize(norm, original)
    assert np.allclose(restored, original)


def test_build_train_data_shapes():
    # create tiny dataframe with required columns
    n = 50
    df = pd.DataFrame({
        "open": np.linspace(1, 2, n),
        "high": np.linspace(1, 2, n),
        "low": np.linspace(1, 2, n),
        "close": np.linspace(1, 2, n),
        "volume": np.linspace(1, 2, n),
        "marketCap": np.linspace(1, 2, n),
    })

    past_days, future_days = 10, 2
    x, y = build_train_data(df, past_days=past_days, future_days=future_days)

    # expected sequences
    expected = n - past_days - future_days
    assert x.shape[0] == expected
    assert y.shape[0] == expected
    # x shape: (samples, past_days, features)
    assert x.shape[1] == past_days
    assert x.shape[2] == df.shape[1]
    # y shape: (samples, future_days)
    assert y.shape[1] == future_days


def test_split_data_time_order():
    x = np.arange(100).reshape(20, 5)  # 20 samples
    y = np.arange(20)

    x_train, y_train, x_test, y_test = split_data(x, y, test_size=0.25)

    # last 25% is test
    assert x_test.shape[0] == 5
    assert y_test.shape[0] == 5
    # ensure ordering preserved
    assert np.array_equal(y_train, np.arange(15))
    assert np.array_equal(y_test, np.arange(15, 20))


