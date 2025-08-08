import numpy as np
import pandas as pd
import pytest

# Skip this module's tests entirely if 'ta' is not installed
ta = None
try:
    import ta  # noqa: F401
    ta = True
except Exception:  # pragma: no cover
    pytest.skip("'ta' not installed; skipping indicator tests", allow_module_level=True)

from src.features.indicators import calculate_technical_indicators


def test_indicators_add_columns():
    # create minimal OHLCV data for indicators
    n = 50
    df = pd.DataFrame({
        "high": np.linspace(1, 10, n),
        "low": np.linspace(0.5, 9.5, n),
        "close": np.linspace(1, 9, n),
        "open": np.linspace(0.8, 8.8, n),
        "volume": np.linspace(100, 200, n),
        "marketCap": np.linspace(1000, 2000, n),
    })

    out = calculate_technical_indicators(df.copy())

    expected_cols = {
        "k_9", "d_9", "ma_6", "ma_9", "ma_12", "bias_6", "bias_3",
        "bias_3_minus_bias_6", "rsi_6", "williams_r_12", "momentum_6",
        "momentum_6_ma", "macd_9", "macd_signal_9", "macd_diff"
    }

    assert expected_cols.issubset(set(out.columns))

