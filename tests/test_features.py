import numpy as np
import pandas as pd

from predictor_v2.features import FEATURE_COLUMNS, build_features, fit_normalizer, normalized_values


def test_features_are_finite():
    n = 150
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    frame = pd.DataFrame({
        "open_time": pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC"),
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": rng.lognormal(4, 0.5, n),
    })
    result = build_features(frame)
    assert list(result.columns[-len(FEATURE_COLUMNS):]) == FEATURE_COLUMNS
    assert np.isfinite(result[FEATURE_COLUMNS].to_numpy()).all()
    mean, std = fit_normalizer([result])
    normalized = normalized_values(result, mean, std)
    assert normalized.shape == (len(result), len(FEATURE_COLUMNS))
    assert np.isfinite(normalized).all()
