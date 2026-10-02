import numpy as np

from predictor_v2.env import ReturnPredictionEnv


def test_environment_shapes_and_step():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(300, 9)).astype(np.float32)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, size=300)))
    env = ReturnPredictionEnv(features, closes, 1, 3, lookback=60)
    obs, info = env.reset(seed=2)
    assert obs.shape == (60 * 9 + 3,)
    assert info == {}
    next_obs, reward, _terminated, truncated, detail = env.step(5)
    assert next_obs.shape == obs.shape
    assert np.isfinite(reward)
    assert not truncated
    assert "actual_return" in detail
