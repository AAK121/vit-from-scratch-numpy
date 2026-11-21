import numpy as np


def xavier_uniform(shape, gain=1.0, rng=None):
    rng = rng or np.random.default_rng()
    fan_in, fan_out = shape[0], shape[-1]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def conv_xavier_uniform(shape, gain=1.0, rng=None):
    """Xavier/Glorot uniform for a (kh, kw, in_chans, out_chans) conv weight -- fan_in/fan_out
    include the kernel receptive field (unlike plain `xavier_uniform`, which assumes a 2D
    (in, out) weight and would otherwise use kh/kw as the fan values)."""
    rng = rng or np.random.default_rng()
    kh, kw, cin, cout = shape
    fan_in, fan_out = kh * kw * cin, kh * kw * cout
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def normal(shape, std=0.02, rng=None):
    rng = rng or np.random.default_rng()
    return (rng.standard_normal(size=shape) * std).astype(np.float32)


def zeros(shape):
    return np.zeros(shape, dtype=np.float32)


def ones(shape):
    return np.ones(shape, dtype=np.float32)
