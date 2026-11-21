import os

import numpy as _numpy

_requested = os.environ.get("VIT_DEVICE", "auto").lower()

GPU_AVAILABLE = False
if _requested != "cpu":
    try:
        import cupy as _cupy
        _cupy.cuda.Device(0).compute_capability  # touch the device to confirm it works
        GPU_AVAILABLE = True
    except Exception:
        _cupy = None
        GPU_AVAILABLE = False
else:
    _cupy = None

USE_GPU = GPU_AVAILABLE and _requested != "cpu"
xp = _cupy if USE_GPU else _numpy
DEVICE = "gpu" if USE_GPU else "cpu"


def to_device(arr):
    """Host (NumPy) array -> backend array."""
    if USE_GPU:
        return xp.asarray(arr)
    return arr


def to_host(arr):
    """Backend array -> host NumPy array (for saving/printing/comparison with PIL data)."""
    if USE_GPU:
        return _cupy.asnumpy(arr)
    return arr


def synchronize():
    """No-op on CPU. On GPU, blocks until queued kernels finish -- required around any
    timed region (e.g. latency measurement) since CUDA kernel launches are asynchronous
    and a bare time.perf_counter() would otherwise measure only queueing time.
    """
    if USE_GPU:
        _cupy.cuda.Stream.null.synchronize()
