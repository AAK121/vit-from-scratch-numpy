import time

import numpy as np

from nn.backend import synchronize, to_device
from nn.losses import softmax_cross_entropy


def accuracy(logits, labels):
    preds = logits.argmax(axis=-1)
    return float((preds == labels).mean())


def evaluate(model, loader):
    """Forward-only loss + accuracy over an entire loader (validation/test)."""
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    for imgs, labels in loader:
        logits = model.forward(imgs, training=False)
        loss, _, _ = softmax_cross_entropy(logits, labels)
        preds = logits.argmax(axis=-1)
        n_correct += int((preds == labels).sum())
        n_total += len(labels)
        total_loss += float(loss) * len(labels)
    return dict(
        loss=total_loss / max(1, n_total),
        accuracy=n_correct / max(1, n_total),
        n=n_total,
    )


def measure_inference(model, loader):
    """Forward-only pass over an entire loader. Returns accuracy/timing/throughput."""
    n_correct = 0
    n_total = 0
    t0 = time.perf_counter()
    for imgs, labels in loader:
        logits = model.forward(imgs, training=False)
        preds = logits.argmax(axis=-1)
        n_correct += int((preds == labels).sum())
        n_total += len(labels)
    elapsed = time.perf_counter() - t0
    acc = n_correct / max(1, n_total)
    throughput = n_total / elapsed if elapsed > 0 else float("inf")
    return dict(accuracy=acc, total_time_s=elapsed, throughput_img_s=throughput, n=n_total)


def measure_latency(model, input_shape, batch_size=1, n_warmup=5, n_iters=20, seed=0):
    """Wall-clock latency of a single forward pass at the given batch size."""
    rng = np.random.default_rng(seed)
    x = to_device(rng.standard_normal((batch_size,) + input_shape).astype(np.float32))
    for _ in range(n_warmup):
        model.forward(x, training=False)
    synchronize()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model.forward(x, training=False)
        synchronize()
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return dict(
        mean_ms=float(times.mean() * 1000),
        std_ms=float(times.std() * 1000),
        p50_ms=float(np.median(times) * 1000),
        throughput_img_s=float(batch_size / times.mean()),
    )
