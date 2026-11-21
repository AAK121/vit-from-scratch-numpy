from nn.backend import xp as np


def softmax_cross_entropy(logits, labels, label_smoothing=0.0):
    """Softmax cross-entropy with optional label smoothing.

    label_smoothing=eps replaces the one-hot target with (1-eps) on the true class plus
    eps/K spread uniformly over all K classes -- a standard, cheap regularizer for
    small-data classification (DeiT/ViT training recipes default to eps=0.1). eps=0
    (default) reduces exactly to plain one-hot cross-entropy, so every existing call site
    (and scripts/gradcheck.py) is unaffected unless it opts in.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)

    B, K = logits.shape
    log_probs = np.log(np.clip(probs, 1e-12, None))

    if label_smoothing > 0.0:
        eps = label_smoothing
        off_value = eps / K
        on_value = 1.0 - eps + off_value
        targets = np.full((B, K), off_value, dtype=probs.dtype)
        targets[np.arange(B), labels] = on_value
        loss = -(targets * log_probs).sum(axis=-1).mean()
        dlogits = (probs - targets) / B
    else:
        correct_log_probs = log_probs[np.arange(B), labels]
        loss = -correct_log_probs.mean()
        dlogits = probs.copy()
        dlogits[np.arange(B), labels] -= 1.0
        dlogits /= B

    return loss, dlogits, probs
