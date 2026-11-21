from nn.backend import xp as np


def _no_decay(name):
    """True for params that should NOT get weight decay: LayerNorm gain/bias, all biases,
    and the learnable [CLS]/positional embeddings -- standard AdamW practice for
    transformers (decaying these tends to just fight the norm layers' own rescaling, or
    shrink embeddings toward zero for no representational benefit).
    """
    return (
        name.endswith(".gamma") or name.endswith(".beta") or name.endswith(".b")
        or "cls_token" in name or "pos_embed" in name
    )


class Adam:
    """Adam / AdamW. `decoupled_weight_decay=True` (default) applies weight decay directly
    to the parameter (`p -= lr * wd * p`, Loshchilov & Hutter's AdamW), decoupled from the
    gradient the first/second moment estimates see -- the standard recipe for transformers,
    and the one recommended by essentially every ViT/DeiT training guide. Set
    `decoupled_weight_decay=False` to recover the older L2-regularization-via-gradient
    behavior (`g <- g + wd*p` before the Adam moments). Either way, params matched by
    `_no_decay` (norm gain/bias, biases, cls/pos embeddings) never get decayed.
    """

    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 decoupled_weight_decay=True):
        self.model = model
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.decoupled = decoupled_weight_decay
        self.t = 0
        self.m = {name: np.zeros_like(p) for name, p in model.named_parameters()}
        self.v = {name: np.zeros_like(p) for name, p in model.named_parameters()}
        self.decay_mask = {name: (0.0 if _no_decay(name) else 1.0)
                            for name, _ in model.named_parameters()}

    def step(self):
        self.t += 1
        params = dict(self.model.named_parameters())
        grads = dict(self.model.named_grads())
        bias_c1 = 1 - self.b1 ** self.t
        bias_c2 = 1 - self.b2 ** self.t
        for name, p in params.items():
            g = grads[name]
            wd = self.wd * self.decay_mask[name]
            if wd and not self.decoupled:
                g = g + wd * p
            self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * g
            self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * (g * g)
            mhat = self.m[name] / bias_c1
            vhat = self.v[name] / bias_c2
            update = mhat / (np.sqrt(vhat) + self.eps)
            if wd and self.decoupled:
                update = update + wd * p
            p -= self.lr * update

    def zero_grad(self):
        self.model.zero_grad()

    def set_lr(self, lr):
        self.lr = lr
