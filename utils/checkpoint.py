
import json

import numpy as np

from nn.backend import to_device, to_host


def save_checkpoint(model, path, extra=None):
    """path: base path without extension. Writes path.npz and path.json."""
    params = dict(model.named_parameters())
    save_dict = {name.replace(".", "__"): to_host(arr) for name, arr in params.items()}
    np.savez(path + ".npz", **save_dict)
    meta = dict(config=model.config, extra=extra or {})
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_meta(path):
    with open(path + ".json") as f:
        return json.load(f)


def load_into_model(model, path):
    data = np.load(path + ".npz")
    params = dict(model.named_parameters())
    for key in data.files:
        name = key.replace("__", ".")
        if name not in params:
            raise KeyError(f"Checkpoint has unexpected parameter '{name}' not found in model")
        params[name][...] = to_device(data[key])
    return load_meta(path)
