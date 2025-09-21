from __future__ import annotations
import os, random
import numpy as np
try:
    import tensorflow as tf
except Exception:
    tf = None

_DEF = 42

def seed_everything(seed: int | None) -> int:
    seed = int(seed or _DEF)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        try: tf.random.set_seed(seed)
        except Exception: pass
    return seed

def make_worker_seed(base_seed: int, worker_id: int) -> int:
    return (int(base_seed) ^ ((worker_id + 1) * 0x9E3779B9)) & 0x7FFFFFFF