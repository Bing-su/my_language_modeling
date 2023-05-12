import os
from platform import uname

import torch
from pytorch_optimizer import load_optimizer
from torch.optim import SGD, Adam, AdamW


def is_in_wsl() -> bool:
    return "microsoft-standard" in uname().release


def create_optimizer(name: str):
    name = name.lower()

    if name == "adam":
        return Adam
    elif name == "adamw":
        return AdamW
    elif name == "sgd":
        return SGD
    elif name.endswith("bnb"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for BNB optimizers")

        if is_in_wsl():
            os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib"

        try:
            import bitsandbytes.optim as optim
        except ImportError as e:
            raise ImportError("install bitsandbytes first") from e

        m = {
            "adagrad_bnb": optim.Adagrad8bit,
            "adam_bnb": optim.Adam8bit,
            "adamw_bnb": optim.AdamW8bit,
            "lamb_bnb": optim.LAMB8bit,
            "lars_bnb": optim.LARS8bit,
            "rmsprop_bnb": optim.RMSprop8bit,
            "lion_bnb": optim.Lion8bit,
            "sgd_bnb": optim.SGD8bit,
        }
        return m[name]
    else:
        return load_optimizer(name)
