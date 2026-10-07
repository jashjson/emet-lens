import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"
