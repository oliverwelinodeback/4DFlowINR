import random
from pathlib import Path
import numpy as np
import torch

def save_ckpt(
    path,
    model,
    adam_opt=None,
    lbfgs_opt=None,
    scheduler=None,
    iteration=0,
    rng=True,
    c_weights=None,
    loss_weights=None,
    standardization_factors=None,
    U_max=None,
    amp_scaler=None,
    config_dict=None,
):
    """Save a training checkpoint, including optimizer and RNG state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            None if adam_opt is None else adam_opt.state_dict()
        ),
        "lbfgs_state_dict": (
            None if lbfgs_opt is None else lbfgs_opt.state_dict()
        ),
        "scheduler_state_dict": (
            None if scheduler is None else scheduler.state_dict()
        ),
        "c_weights": (
            None if c_weights is None else np.asarray(c_weights)
        ),
        "loss_weights": (
            None
            if loss_weights is None
            else (
                loss_weights
                if isinstance(loss_weights, (list, tuple, dict))
                else np.asarray(loss_weights)
            )
        ),
        "standardization_factors": standardization_factors,
        "U_max": U_max,
        "amp_scaler_state_dict": (
            None if amp_scaler is None else amp_scaler.state_dict()
        ),
        "config": config_dict,
    }

    if rng:
        checkpoint["rng"] = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    torch.save(checkpoint, str(path))

def load_model_weights(path, model, device):
    """Load model weights from a 4DFlowINR checkpoint."""

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint '{path}' does not contain model_state_dict."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint