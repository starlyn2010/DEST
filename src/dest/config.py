import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class RunResult:
    experiment_id: str
    dataset: str
    sampler_name: str
    seed: int
    mode: str
    train_losses: List[float]
    val_losses: List[float]
    test_losses: List[float]
    train_accs: List[float]
    val_accs: List[float]
    test_accs: List[float]
    generalization_gaps: List[float]
    f1_per_epoch: List[float]
    precision_per_epoch: List[float]
    recall_per_epoch: List[float]
    final_test_acc: float
    final_test_loss: float
    final_f1: float
    final_precision: float
    final_recall: float
    final_ece: float
    final_generalization_gap: float
    convergence_epoch_90: Optional[int]
    convergence_epoch_95: Optional[int]
    best_test_acc: float
    best_test_epoch: int
    sampler_time_per_epoch: List[float]
    train_time_per_epoch: List[float]
    eval_time_per_epoch: List[float]
    total_time_per_epoch: List[float]
    total_runtime_seconds: float
    samples_per_second: List[float]
    gpu_memory_peak_mb: float
    train_loss_variance: float
    test_acc_variance: float
    config_snapshot: Dict[str, Any]
    timestamp: str
    status: str = "COMPLETE"


# Canonical dataset -> model mapping
DATASET_MODEL_MAP: Dict[str, str] = {
    "MNIST":        "SmallCNN",
    "FASHIONMNIST": "SmallCNN",
    "CIFAR10":      "ResNet9",
    "CIFAR100":     "ResNet18",
    "TINYIMAGENET": "ResNet18",
}

# Dataset difficulty rank (for scaling plot x-axis)
DATASET_DIFFICULTY: Dict[str, int] = {
    "MNIST":        0,
    "FASHIONMNIST": 1,
    "CIFAR10":      2,
    "CIFAR100":     3,
    "TINYIMAGENET": 4,
}

# Samplers evaluated in Phase 2
SAMPLERS_TO_COMPARE: List[str] = [
    "stochastic",
    "sobol",
    "collatz_v1",
    "collatz_v2",
    "collatz_v3",
]

_BASE: Dict[str, Any] = {
    "optimizer":       "SGD",
    "momentum":        0.9,
    "weight_decay":    1e-4,
    "val_fraction":    0.1,
    "device":          "auto",
    "num_workers":     2,
    "save_checkpoints": True,
    "verbose":         True,
    "samplers":        SAMPLERS_TO_COMPARE,
    "dataset_model_map": DATASET_MODEL_MAP,
}

_MODES: Dict[str, Dict[str, Any]] = {
    "DEBUG": {
        "execution_mode": "DEBUG",
        "datasets":       ["MNIST"],
        "epochs":         3,
        "seeds":          [42, 43, 44],
        "batch_size":     256,
        "lr":             0.01,
        "lr_schedule":    "constant",
        "output_dir":     "./dest_scaling_debug",
    },
    "VALIDATION": {
        "execution_mode": "VALIDATION",
        "datasets":       ["FASHIONMNIST"],
        "epochs":         10,
        "seeds":          list(range(42, 52)),
        "batch_size":     256,
        "lr":             0.01,
        "lr_schedule":    "cosine",
        "output_dir":     "./dest_scaling_validation",
    },
    "PAPER": {
        "execution_mode": "PAPER",
        "datasets":       ["CIFAR10"],
        "epochs":         15,
        "seeds":          list(range(42, 62)),
        "batch_size":     128,
        "lr":             0.01,
        "lr_schedule":    "cosine",
        "output_dir":     "./dest_scaling_paper",
    },
    "FULL": {
        "execution_mode": "FULL",
        "datasets":       ["MNIST", "FASHIONMNIST", "CIFAR10", "CIFAR100", "TINYIMAGENET"],
        "epochs":         20,
        "seeds":          list(range(42, 62)),
        "batch_size":     128,
        "lr":             0.01,
        "lr_schedule":    "cosine",
        "output_dir":     "./dest_scaling_full",
    },
}


def get_config(
    mode: str = "DEBUG",
    dataset_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a fully-populated configuration dict for the given execution mode."""
    key = mode.upper()
    if key not in _MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(_MODES.keys())}")
    cfg = {**_BASE, **_MODES[key]}
    if dataset_override:
        cfg["datasets"] = [dataset_override.upper()]
    if model_override:
        cfg["model_override"] = model_override
    return cfg
