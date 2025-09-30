import json
import os
import yaml
import typing
from pydantic import BaseModel


class ModelConfig(BaseModel):
    vocab_size: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    max_seq_len: int
    theta: float = 100000.0
    ffn_type: str = "silu"


class ExperimentConfig(BaseModel):
    name: str = ""
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    batch_size: int = 16
    eval_batch_size: int = 16
    steps: int = 1000
    checkpoints_interval: int = 1
    checkpoints_path: str = "checkpoints"
    eval_steps: int = 100
    use_autocast: bool = False



class DataConfig(BaseModel):
    train_data: str
    valid_data: str


class OptimizerConfig(BaseModel):
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    betas: typing.List[float] = [0.9, 0.999]
    grad_clip: float = 0.0

    lr_schedule: str = "constant"  # Options: "constant", "linear", "cosine"
    lr_min: float = 0
    lr_max: float = 0.001
    warmup_ratio: float = 0.1
    cosine_cycle_ratio: float = 1.0

def load_config_from_file(config_path: str) -> dict:
    """Load configuration from a file and return as a dictionary"""
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path) as f:
        if ext == ".json":
            return json.load(f)
        elif ext in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
