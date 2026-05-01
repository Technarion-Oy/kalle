from dataclasses import dataclass

@dataclass
class Config:
    # Model Architecture
    vocab_size: int = 50257
    max_seq_len: int = 512
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1

    # Training Hyperparameters
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    epochs: int = 1
    steps_per_epoch: int = 10000
    
    # DeepSpeed & Batching
    # These should match ds_config.json
    train_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    
    # Logging & Checkpointing
    log_interval: int = 10
    tracker_interval: int = 100
    checkpoint_interval: int = 2500
    checkpoint_dir: str = "checkpoints"

    # Dataset
    dataset_name: str = "roneneldan/TinyStories"
    shuffle_buffer: int = 100
