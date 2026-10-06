"""Reference per-update linear warmup followed by inverse-square-root decay."""
import math
from transformer.config import TransformerConfig


def learning_rate(update: int, config: TransformerConfig) -> float:
    """update is the completed-update count; zero returns warmup_init_lr."""
    if update < 0:
        raise ValueError("update must be nonnegative")
    if config.fixed_lr:
        return config.learning_rate
    if update <= config.warmup_updates:
        return config.warmup_init_lr + (config.learning_rate - config.warmup_init_lr) * update / config.warmup_updates
    return config.learning_rate * math.sqrt(config.warmup_updates / update)
