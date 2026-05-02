try:
    from .base_trainer_accelerate import BaseTrainer
except ModuleNotFoundError:  # pragma: no cover - dependency may be absent in test env
    BaseTrainer = None

try:
    from .pi3_trainer import Pi3Trainer
except ModuleNotFoundError:  # pragma: no cover - dependency may be absent in test env
    Pi3Trainer = None

try:
    from .pi3x_trainer import Pi3XTrainer
except ModuleNotFoundError:  # pragma: no cover - dependency may be absent in test env
    Pi3XTrainer = None
