from src.enums.training import Optimizer
from src.modules.trainer.optimizers.optimizers import (
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    BaseOptimizer,
    RMSprop,
)


class OptimizerManager:
    @staticmethod
    def get_optimizer(name: str, lr: float) -> BaseOptimizer:
        optimizer_map = {
            Optimizer.ADADELTA: Adadelta,
            Optimizer.ADAGRAD: Adagrad,
            Optimizer.ADAM: Adam,
            Optimizer.ADAMAX: Adamax,
            Optimizer.ADAMW: AdamW,
            Optimizer.RMSPROP: RMSprop,
            Optimizer.SGD: SGD,
        }

        optimizer_class = optimizer_map.get(name.lower())
        if not optimizer_class:
            raise ValueError(f"Optimizer '{name}' not found.")

        return optimizer_class(lr=lr)
