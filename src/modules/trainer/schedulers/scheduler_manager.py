from src.enums.training import Scheduler
from src.modules.trainer.schedulers.schedulers import (
    BaseScheduler,
    CosineAnnealingLRWithCustomWarmUp,
    CosineAnnealingWarmRestartsWithCustomWarmUp,
    MultiStepLR,
    PolynomialLRWithWarmUp,
    StepLR,
)


class SchedulerManager:
    @staticmethod
    def get_scheduler(name: str) -> BaseScheduler:
        scheduler_map = {
            Scheduler.STEP_LR: StepLR,
            Scheduler.POLYNOMIAL_LR: PolynomialLRWithWarmUp,
            Scheduler.COSINE_ANNEALING_LR: CosineAnnealingLRWithCustomWarmUp,
            Scheduler.COSINE_ANNEALING_WARM_RESTARTS: CosineAnnealingWarmRestartsWithCustomWarmUp,
            Scheduler.MULTI_STEP_LR: MultiStepLR,
        }

        scheduler_class = scheduler_map.get(name.lower())
        if not scheduler_class:
            raise ValueError(f"Scheduler '{name}' not found.")

        return scheduler_class()
