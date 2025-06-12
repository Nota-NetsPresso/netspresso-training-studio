from typing import List, Optional

from sqlalchemy.orm import Session

from src.enums.task import TaskStatus, TaskType
from src.exceptions.task import TaskIsDeletedException, TaskNotFoundException
from src.models.training import TrainingTask
from src.repositories.base import BaseRepository


class TrainingTaskRepository(BaseRepository[TrainingTask]):
    def __is_available(self, task: Optional[TrainingTask]) -> TrainingTask:
        if task is None:
            raise TaskNotFoundException(task_type=TaskType.TRAINING)

        if task.is_deleted:
            raise TaskIsDeletedException(task_type=TaskType.TRAINING, task_id=task.task_id)

        return task

    def get_by_task_id(self, db: Session, task_id: str) -> Optional[TrainingTask]:
        conditions = [self.model.task_id == task_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(task=task)

    def get_by_model_id(self, db: Session, model_id: str) -> Optional[TrainingTask]:
        conditions = [self.model.model_id == model_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(task=task)

    def get_completed_tasks(self, db: Session, user_id: str) -> List[TrainingTask]:
        conditions = [self.model.status == TaskStatus.COMPLETED, self.model.user_id == user_id]
        tasks = self.find_all(
            db=db,
            conditions=conditions,
        )

        return tasks


training_task_repository = TrainingTaskRepository(TrainingTask)
