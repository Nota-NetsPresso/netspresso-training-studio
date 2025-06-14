from typing import List, Optional, Tuple

from loguru import logger
from sqlalchemy.orm import Session

from src.api.v1.schemas.tasks.compression_task import (
    CompressionCreate,
    CompressionCreatePayload,
    CompressionPayload,
)
from src.enums.task import TaskStatus
from src.models.base import generate_uuid
from src.models.compression import CompressionTask
from src.repositories.compression import compression_task_repository
from src.repositories.model import model_repository
from src.worker.compression_task import compress_model


class CompressionTaskService:
    def create_compression_task(self, db: Session, compression_in: CompressionCreate, api_key: str) -> CompressionCreatePayload:
        # Check if a task with the same options already exists
        existing_tasks = compression_task_repository.get_all_by_input_model_id(
            db=db,
            input_model_id=compression_in.input_model_id
        )

        # Filter tasks by compression parameters
        for task in existing_tasks:
            # Check if this task has the same compression parameters
            is_same_options = (
                task.method == compression_in.method and
                float(task.ratio) == float(compression_in.ratio)  # Convert to float for comparison
            )

            if is_same_options:
                # If task is in NOT_STARTED, IN_PROGRESS, or COMPLETED state, return it
                reusable_states = [TaskStatus.NOT_STARTED, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]
                if task.status in reusable_states:
                    logger.info(f"Returning existing compression task with status {task.status}: {task.task_id}")
                    return CompressionCreatePayload(task_id=task.task_id)

                # For STOPPED or ERROR, we'll create a new task below
                logger.info(f"Previous compression task ended with status {task.status}, creating new task")
                break

        # Get model from trained models repository
        compression_task_id = generate_uuid(entity="task")
        _ = compress_model.apply_async(
            kwargs={
                "api_key": api_key,
                "method": compression_in.method,
                "recommendation_method": compression_in.recommendation_method,
                "ratio": compression_in.ratio,
                "options": compression_in.options.model_dump(),
                "input_model_id": compression_in.input_model_id,
                "compression_task_id": compression_task_id,
            },
            compression_task_id=compression_task_id,
        )
        return CompressionCreatePayload(task_id=compression_task_id)

    def get_compression_task(self, db: Session, compression_task_id: str, api_key: str) -> CompressionPayload:
        compression_task = compression_task_repository.get_by_task_id(db=db, task_id=compression_task_id)

        compression_task = CompressionPayload.model_validate(compression_task)
        related_tasks = compression_task_repository.get_all_by_input_model_id(
            db=db, input_model_id=compression_task.input_model_id
        )
        compression_task.related_task_ids = [task.task_id for task in related_tasks]

        return compression_task

    def get_compression_tasks(self, db: Session, model_id: str, token: str) -> List[CompressionPayload]:
        compression_tasks = compression_task_repository.get_all_by_input_model_id(
            db=db, input_model_id=model_id
        )
        compression_tasks = [CompressionPayload.model_validate(task) for task in compression_tasks]

        return compression_tasks

    def delete_compression_task(self, db: Session, compression_task_id: str, api_key: str) -> CompressionPayload:
        compression_task = compression_task_repository.get_by_task_id(db=db, task_id=compression_task_id)
        compression_task = compression_task_repository.soft_delete(db=db, model=compression_task)

        # Delete compressed model from model repository
        model = model_repository.get_by_model_id(db=db, model_id=compression_task.model_id)
        model = model_repository.soft_delete(db=db, model=model)

        compression_task = CompressionPayload.model_validate(compression_task)
        related_tasks = compression_task_repository.get_all_by_input_model_id(
            db=db, input_model_id=compression_task.input_model_id
        )
        compression_task.related_task_ids = [task.task_id for task in related_tasks]

        return compression_task

    def soft_delete_compression_task(self, db: Session, compression_task: CompressionTask) -> CompressionPayload:
        compression_task = compression_task_repository.soft_delete(db=db, model=compression_task)

        return CompressionPayload.model_validate(compression_task)

    def _get_compression_info(self, db: Session, model_id: str) -> Tuple[Optional[str], List[str]]:
        """Get compression task information for a model

        Args:
            db: Database session
            model_id: Model ID to get compression tasks for

        Returns:
            tuple: (latest status, task IDs)
        """
        compression_tasks = compression_task_repository.get_all_by_input_model_id(db=db, input_model_id=model_id)
        if not compression_tasks:
            return None, []

        task_ids = [task.task_id for task in compression_tasks]
        compression_task = compression_task_repository.get_latest_compression_task(db=db, input_model_id=model_id)
        latest_status = compression_task.status

        return latest_status, task_ids

    def get_compression_task_by_model_id(self, db: Session, model_id: str) -> CompressionPayload:
        compression_task = compression_task_repository.get_by_model_id(
            db=db,
            model_id=model_id
        )
        compression_task = CompressionPayload.model_validate(compression_task)
        related_tasks = compression_task_repository.get_all_by_input_model_id(
            db=db, input_model_id=compression_task.input_model_id
        )
        compression_task.related_task_ids = [task.task_id for task in related_tasks]

        return compression_task


compression_task_service = CompressionTaskService()

