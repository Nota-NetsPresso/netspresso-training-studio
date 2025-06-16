from typing import List, Optional, Tuple

from loguru import logger
from sqlalchemy.orm import Session

from src.api.v1.schemas.tasks.compression.compression_task import (
    CompressionCreate,
    CompressionCreatePayload,
    CompressionPayload,
)
from src.enums.compression import CompressionMethod
from src.enums.model import ModelType
from src.enums.task import TaskStatus
from src.models.base import generate_uuid
from src.models.compression import CompressionTask
from src.models.model import Model
from src.repositories.compression import compression_task_repository
from src.repositories.model import model_repository
from src.worker.compression_task import compress_model


class CompressionTaskService:
    def _generate_model_name(self, input_model_name: str, compression_method: CompressionMethod, recommendation_ratio: float) -> str:
        model_name_parts = [
            input_model_name,
            compression_method.value,
            recommendation_ratio,
        ]
        model_name = "_".join(map(str, model_name_parts))
        logger.info(f"Model name: {model_name}")

        return model_name

    def create_compressed_model(self, db: Session, model: Model, compression_method: CompressionMethod, recommendation_ratio: float) -> Model:
        model_id = generate_uuid(entity="model")
        base_object_path = f"{model.user_id}/{model.project_id}/{model_id}"
        model_name = self._generate_model_name(
            input_model_name=model.name,
            compression_method=compression_method,
            recommendation_ratio=recommendation_ratio,
        )
        model_obj = Model(
            model_id=model_id,
            name=model_name,
            type=ModelType.COMPRESSED_MODEL,
            is_retrainable=True,
            project_id=model.project_id,
            user_id=model.user_id,
            object_path=base_object_path  # Store base path only
        )
        model_obj = model_repository.save(db=db, model=model_obj)

        return model_obj

    def check_compression_task_exists(self, db: Session, compression_in: CompressionCreate) -> Optional[CompressionCreatePayload]:
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

        return None

    def create_compression_task(self, db: Session, compression_in: CompressionCreate, api_key: str) -> CompressionTask:
        input_model = model_repository.get_by_model_id(db=db, model_id=compression_in.input_model_id)

        compressed_model = self.create_compressed_model(
            db=db,
            model=input_model,
            compression_method=compression_in.method,
            recommendation_ratio=compression_in.ratio,
        )

        compression_task = CompressionTask(
            method=compression_in.method,
            ratio=compression_in.ratio,
            options=compression_in.options.model_dump(),
            status=TaskStatus.NOT_STARTED,
            input_model_id=compression_in.input_model_id,
            model_id=compressed_model.model_id,
            user_id=input_model.user_id,
        )
        compression_task = compression_task_repository.save(db=db, model=compression_task)

        return compression_task

    def start_compression_task(
        self,
        compression_in: CompressionCreate,
        compression_task: CompressionTask,
        api_key: str,
    ) -> CompressionCreatePayload:
        # Prepare worker task parameters
        worker_params = {
            "api_key": api_key,
            "input_model_id": compression_in.input_model_id,
            "compression_task_id": compression_task.task_id,
            "compression_method": compression_in.method,
            "recommendation_method": compression_in.recommendation_method,
            "recommendation_ratio": compression_in.ratio,
            "options": compression_in.options.model_dump(),
        }

        _ = compress_model.apply_async(
            kwargs=worker_params,
            compression_task_id=compression_task.task_id,
        )

        return CompressionCreatePayload(task_id=compression_task.task_id)

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

