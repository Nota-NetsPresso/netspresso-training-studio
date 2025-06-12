from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.api.v1.schemas.model import ModelPayload, PresignedUrl
from src.configs.settings import settings
from src.enums.model import ModelType
from src.enums.task import RetrievalTaskType, TaskStatus
from src.exceptions.auth import UnauthorizedUserAccessException
from src.exceptions.model import ModelCannotBeDeletedException
from src.repositories.model import model_repository
from src.services.benchmark_task import benchmark_task_service
from src.services.compression_task import compression_task_service
from src.services.conversion_task import conversion_task_service
from src.services.evaluation_task import evaluation_task_service
from src.services.training_task import training_task_service
from src.services.user import user_service
from src.zenko.storage_handler import ObjectStorageHandler


class ModelService:
    def __init__(self):
        self.storage_handler = ObjectStorageHandler()
        self.BUCKET_NAME = settings.MODEL_BUCKET_NAME

    def _attach_child_task_info(self, db: Session, model: ModelPayload) -> ModelPayload:
        """Attach child tasks (conversion, benchmark, evaluation) information to model

        Args:
            db: Database session
            model: Model to attach task information to

        Returns:
            ModelPayload: Model with attached task information
        """
        # Get evaluation tasks
        eval_status, eval_task_ids = evaluation_task_service._get_evaluation_info(db, model.model_id)
        if eval_status:
            model.latest_experiments.evaluate = eval_status
            model.evaluation_task_ids.extend(eval_task_ids)

        # Get compression tasks
        comp_status, comp_task_ids = compression_task_service._get_compression_info(db, model.model_id)
        if comp_status:
            model.latest_experiments.compress = comp_status
            model.compress_task_ids.extend(comp_task_ids)

        # Get conversion tasks and their benchmark tasks
        conv_status, conv_task_ids, conv_model_ids = conversion_task_service._get_conversion_info(db, model.model_id)
        if conv_status:
            model.latest_experiments.convert = conv_status
            model.convert_task_ids.extend(conv_task_ids)

            # Get benchmark tasks for converted models
            bench_status, bench_task_ids = benchmark_task_service._get_benchmark_info(db, conv_model_ids)
            if bench_status:
                model.latest_experiments.benchmark = bench_status
                model.benchmark_task_ids.extend(bench_task_ids)

        return model

    def get_models(
        self,
        db: Session,
        token: str,
        task_type: Optional[RetrievalTaskType] = None,
        project_id: Optional[str] = None
    ) -> List[ModelPayload]:
        user_info = user_service.get_user_info(token=token)

        # Base query by user ID
        user_id = user_info.user_id

        # If project_id is provided, filter by project
        if project_id:
            models = model_repository.get_all_by_project_id(db=db, project_id=project_id)
            # Filter models that belong to the user for security
            models = [model for model in models if model.user_id == user_id]
        else:
            models = model_repository.get_all_by_user_id(db=db, user_id=user_id)

        new_models = []
        for model in models:
            # Handle retraining task - only return completed trained and compressed models
            if task_type == RetrievalTaskType.RETRAIN:
                if model.type not in [ModelType.TRAINED_MODEL, ModelType.COMPRESSED_MODEL]:
                    continue
                training_task = (
                    training_task_service.get_training_task_by_model_id(db=db, model_id=model.model_id)
                    if model.type == ModelType.TRAINED_MODEL
                    else training_task_service.get_training_task_by_model_id(
                        db=db,
                        model_id=compression_task_service.get_compression_task_by_model_id(
                            db=db,
                            model_id=model.model_id
                        ).input_model_id
                    )
                )
                if not training_task or training_task.status != TaskStatus.COMPLETED:
                    continue
            # Handle compression task separately as it only needs trained models
            elif task_type == RetrievalTaskType.COMPRESS:
                if model.type != ModelType.TRAINED_MODEL:
                    continue
            # Handle other conversion/evaluation tasks that can use both trained and compressed models
            elif task_type in [RetrievalTaskType.BENCHMARK, RetrievalTaskType.EVALUATE, RetrievalTaskType.CONVERT]:
                if model.type not in [ModelType.TRAINED_MODEL, ModelType.COMPRESSED_MODEL]:
                    continue
            # For other cases (like listing), exclude converted and benchmarked models
            else:
                if model.type in [ModelType.CONVERTED_MODEL, ModelType.BENCHMARKED_MODEL]:
                    continue

            model_payload = ModelPayload.model_validate(model)

            # Get training task based on model type
            if model.type == ModelType.COMPRESSED_MODEL:
                compression_task = compression_task_service.get_compression_task_by_model_id(
                    db=db,
                    model_id=model.model_id
                )
                training_task = training_task_service.get_training_task_by_model_id(
                    db=db,
                    model_id=compression_task.input_model_id
                )
            else:
                training_task = training_task_service.get_training_task_by_model_id(
                    db=db,
                    model_id=model.model_id
                )

            model_payload.train_task_id = training_task.task_id
            model_payload.status = training_task.status
            model_payload = self._attach_child_task_info(db, model_payload)
            new_models.append(model_payload)

        return new_models

    def get_model(self, db: Session, model_id: str, token: str) -> ModelPayload:
        user_info = user_service.get_user_info(token=token)

        model = model_repository.get_by_model_id(db=db, model_id=model_id)
        if user_info.user_id != model.user_id:
            raise UnauthorizedUserAccessException(user_id=user_info.user_id)

        model_payload = ModelPayload.model_validate(model)

        if model.type == ModelType.COMPRESSED_MODEL:
            compression_task = compression_task_service.get_compression_task_by_model_id(
                db=db,
                model_id=model_id
            )
            training_task = training_task_service.get_training_task_by_model_id(
                db=db,
                model_id=compression_task.input_model_id
            )
            model_payload.status = compression_task.status
        else:
            training_task = training_task_service.get_training_task_by_model_id(db=db, model_id=model_id)
            model_payload.status = training_task.status

        model_payload.train_task_id = training_task.task_id
        return self._attach_child_task_info(db, model_payload)

    def delete_model(self, db: Session, model_id: str, token: str) -> ModelPayload:
        """Delete model and all related tasks

        Args:
            db: Database session
            model_id: Model ID to delete
            token: API key for authentication

        Returns:
            ModelPayload: Deleted model info

        Raises:
            HTTPException: If model not found
        """
        user_info = user_service.get_user_info(token=token)

        # Get model before deletion
        model = model_repository.get_by_model_id(db=db, model_id=model_id)
        if user_info.user_id != model.user_id:
            raise UnauthorizedUserAccessException(user_id=user_info.user_id)

        if model.type not in [ModelType.TRAINED_MODEL, ModelType.COMPRESSED_MODEL]:
            raise ModelCannotBeDeletedException(model_id=model_id)

        # Delete model and training task
        model = model_repository.soft_delete(db=db, model=model)

        if model.type == ModelType.COMPRESSED_MODEL:
            compression_task = compression_task_service.get_compression_task_by_model_id(
                db=db,
                model_id=model_id
            )
            training_task = training_task_service.get_training_task_by_model_id(
                db=db,
                model_id=compression_task.input_model_id
            )
            compression_task_service.soft_delete_compression_task(db=db, compression_task=compression_task)
        else:
            training_task = training_task_service.get_training_task_by_model_id(db=db, model_id=model_id)

        # Process and return model info
        model_payload = ModelPayload.model_validate(model)
        model_payload.train_task_id = training_task.task_id
        model_payload.status = training_task.status

        return self._attach_child_task_info(db, model_payload)

    def download_model(self, db: Session, model_id: str, token: str) -> PresignedUrl:
        """Download model file from Zenko

        Args:
            db: Database session
            model_id: Model ID to download
            token: API key for authentication

        Returns:
            FileResponse: Model file response

        Raises:
            HTTPException: If model not found or file not accessible
        """
        user_info = user_service.get_user_info(token=token)

        model = model_repository.get_by_model_id(db=db, model_id=model_id)
        if user_info.user_id != model.user_id:
            raise UnauthorizedUserAccessException(user_id=user_info.user_id)

        if model.type == ModelType.TRAINED_MODEL:
            object_path = f"{model.object_path}/model.onnx"
            extension = "onnx"
        else:
            object_path = model.object_path
            extension = Path(object_path).suffix.lstrip('.')

        try:
            # Generate presigned URL for download
            download_name = f"{model.name}.{extension}"  # 모델 이름으로 파일명 생성
            url = self.storage_handler.get_download_presigned_url(
                bucket_name=self.BUCKET_NAME,
                object_path=object_path,
                download_name=download_name,
                expires_in=3600  # URL expires in 1 hour
            )

            return PresignedUrl(
                model_id=model.model_id,
                file_name=download_name,  # 변경된 파일명 사용
                presigned_url=url
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate download URL: {str(e)}"
            )


model_service = ModelService()
