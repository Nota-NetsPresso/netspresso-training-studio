import tempfile
from pathlib import Path
from typing import List, Optional

from loguru import logger
from netspresso_trainer.evaluator_main import evaluation_with_yaml_impl
from sqlalchemy.orm import Session

from src.core.db.session import get_db_session
from src.enums.conversion import EvaluationTargetFramework
from src.enums.model import ModelType
from src.enums.task import TaskStatus
from src.exceptions.conversion import ConversionTaskNotFoundException
from src.exceptions.evaluation import (
    EvaluationDownloadURLGenerationException,
    EvaluationResultFileNotFoundException,
    EvaluationTaskAlreadyExistsException,
    UnsupportedEvaluationFrameworkException,
)
from src.exceptions.trainer import NotCompletedTrainingException
from src.exceptions.training import TrainingTaskNotFoundException
from src.models.evaluation import EvaluationDataset, EvaluationTask
from src.models.model import Model
from src.modules.trainer.trainer import Trainer
from src.modules.trainer.trainer_configs import TrainerConfigs
from src.repositories.compression import compression_task_repository
from src.repositories.conversion import conversion_task_repository
from src.repositories.evaluation import evaluation_dataset_repository, evaluation_task_repository
from src.repositories.model import model_repository
from src.repositories.training import training_task_repository
from src.utils.file import FileHandler
from src.zenko.storage_handler import ObjectStorageHandler

storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"
EVALUATION_BUCKET_NAME = "evaluation"


class Evaluator:
    def __init__(self, trainer: Optional[Trainer] = None):
        self.trainer = trainer

    def evaluate(self, model_path: str, confidence_score: float, gpus: int = 0):
        if self.trainer is None:
            raise ValueError("Trainer is required for evaluate method")

        try:
            self.trainer.model.checkpoint.path = model_path
            self.trainer.environment.batch_size = 1
            self.trainer.model.postprocessor["params"]["score_thresh"] = confidence_score

            self.configs = TrainerConfigs(
                self.trainer.data,
                self.trainer.augmentation,
                self.trainer.model,
                self.trainer.training,
                self.trainer.logging,
                self.trainer.environment,
            )

            evaluation_logging_dir = evaluation_with_yaml_impl(
                gpus=gpus,
                data=self.configs.data.as_posix(),
                augmentation=self.configs.augmentation.as_posix(),
                model=self.configs.model.as_posix(),
                logging=self.configs.logging.as_posix(),
                environment=self.configs.environment.as_posix(),
            )

            return evaluation_logging_dir

        except Exception as e:
            raise e

    def evaluate_model(
        self,
        db: Session,
        model_id: str,
        evaluation_task_id: str,
        confidence_score: float,
        gpus: int = 0,
    ) -> str:
        try:
            logger.info(f"Starting evaluation for model {model_id} with confidence score {confidence_score}")

            output_dir = tempfile.mkdtemp(prefix="netspresso_evaluate_")

            input_model = model_repository.get_by_model_id(db=db, model_id=model_id)

            local_path = self._download_model(input_model, output_dir)

            conversion_task = None
            try:
                logger.info(f"Getting conversion task for model {model_id}")
                conversion_task = conversion_task_repository.get_by_model_id(db=db, model_id=model_id)
            except Exception:
                logger.info(f"No conversion task found for model {model_id}. Treating as direct ONNX model.")

            if conversion_task is None:
                # For ONNX models, model_id is the same as training_task's output_model_id
                # Get training task based on model type
                if input_model.type == ModelType.COMPRESSED_MODEL:
                    compression_task = compression_task_repository.get_by_model_id(db=db, model_id=input_model.model_id)
                    training_task = training_task_repository.get_by_model_id(db=db, model_id=compression_task.input_model_id)
                else:
                    training_task = training_task_repository.get_by_model_id(db=db, model_id=input_model.model_id)

            else:
                conversion_task_input_model = model_repository.get_by_model_id(db=db, model_id=conversion_task.input_model_id)
                if conversion_task_input_model.type == ModelType.COMPRESSED_MODEL:
                    compression_task = compression_task_repository.get_by_model_id(db=db, model_id=conversion_task_input_model.model_id)
                    training_task = training_task_repository.get_by_model_id(db=db, model_id=compression_task.input_model_id)
                else:
                    training_task = training_task_repository.get_by_model_id(db=db, model_id=conversion_task_input_model.model_id)

                if conversion_task.framework not in [EvaluationTargetFramework.TENSORFLOW_LITE, EvaluationTargetFramework.ONNX]:
                    raise UnsupportedEvaluationFrameworkException(framework=conversion_task.framework)

            if training_task.status != TaskStatus.COMPLETED:
                raise NotCompletedTrainingException(training_task_id=training_task.task_id)

            # Create task with DB session
            evaluation_task = evaluation_task_repository.get_by_task_id(db=db, task_id=evaluation_task_id)
            if self.trainer.test_dataset_id:
                evaluation_task.dataset_id = self.trainer.test_dataset_id
            if self.trainer.test_dataset:
                evaluation_task.dataset_id = self.trainer.test_dataset.dataset_id

            evaluation_task = evaluation_task_repository.save(db=db, model=evaluation_task)
            logger.info(f"Created new evaluation task with ID: {evaluation_task.task_id}")

            # Update status - Pass DB session
            evaluation_task.status = TaskStatus.IN_PROGRESS
            evaluation_task = evaluation_task_repository.save(db=db, model=evaluation_task)

            logger.info(f"Running evaluation with confidence score: {confidence_score}")
            evaluation_logging_dir = self.evaluate(
                model_path=str(local_path),
                confidence_score=confidence_score,
                gpus=gpus
            )
            logger.info(f"Evaluation completed. Logging directory: {evaluation_logging_dir}")

            # predictions.json file path
            predictions_file = evaluation_logging_dir / "predictions.json"
            logger.info(f"Predictions file: {predictions_file}")

            # Raise error if predictions.json file doesn't exist
            if not predictions_file.exists():
                error_msg = f"Predictions file not found: {predictions_file}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Upload predictions.json file to evaluation bucket
            object_path = f"{input_model.user_id}/{evaluation_task.task_id}/predictions.json"
            logger.info(f"Uploading predictions.json to storage: {object_path}")
            storage_handler.upload_file_to_s3(
                bucket_name=EVALUATION_BUCKET_NAME,
                local_path=str(predictions_file),
                object_path=object_path
            )

            # Upload result images to evaluation bucket
            result_images_dir = Path(evaluation_logging_dir) / "result_image" / "evaluation"
            VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
            for image_file in result_images_dir.glob("*"):
                if image_file.suffix not in VALID_IMAGE_EXTENSIONS:
                    logger.warning(f"Skipping non-image file: {image_file}")
                    continue
                object_path = f"{input_model.user_id}/{evaluation_task.task_id}/result_images/{image_file.name}"
                logger.info(f"Uploading result image to storage: {object_path}")
                storage_handler.upload_file_to_s3(
                    bucket_name=EVALUATION_BUCKET_NAME,
                    local_path=str(image_file),
                    object_path=object_path
                )

            evaluation_summary_path = Path(evaluation_logging_dir) / "evaluation_summary.json"
            evaluation_summary = FileHandler.load_json(evaluation_summary_path)

            # Update status after evaluation complete - Use the same DB session
            evaluation_task.metrics = evaluation_summary["metrics"]
            evaluation_task.metrics_names = evaluation_summary["metrics_list"]
            evaluation_task.primary_metric = evaluation_summary["primary_metric"]
            evaluation_task.status = TaskStatus.COMPLETED
            evaluation_task.results_path = object_path  # Save storage path
            evaluation_task_repository.save(db=db, model=evaluation_task)

            return evaluation_task.task_id

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")

            # Use the same session for error handling
            if evaluation_task is not None:
                try:
                    evaluation_task.status = TaskStatus.ERROR
                    evaluation_task.error_detail = {"error": str(e)}
                    evaluation_task_repository.save(db=db, model=evaluation_task)
                except Exception as inner_e:
                    logger.error(f"Failed to update task status: {str(inner_e)}")
            raise e

    def _download_model(self, input_model: Model, output_dir: str) -> Path:
        download_dir = Path(output_dir) / "input_model"
        download_dir.mkdir(parents=True, exist_ok=True)

        if input_model.type == ModelType.TRAINED_MODEL:
            remote_model_path = Path(input_model.object_path) / "model.onnx"
        elif input_model.type == ModelType.COMPRESSED_MODEL:
            remote_model_path = Path(input_model.object_path).parent / "model.onnx"
        else:
            remote_model_path = Path(input_model.object_path)

        local_path = download_dir / remote_model_path.name

        logger.info(f"Downloading input model from Zenko: {remote_model_path}")
        storage_handler.download_file_from_s3(
            bucket_name=BUCKET_NAME,
            local_path=str(local_path),
            object_path=str(remote_model_path)
        )
        logger.info(f"Downloaded input model from Zenko: {local_path}")

        return local_path
