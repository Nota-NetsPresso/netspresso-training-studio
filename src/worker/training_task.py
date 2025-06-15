import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from src.api.v1.schemas.tasks.training_task import TrainingCreate
from src.configs.settings import settings
from src.core.db.session import get_db_session
from src.enums.conversion import EvaluationTargetFramework
from src.enums.task import TaskStatus
from src.enums.training import StorageLocation
from src.modules.trainer.augmentations.augmentation import Normalize, Pad, Resize, ToTensor
from src.modules.trainer.optimizers.optimizer_manager import OptimizerManager
from src.modules.trainer.schedulers.scheduler_manager import SchedulerManager
from src.modules.trainer.storage.dataforge import Split
from src.modules.trainer.trainer import Trainer
from src.repositories.model import model_repository
from src.repositories.project import project_repository
from src.repositories.training import training_task_repository
from src.worker.celery_app import celery_app
from src.zenko.storage_handler import ObjectStorageHandler

# Constants
DEFAULT_CONFIDENCE_SCORES = [0.3, 0.5, 0.6]
DEFAULT_AUGMENTATIONS = [Resize(), Pad(fill=114), ToTensor(), Normalize()]
storage_handler = ObjectStorageHandler()

def prepare_training_data(trainer: Trainer, training_in: TrainingCreate) -> Path:
    """Prepare training data based on storage location.

    This function handles both local and storage datasets:
    - For local datasets: Validates and uses the specified local path
    - For storage datasets: Downloads and prepares data from storage

    Args:
        trainer: Trainer object for model training
        training_in: Training configuration dictionary

    Returns:
        Path: Directory path containing the prepared dataset

    Raises:
        ValueError: If local dataset paths don't exist
    """
    dataset_info = training_in.dataset

    if dataset_info.storage_location == StorageLocation.LOCAL:
        # Use local dataset
        train_dataset_path = Path(dataset_info.train_path)
        if not train_dataset_path.exists():
            raise ValueError(f"Training dataset not found at path: {train_dataset_path}")

        if dataset_info.test_path:
            test_dataset_path = Path(dataset_info.test_path)
            if not test_dataset_path.exists():
                raise ValueError(f"Test dataset not found at path: {test_dataset_path}")

        # Configure local dataset
        trainer.set_dataset(
            dataset_root_path=str(train_dataset_path),
            dataset_name=train_dataset_path.name,
        )
        return train_dataset_path

    else:  # StorageLocation.STORAGE
        dataset_dir = Path(settings.NP_TRAINING_STUDIO_PATH) / "datasets" / "storage"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading training dataset: {dataset_info.train_path}")
        train_dataset_path = trainer.download_dataset_for_training(
            dataset_uuid=dataset_info.train_path,
            output_dir=dataset_dir.as_posix()
        )

        train_dataset_version = trainer.get_dataset_version_from_storage(
            dataset_uuid=dataset_info.train_path,
            split=Split.TRAIN
        )

        train_dataset_info = trainer.get_dataset_info_from_storage(
            project_id=train_dataset_version.project_id,
            dataset_uuid=dataset_info.train_path,
            split=Split.TRAIN
        )

        trainer.set_dataset(
            dataset_root_path=train_dataset_path,
            dataset_name=train_dataset_info.dataset.dataset_title,
        )

        return dataset_dir


def configure_model_and_training(trainer: Trainer, training_in: TrainingCreate, input_model_info: Optional[Dict[str, Any]] = None):
    """Configure model, augmentations, and training parameters.

    Args:
        trainer: Trainer instance
        training_in: Training configuration
        input_model_info: Optional information about input model for retraining
    """
    img_size = training_in.input_shapes[0].dimension[0]

    if training_in.pretrained_model:
        trainer.set_model_config(
            model_name=training_in.pretrained_model,
            img_size=img_size,
        )
    elif training_in.input_model_id and input_model_info:
        # Use input model info provided by the service
        temp_dir = tempfile.mkdtemp(prefix="netspresso_training_")
        download_dir = Path(temp_dir) / "input_model"
        download_dir.mkdir(parents=True, exist_ok=True)

        remote_model_path = input_model_info["remote_model_path"]
        local_path = download_dir / Path(remote_model_path).name

        logger.info(f"Downloading input model from Zenko: {remote_model_path}")
        storage_handler.download_file_from_s3(
            bucket_name=settings.MODEL_BUCKET_NAME,
            local_path=str(local_path),
            object_path=str(remote_model_path),
        )
        logger.info(f"Downloaded input model from Zenko: {local_path}")

        trainer.set_model_config(
            model_name=input_model_info["pretrained_model"],
            img_size=img_size,
            path=str(local_path),
        )
    else:
        raise ValueError("Either pretrained_model or input_model_id with input_model_info must be provided")

    logger.info(f"Setting model config with size: {img_size} and model: {training_in.pretrained_model or input_model_info.get('pretrained_model')}")

    trainer.set_augmentation_config(
        train_transforms=DEFAULT_AUGMENTATIONS,
        inference_transforms=DEFAULT_AUGMENTATIONS,
    )

    optimizer = OptimizerManager.get_optimizer(
        name=training_in.hyperparameter.optimizer,
        lr=training_in.hyperparameter.learning_rate,
    )

    scheduler = SchedulerManager.get_scheduler(
        name=training_in.hyperparameter.scheduler
    )

    trainer.set_training_config(
        epochs=training_in.hyperparameter.epochs,
        batch_size=training_in.hyperparameter.batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def prepare_evaluation_data(trainer: Trainer, training_in: TrainingCreate, dataset_dir: Path):
    """Download and prepare evaluation dataset.

    This function handles both local and storage datasets for evaluation:
    - For local datasets: Validates and uses the specified local path
    - For storage datasets: Downloads and prepares data from storage

    Args:
        trainer: Trainer object for model training
        training_in: Training configuration containing dataset info
        dataset_dir: Base directory for datasets

    Raises:
        ValueError: If local dataset paths don't exist or have invalid structure
    """
    if training_in.dataset.storage_location == StorageLocation.LOCAL:
        # Handle local dataset
        logger.info(f"Using local test dataset: {training_in.dataset.test_path}")
        test_dataset_path = Path(training_in.dataset.test_path)

        if not test_dataset_path.exists():
            raise ValueError(f"Test dataset not found at path: {test_dataset_path}")

        # Verify required directory structure
        images_test_path = test_dataset_path / "images" / "test"
        if not images_test_path.exists():
            raise ValueError(
                f"Invalid dataset structure. Expected 'images/test' directory in {test_dataset_path}. "
                "Please ensure the dataset follows the required structure."
            )

        trainer.set_test_dataset(
            str(test_dataset_path),
            test_dataset_path.name
        )

    else:  # StorageLocation.STORAGE
        # Handle storage dataset
        logger.info(f"Downloading test dataset: {training_in.dataset.test_path}")

        test_dataset_path = trainer.download_dataset_for_evaluation(
            dataset_uuid=training_in.dataset.test_path,
            output_dir=dataset_dir.as_posix()
        )

        test_dataset_version = trainer.get_dataset_version_from_storage(
            dataset_uuid=training_in.dataset.test_path,
            split=Split.TEST
        )

        test_dataset_info = trainer.get_dataset_info_from_storage(
            project_id=test_dataset_version.project_id,
            dataset_uuid=training_in.dataset.test_path,
            split=Split.TEST
        )

        trainer.set_test_dataset(
            test_dataset_path,
            test_dataset_info.dataset.dataset_title
        )


def get_model_paths(training_task_id: str) -> Optional[Tuple[Path, Path]]:
    """Get paths for model files and directories.

    Returns:
        Tuple containing (input_model_path, output_dir) or None if unsuccessful
    """
    with get_db_session() as db:
        training_task = training_task_repository.get_by_task_id(db=db, task_id=training_task_id)
        model_info = model_repository.get_by_model_id(db=db, model_id=training_task.model_id)

        input_model_dir = Path(model_info.object_path)
        input_model_path = input_model_dir / "model.onnx"
        output_dir = input_model_dir / "converted"

        logger.info(f"Input model path: {input_model_path}")
        logger.info(f"Output directory: {output_dir}")

        # If model file doesn't exist locally, download from storage
        if not input_model_path.exists():
            logger.info(f"Model file not found locally at {input_model_path}, trying to download from storage")

            # Create directory if it doesn't exist
            input_model_dir.mkdir(parents=True, exist_ok=True)

            # Set object path (storage path can be extracted from model_info.object_path)
            try:
                # Download file from storage
                storage_handler.download_file_from_s3(
                    bucket_name=settings.MODEL_BUCKET_NAME,
                    object_path=str(input_model_path),
                    local_path=str(input_model_path)
                )
                logger.info(f"Successfully downloaded model file from storage to {input_model_path}")

                # Check if file exists after download
                if not input_model_path.exists():
                    logger.error(f"Failed to download model file: {input_model_path} still not found")
                    return None

            except Exception as e:
                logger.error(f"Error downloading model file: {str(e)}")
                return None

        return input_model_path, output_dir


def trigger_conversion_evaluation(
    api_key: str,
    input_model_path: Path,
    output_dir: Path,
    model_id: str,
    training_in: TrainingCreate,
    training_task_id: str
):
    """Trigger the conversion and evaluation chain."""
    from src.worker.evaluation_task import chain_conversion_and_evaluation, run_multiple_evaluations

    conversion_option = training_in.conversion

    # For ONNX models, skip conversion and run evaluation directly
    if conversion_option.framework == EvaluationTargetFramework.ONNX:
        logger.info("ONNX model detected - skipping conversion and running evaluation directly")

        # Evaluate ONNX model directly
        _ = run_multiple_evaluations.apply_async(
            kwargs={
                "api_key": api_key,
                "model_id": model_id,  # Use the already trained ONNX model ID
                "dataset_id": training_in.dataset.test_path,
                "training_task_id": training_task_id,
                "confidence_scores": DEFAULT_CONFIDENCE_SCORES,
            }
        )
    else:
        # Original logic: conversion then evaluation
        _ = chain_conversion_and_evaluation.apply_async(
            kwargs={
                "api_key": api_key,
                "input_model_path": input_model_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "target_framework": conversion_option.framework,
                "target_device_name": conversion_option.device_name,
                "target_data_type": conversion_option.precision,
                "target_software_version": conversion_option.software_version,
                "input_layer": None,
                "dataset_path": None,
                "input_model_id": model_id,
                "dataset_id": training_in.dataset.test_path,
                "training_task_id": training_task_id,
                "confidence_scores": DEFAULT_CONFIDENCE_SCORES,
            }
        )

    logger.info("Successfully initiated conversion and evaluation process")


@celery_app.task(bind=True, name='train_model')
def train_model(
    self,
    training_task_id: str,
    api_key: str,
    training_in: Dict[str, Any],
    unique_model_name: str,
    dataset_path: str,
    training_type: str = "training",
    input_model_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Main training task that orchestrates the training, conversion, and evaluation workflow."""

    with get_db_session() as db:
        training_in: TrainingCreate = TrainingCreate.model_validate(training_in)
        logger.info(f"Starting training task: {training_task_id} for model: {unique_model_name}")

        # Set environment variables
        os.environ['CUDA_VISIBLE_DEVICES'] = training_in.environment.gpus

        # Initialize NetsPresso
        trainer = Trainer(task=training_in.task)

        training_task = training_task_repository.get_by_task_id(db=db, task_id=training_task_id)

        # Configure model and training parameters
        logger.info(f"Configuring model and training parameters for task_id: {training_task_id}")
        configure_model_and_training(trainer, training_in, input_model_info)

        trainer.set_dataset(
            dataset_root_path=training_task.dataset.path,
            dataset_name=training_task.dataset.name,
        )

        # Execute training
        logger.info(f"Starting training with task_id: {training_task_id}")
        training_task = trainer.train(
            db=db,
            gpus=training_in.environment.gpus,
            training_task_id=training_task_id,
        )
        logger.info(f"Training completed with task_id: {training_task_id}")

        # Verify training result
        if training_task.status != TaskStatus.COMPLETED:
            logger.warning(f"Training task {training_task_id} did not complete successfully. Status: {training_task.status}")
            return {"task_id": training_task_id, "status": training_task.status}

        result = {"task_id": training_task_id, "status": "completed"}

        # Process conversion and evaluation if training was successful
        if training_in.dataset.test_path and training_in.conversion:
            try:
                logger.info("Starting post-training chain for conversion and evaluation")

                # Prepare evaluation data
                prepare_evaluation_data(trainer, training_in, dataset_path)

                # Get model paths
                paths = get_model_paths(training_task_id)
                if not paths:
                    return result
                input_model_path, output_dir = paths
                model_id = training_task.model_id

                # Trigger conversion and evaluation
                trigger_conversion_evaluation(
                    api_key=api_key,
                    input_model_path=input_model_path,
                    output_dir=output_dir,
                    model_id=model_id,
                    training_in=training_in,
                    training_task_id=training_task_id
                )

            except Exception as chain_error:
                logger.error(f"Error in conversion-evaluation chain: {str(chain_error)}")

        return result
