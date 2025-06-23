import os
from pathlib import Path
from typing import Dict, List

from celery import chain, signature
from loguru import logger

from src.api.v1.schemas.tasks.common.dataset import DatasetCreate
from src.api.v1.schemas.tasks.training.environment import EnvironmentCreate
from src.api.v1.schemas.tasks.training.hyperparameter import HyperparameterCreate
from src.api.v1.schemas.tasks.training.training_task import TrainingCreate
from src.core.db.session import SessionLocal, get_db_session
from src.enums.task import TaskStatus
from src.enums.training import StorageLocation
from src.models.base import generate_uuid
from src.modules.evaluator.evaluator import Evaluator
from src.modules.trainer.augmentations.augmentation import Normalize, Pad, Resize, ToTensor
from src.modules.trainer.optimizers.optimizer_manager import OptimizerManager
from src.modules.trainer.schedulers.scheduler_manager import SchedulerManager
from src.modules.trainer.storage.dataforge import Split
from src.modules.trainer.trainer import Trainer
from src.repositories.conversion import conversion_task_repository
from src.repositories.evaluation import evaluation_dataset_repository, evaluation_task_repository
from src.worker.celery_app import celery_app

POLLING_INTERVAL = 30  # seconds
NP_TRAINING_STUDIO_PATH = os.environ.get("NP_TRAINING_STUDIO_PATH", "/np_training_studio")


@celery_app.task(bind=True, name='evaluate_model_task')
def evaluate_model_task(
    self,
    api_key: str,
    model_id: str,
    dataset_id: str,
    training_task_id: str,
    evaluation_task_id: str,
    confidence_score: float,
    gpus: int = 0,
):
    """Celery task to perform evaluation with a specific confidence score

    Args:
        api_key: API key for authentication
        model_id: ID of the model to evaluate
        dataset_id: ID of the dataset to use for evaluation (or local path if using local dataset)
        training_task_id: ID of the related training task
        evaluation_task_id: Evaluation task ID
        confidence_score: Confidence score for evaluation (one of 0.3, 0.5, 0.6)
        gpus: Number of GPUs to use

    Returns:
        result_id: Generated evaluation result ID
    """
    session = SessionLocal()
    try:
        from src.services.training_task import training_task_service

        training_task = training_task_service.get_training_task(
            db=session,
            task_id=training_task_id,
            token=api_key,
        )

        # Get trainer instance from the training task
        trainer = Trainer(task=training_task.task.name)

        logger.info(f"Using pretrained model: {training_task.pretrained_model.name}")

        training_in = TrainingCreate(
            pretrained_model=training_task.pretrained_model.name,
            task=training_task.task.name,
            input_shapes=training_task.input_shapes,
            dataset=DatasetCreate(
                train_path=training_task.dataset.path,
                test_path=training_task.dataset.path,
                storage_location=training_task.dataset.storage_location,
            ),
            hyperparameter=HyperparameterCreate(
                epochs=training_task.hyperparameter.epochs,
                batch_size=training_task.hyperparameter.batch_size,
                learning_rate=training_task.hyperparameter.learning_rate,
                optimizer=training_task.hyperparameter.optimizer.name,
                scheduler=training_task.hyperparameter.scheduler.name,
            ),
            environment=EnvironmentCreate(
                gpus=training_task.environment.gpus,
            ),
            project_id="",
            name="",
        )

        if training_in.dataset.storage_location == StorageLocation.LOCAL:
            # Handle local dataset
            logger.info(f"Using local dataset path: {dataset_id}")
            test_dataset_path = Path(dataset_id)
            if not test_dataset_path.exists():
                raise ValueError(f"Local test dataset not found at path: {test_dataset_path}")

            # Verify required directory structure
            images_test_path = test_dataset_path / "images" / "test"
            if not images_test_path.exists():
                raise ValueError(
                    f"Invalid dataset structure. Expected 'images/test' directory in {test_dataset_path}. "
                    "Please ensure the dataset follows the required structure."
                )

            evaluation_dataset = evaluation_dataset_repository.get_by_dataset_path(db=session, dataset_path=test_dataset_path)
            logger.info(f"evaluation_dataset: {evaluation_dataset}")
            if evaluation_dataset:
                trainer.set_test_dataset_no_create(test_dataset_path, evaluation_dataset.name)
                trainer.test_dataset_id = evaluation_dataset.dataset_id
            else:
                trainer.set_test_dataset(str(test_dataset_path), test_dataset_path.name)

        else:  # StorageLocation.STORAGE
            # Handle storage dataset
            dataset_dir = os.path.join(NP_TRAINING_STUDIO_PATH, "datasets", "storage")
            os.makedirs(dataset_dir, exist_ok=True)

            logger.info(f"Downloading dataset from DataForge: {dataset_id}")

            existing_dataset = evaluation_dataset_repository.get_by_dataforge_dataset_id(db=session, dataset_id=dataset_id)
            logger.info(f"Existing dataset: {existing_dataset}")
            if existing_dataset:
                logger.info(f"Found existing evaluation dataset for dataforge dataset {dataset_id}")
                test_dataset_path = existing_dataset.path
                trainer.set_test_dataset_no_create(test_dataset_path, existing_dataset.name)
                trainer.test_dataset_id = existing_dataset.dataset_id
            else:
                test_dataset_path = trainer.download_dataset_for_evaluation(dataset_uuid=dataset_id, output_dir=dataset_dir)
                test_dataset_version = trainer.get_dataset_version_from_storage(dataset_uuid=dataset_id, split=Split.TEST)
                test_dataset_info = trainer.get_dataset_info_from_storage(project_id=test_dataset_version.project_id, dataset_uuid=dataset_id, split=Split.TEST)
                trainer.set_test_dataset(test_dataset_path, test_dataset_info.dataset.dataset_title)

        logger.info(f"Using dataset path: {test_dataset_path}")

        img_size = training_in.input_shapes[0].dimension[0]
        trainer.set_model_config(model_name=training_in.pretrained_model, img_size=img_size)
        trainer.set_augmentation_config(
            train_transforms=[Resize(), Pad(fill=114), ToTensor(), Normalize()],
            inference_transforms=[Resize(), Pad(fill=114), ToTensor(), Normalize()],
        )
        optimizer = OptimizerManager.get_optimizer(
            name=training_in.hyperparameter.optimizer,
            lr=training_in.hyperparameter.learning_rate,
        )
        scheduler = SchedulerManager.get_scheduler(name=training_in.hyperparameter.scheduler)
        trainer.set_training_config(
            epochs=training_in.hyperparameter.epochs,
            batch_size=training_in.hyperparameter.batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        trainer._apply_img_size()

        # Create evaluator
        evaluator = Evaluator(trainer=trainer)

        # Perform actual evaluation
        try:
            task_id = evaluator.evaluate_from_id(
                model_id=model_id,
                confidence_score=confidence_score,
                gpus=gpus,
                evaluation_task_id=evaluation_task_id,
            )
            result = {"task_id": task_id, "status": "completed"}
            return result
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise e

    except Exception as e:
        logger.error(f"Evaluation task error: {str(e)}")
        raise e
    finally:
        session.close()


@celery_app.task(bind=True, name='run_multiple_evaluations')
def run_multiple_evaluations(
    self,
    api_key: str,
    evaluation_task_ids: List[str],
) -> str:
    """Run evaluation for a model with multiple confidence scores."""
    with get_db_session() as db:
        for task_id in evaluation_task_ids:
            evaluation_task = evaluation_task_repository.get_by_task_id(db=db, task_id=task_id)
            if not evaluation_task:
                logger.error(f"Evaluation task with ID {task_id} not found. Skipping.")
                continue

            try:
                evaluation_task.status = TaskStatus.IN_PROGRESS
                evaluation_task_repository.update(db=db, model=evaluation_task)

                evaluator = Evaluator()
                evaluator.evaluate_from_id(
                    db=db,
                    model_id=evaluation_task.input_model_id,
                    evaluation_task_id=evaluation_task.task_id,
                    confidence_score=evaluation_task.confidence_score,
                )
            except Exception as e:
                logger.error(f"Error during evaluation for task {task_id}: {str(e)}")
                evaluation_task.status = TaskStatus.ERROR
                evaluation_task.error_detail = str(e)
                evaluation_task_repository.update(db=db, model=evaluation_task)

        return evaluation_task_ids[0] if evaluation_task_ids else None


@celery_app.task(bind=True, name='poll_and_start_evaluation')
def poll_and_start_evaluation(
    self,
    api_key: str,
    conversion_task_id: str,
    evaluation_task_ids: List[str],
):
    """Poll conversion task status and start evaluation when complete."""
    with get_db_session() as db:
        try:
            conversion_task = conversion_task_repository.get_by_task_id(db=db, task_id=conversion_task_id)

            if conversion_task.status == TaskStatus.COMPLETED:
                logger.info(f"Conversion completed for task {conversion_task_id}. Starting evaluation tasks.")
                run_multiple_evaluations.apply_async(
                    kwargs={"api_key": api_key, "evaluation_task_ids": evaluation_task_ids}
                )
                return {"status": "EvaluationTriggered", "evaluation_task_ids": evaluation_task_ids}

            elif conversion_task.status in [TaskStatus.STOPPED, TaskStatus.ERROR]:
                error_message = f"Conversion failed with status {conversion_task.status}: {conversion_task.error_detail}"
                logger.error(error_message)
                # Fail all linked evaluation tasks
                for task_id in evaluation_task_ids:
                    eval_task = evaluation_task_repository.get_by_task_id(db=db, task_id=task_id)
                    if eval_task:
                        eval_task.status = TaskStatus.ERROR
                        eval_task.error_detail = error_message
                        evaluation_task_repository.update(db=db, model=eval_task)
                raise Exception(error_message)
            else:
                logger.info(f"Conversion in progress for {conversion_task_id}. Retrying poll in {POLLING_INTERVAL}s.")
                self.retry(countdown=POLLING_INTERVAL, max_retries=None)

        except Exception as e:
            logger.error(f"Polling task failed for conversion {conversion_task_id}: {str(e)}")
            raise


@celery_app.task(bind=True, name='chain_conversion_and_evaluation')
def chain_conversion_and_evaluation(
    self,
    api_key: str,
    conversion_task_id: str,
    evaluation_task_ids: List[str],
):
    """Chain conversion and evaluation tasks using Celery's chain."""
    logger.info(f"Chaining conversion ({conversion_task_id}) with evaluations ({evaluation_task_ids}).")

    # The conversion task is already created and started by the service.
    # We just need to poll for its completion.
    poll_task = signature(
        'poll_and_start_evaluation',
        kwargs={
            "api_key": api_key,
            "conversion_task_id": conversion_task_id,
            "evaluation_task_ids": evaluation_task_ids,
        }
    )
    poll_task.apply_async()

    return {"chain_started": True, "conversion_task_id": conversion_task_id}
