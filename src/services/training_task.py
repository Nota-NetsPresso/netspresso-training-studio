import copy
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from src.api.v1.schemas.tasks.dataset import LocalTrainingDatasetPayload
from src.api.v1.schemas.tasks.hyperparameter import (
    OptimizerPayload,
    SchedulerPayload,
    TrainerModel,
)
from src.api.v1.schemas.tasks.training_task import (
    FrameworkPayload,
    InputShape,
    PretrainedModelPayload,
    TaskPayload,
    TrainingCreate,
    TrainingCreatePayload,
    TrainingPayload,
)
from src.configs.settings import settings
from src.enums.model import ModelType
from src.enums.task import TaskStatus
from src.enums.training import MODEL_DISPLAY_MAP, MODEL_GROUP_MAP, StorageLocation, TrainingType
from src.models.base import generate_uuid
from src.models.model import Model
from src.models.training import Augmentation, Dataset, Environment, Hyperparameter, TrainingTask
from src.modules.trainer.augmentations.augmentation import Normalize, Pad, Resize, ToTensor, Transform
from src.modules.trainer.models import get_all_available_models
from src.modules.trainer.optimizers.optimizer_manager import OptimizerManager
from src.modules.trainer.optimizers.optimizers import get_supported_optimizers
from src.modules.trainer.schedulers.scheduler_manager import SchedulerManager
from src.modules.trainer.schedulers.schedulers import get_supported_schedulers
from src.repositories.compression import compression_task_repository
from src.repositories.model import model_repository
from src.repositories.training import training_task_repository
from src.services.user import user_service
from src.worker.training_task import train_model

DEFAULT_AUGMENTATIONS: List[Transform] = [Resize(), Pad(fill=114), ToTensor(), Normalize()]


class TrainingTaskService:
    def get_supported_models(self) -> Dict[str, List[TrainerModel]]:
        """Get all supported models grouped by task."""
        available_models = get_all_available_models()
        supported_models = {
            task: [
                TrainerModel(
                    name=model,
                    display_name=MODEL_DISPLAY_MAP.get(model),
                    group_name=MODEL_GROUP_MAP.get(model),
                )
                for model in models
            ]
            for task, models in available_models.items()
        }

        return supported_models

    def get_supported_optimizers(self) -> List[OptimizerPayload]:
        """Get all supported optimizers."""
        supported_optimizers = get_supported_optimizers()
        optimizers = [OptimizerPayload(name=optimizer.get("name")) for optimizer in supported_optimizers]
        return optimizers

    def get_supported_schedulers(self) -> List[SchedulerPayload]:
        """Get all supported schedulers."""
        supported_schedulers = get_supported_schedulers()
        schedulers = [SchedulerPayload(name=scheduler.get("name")) for scheduler in supported_schedulers]
        return schedulers

    def _convert_to_payload_format(self, training_task: TrainingTask) -> TrainingPayload:
        """Convert training task to payload format."""
        # Perform deep copy to avoid modifying the original data
        task_data = copy.deepcopy(training_task.__dict__)

        # Convert required fields to Payload objects
        task_data['task'] = TaskPayload(name=training_task.task)
        task_data['framework'] = FrameworkPayload(name=training_task.framework)
        task_data['pretrained_model'] = PretrainedModelPayload(name=training_task.pretrained_model)

        # Convert hyperparameter information
        hyperparameter_data = copy.deepcopy(training_task.hyperparameter.__dict__)
        hyperparameter_data['learning_rate'] = training_task.hyperparameter.optimizer["lr"]
        hyperparameter_data['optimizer'] = OptimizerPayload(name=training_task.hyperparameter.optimizer["name"])
        hyperparameter_data['scheduler'] = SchedulerPayload(name=training_task.hyperparameter.scheduler["name"])
        task_data['hyperparameter'] = hyperparameter_data

        # Set model ID
        task_data['model_id'] = training_task.model.model_id if training_task.model else None

        # Remove SQLAlchemy internal state attributes
        if '_sa_instance_state' in task_data:
            del task_data['_sa_instance_state']

        # Convert to Pydantic model and return
        return TrainingPayload.model_validate(task_data)

    def _create_training_dataset(self, training_in: TrainingCreate) -> Dataset:
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

            dataset_root_path = str(train_dataset_path)
            dataset_name = train_dataset_path.name
            dataset_id = None

        else:  # StorageLocation.STORAGE
            # dataset_dir = Path(settings.NP_TRAINING_STUDIO_PATH) / "datasets" / "storage"
            # dataset_dir.mkdir(parents=True, exist_ok=True)

            # train_dataset_path = trainer.download_dataset_for_training(
            #     dataset_uuid=dataset_info.train_path,
            #     output_dir=dataset_dir.as_posix()
            # )

            # train_dataset_version = trainer.get_dataset_version_from_storage(
            #     dataset_uuid=dataset_info.train_path,
            #     split=Split.TRAIN
            # )

            # train_dataset_info = trainer.get_dataset_info_from_storage(
            #     project_id=train_dataset_version.project_id,
            #     dataset_uuid=dataset_info.train_path,
            #     split=Split.TRAIN
            # )

            # dataset_root_path = train_dataset_path
            # dataset_name = train_dataset_info.dataset.dataset_title
            # dataset_id = Path(dataset_root_path).name
            pass

        train_dataset = Dataset(
            name=dataset_name,
            path=dataset_root_path,
            task_type=training_in.task,
            storage_location=dataset_info.storage_location,
            storage_info={"dataset_id": dataset_id}
        )

        return train_dataset

    def _generate_unique_model_name(self, db: Session, project_id: str, name: str) -> str:
        """Generate a unique model name by adding numbering if necessary.

        Args:
            db (Session): Database session
            project_id (str): Project ID to check existing models
            name (str): Original model name

        Returns:
            str: Unique model name with numbering if needed
        """
        # Get existing model names from the database for the same project
        models = model_repository.get_all_by_project_id(
            db=db,
            project_id=project_id,
        )

        # Extract existing names from models and count occurrences of base name
        base_name_count = sum(1 for model in models if model.type == ModelType.TRAINED_MODEL and model.name.startswith(name))

        # If no models with this name exist, return original name
        if base_name_count == 0:
            return name

        # If models exist, return name with count
        return f"{name} ({base_name_count})"

    def create_trained_model(self, db: Session, model_name: str, user_id: str, project_id: str) -> Model:
        model_id = generate_uuid(entity="model")
        base_object_path = f"{user_id}/{project_id}/{model_id}"
        model_name = self._generate_unique_model_name(
            db=db,
            project_id=project_id,
            name=model_name,
        )
        model_obj = Model(
            model_id=model_id,
            name=model_name,
            type=ModelType.TRAINED_MODEL,
            is_retrainable=True,
            project_id=project_id,
            user_id=user_id,
            object_path=base_object_path  # Store base path only
        )
        model_obj = model_repository.save(db=db, model=model_obj)

        return model_obj

    def create_training_task(self, db: Session, training_in: TrainingCreate, token: str) -> TrainingTask:
        """Create a new training task.

        Args:
            db: Database session
            training_in: Training configuration
            token: User authentication token

        Returns:
            TrainingCreatePayload with the task ID
        """
        user_info = user_service.get_user_info(token=token)

        # Create trained model object
        model_obj = self.create_trained_model(
            db=db,
            model_name=training_in.name,
            user_id=user_info.user_id,
            project_id=training_in.project_id,
        )

        # Create augmentations configuration
        augs = [
            Augmentation(
                name=aug.name,
                parameters=aug.to_parameters(),
                phase=phase,
            )
            for phase, aug_list in [("train", DEFAULT_AUGMENTATIONS), ("inference", DEFAULT_AUGMENTATIONS)]
            for aug in aug_list
        ]

        # Configure optimizer and scheduler
        optimizer = OptimizerManager.get_optimizer(
            name=training_in.hyperparameter.optimizer,
            lr=training_in.hyperparameter.learning_rate,
        )
        scheduler = SchedulerManager.get_scheduler(
            name=training_in.hyperparameter.scheduler
        )

        # Create hyperparameter configuration
        hyperparameter = Hyperparameter(
            epochs=training_in.hyperparameter.epochs,
            batch_size=training_in.hyperparameter.batch_size or 8,
            optimizer=optimizer.asdict(),
            scheduler=scheduler.asdict(),
            augmentations=augs,
        )

        # Create environment configuration with defaults
        environment = Environment(
            seed=training_in.environment.seed,
            num_workers=training_in.environment.num_workers,
            gpus=training_in.environment.gpus
        )

        # Get image size from input shapes
        img_size = training_in.input_shapes[0].dimension[0]

        train_dataset = self._create_training_dataset(training_in=training_in)

        # Determine training type
        training_type = TrainingType.RETRAINING if training_in.input_model_id else TrainingType.TRAINING

        # Create training task
        training_task_id = generate_uuid(entity="task")
        training_task = TrainingTask(
            task_id=training_task_id,
            pretrained_model=training_in.pretrained_model,
            task=training_in.task,
            framework="pytorch",
            input_shapes=[InputShape(batch=1, channel=3, dimension=[img_size, img_size]).__dict__],
            status=TaskStatus.IN_PROGRESS,
            hyperparameter=hyperparameter,
            environment=environment,
            model_id=model_obj.model_id,
            user_id=model_obj.user_id,
            training_type=training_type,
            input_model_id=training_in.input_model_id,
            dataset=train_dataset,
        )
        training_task = training_task_repository.save(db=db, model=training_task)

        return training_task

    def start_training_task(self, db: Session, training_in: TrainingCreate, training_task: TrainingTask, token: str) -> TrainingCreatePayload:
        # Get input model info if retraining
        input_model_info = None
        if training_in.input_model_id:
            input_model_info = self._get_input_model_info(db, training_in.input_model_id)

        # Prepare worker task parameters
        worker_params = {
            "training_task_id": training_task.task_id,
            "api_key": token,
            "training_in": training_in.model_dump(),
            "unique_model_name": training_task.model.name,
            "training_type": training_task.training_type,
            "dataset_path": training_task.dataset.path,
        }

        # Add input model info if retraining
        if input_model_info:
            worker_params["input_model_info"] = input_model_info

        # Start async training task
        _ = train_model.apply_async(
            kwargs=worker_params,
            training_task_id=training_task.task_id,
        )

        return TrainingCreatePayload(task_id=training_task.task_id)

    def get_training_task(self, db: Session, task_id: str, token: str) -> TrainingPayload:
        """Get training task by task ID."""
        training_task = training_task_repository.get_by_task_id(db=db, task_id=task_id)

        return self._convert_to_payload_format(training_task)

    def delete_training_task_by_model_id(self, db: Session, model_id: str) -> TrainingPayload:
        """Delete training task by model ID."""
        training_task = training_task_repository.get_by_model_id(db=db, model_id=model_id)
        training_task_repository.soft_delete(db=db, model=training_task)

        return self._convert_to_payload_format(training_task)

    def delete_training_task_by_task_id(self, db: Session, task_id: str) -> TrainingPayload:
        """Delete training task by task ID."""
        training_task = training_task_repository.get_by_task_id(db=db, task_id=task_id)
        training_task_repository.soft_delete(db=db, model=training_task)

        return self._convert_to_payload_format(training_task)

    def get_training_datasets_from_local(self) -> List[LocalTrainingDatasetPayload]:
        """Get training datasets from local directory.

        Returns:
            LocalTrainingDatasetsResponse: List of dataset information including name and path
        """
        training_datasets_dir = Path(settings.NP_TRAINING_STUDIO_PATH) / "datasets" / "local"
        training_datasets = [d for d in training_datasets_dir.iterdir() if d.is_dir()]

        training_datasets_payload = [
            LocalTrainingDatasetPayload(
                name=dataset.name,
                path=str(dataset.absolute()),
            )
            for dataset in training_datasets
        ]

        return training_datasets_payload

    def get_training_task_by_model_id(self, db: Session, model_id: str) -> TrainingPayload:
        """Get training task by model ID."""
        training_task = training_task_repository.get_by_model_id(db=db, model_id=model_id)

        return self._convert_to_payload_format(training_task)

    def _get_input_model_info(self, db: Session, input_model_id: str) -> Dict[str, Any]:
        """Get input model information for retraining.

        Args:
            db: Database session
            input_model_id: ID of the input model

        Returns:
            Dict containing model information needed for retraining
        """
        input_model = model_repository.get_by_model_id(db=db, model_id=input_model_id)

        if input_model.type == ModelType.TRAINED_MODEL:
            training_task = training_task_repository.get_by_model_id(db=db, model_id=input_model_id)
            pretrained_model = training_task.pretrained_model
        else:
            compression_task = compression_task_repository.get_by_model_id(db=db, model_id=input_model_id)
            training_task = training_task_repository.get_by_model_id(db=db, model_id=compression_task.input_model_id)
            pretrained_model = training_task.pretrained_model

        # Construct model paths
        if input_model.type == ModelType.TRAINED_MODEL:
            remote_model_path = f"{input_model.object_path}/model.pt"
        else:
            remote_model_path = input_model.object_path

        return {
            "pretrained_model": pretrained_model,
            "remote_model_path": remote_model_path,
            "model_type": input_model.type
        }


training_task_service = TrainingTaskService()
