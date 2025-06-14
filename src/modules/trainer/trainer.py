import tempfile
from dataclasses import asdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from src.configs.settings import settings
from src.core.db.session import get_db_session
from src.enums.task import TaskStatus
from src.enums.training import StorageLocation, Task
from src.exceptions.training import (
    BaseDirectoryNotFoundException,
    DirectoryNotFoundException,
    FailedTrainingException,
    FileNotFoundErrorException,
    NotSetDatasetException,
    NotSetModelException,
    NotSupportedModelException,
    NotSupportedTaskException,
    RetrainingFunctionException,
)
from src.models.evaluation import EvaluationDataset
from src.models.model import Model
from src.models.training import (
    Dataset,
    Performance,
    TrainingTask,
)
from src.modules.inferencer.preprocessors.base import Preprocessor
from src.modules.trainer.augmentations import AUGMENTATION_CONFIG_TYPE, AugmentationConfig, Transform
from src.modules.trainer.data import DATA_CONFIG_TYPE, ImageLabelPathConfig, PathConfig
from src.modules.trainer.models import (
    CLASSIFICATION_MODELS,
    DETECTION_MODELS,
    SEGMENTATION_MODELS,
    CheckpointConfig,
)
from src.modules.trainer.optimizers.optimizers import get_supported_optimizers
from src.modules.trainer.schedulers.schedulers import get_supported_schedulers
from src.modules.trainer.trainer_configs import TrainerConfigs
from src.modules.trainer.training import TRAINING_CONFIG_TYPE, EnvironmentConfig, LoggingConfig, ScheduleConfig
from src.modules.trainer.training.logging import Metrics, ModelSaveOptions
from src.repositories.evaluation import evaluation_dataset_repository
from src.repositories.training import training_task_repository
from src.utils.file import FileHandler
from src.zenko.storage_handler import ObjectStorageHandler

storage_handler = ObjectStorageHandler()


class Trainer:
    """
    NetsPresso Trainer Class: Base class for training models.
    """

    def __init__(self, task: Task) -> None:
        """Initialize the Trainer.

        Args:
            task (Task): The type of task (classification, detection, segmentation).
        """

        self._initialize_from_task(task)

        self.is_dataforge = False
        self.test_dataset = None
        self.test_dataset_id = None

    def _initialize_from_task(self, task: Union[str, Task]) -> None:
        """Initialize the Trainer object based on the provided task.

        Args:
            task (Union[str, Task]): The task for which the Trainer is initialized.
        """

        self.task = self._validate_task(task)
        self.available_models = list(self._get_available_models().keys())
        self.data = None
        self.model = None
        self.training = TRAINING_CONFIG_TYPE[self.task]()
        self.augmentation = AUGMENTATION_CONFIG_TYPE[self.task]()
        self.logging = LoggingConfig()
        self.environment = EnvironmentConfig()

    def _validate_task(self, task: Union[str, Task]):
        """Validate the provided task.

        Args:
            task (Union[str, Task]): The task to be validated.

        Raises:
            ValueError: If the provided task is not supported.

        Returns:
            Task: The validated task.
        """

        available_tasks = [task.value for task in Task]
        if task not in available_tasks:
            raise NotSupportedTaskException(available_tasks, task)
        return task

    def _validate_config(self):
        """Validate the configuration setup.

        Raises:
            ValueError: Raised if the dataset is not set. Use `set_dataset_config` or `set_dataset_config_with_yaml` to set the dataset configuration.
            ValueError: Raised if the model is not set. Use `set_model_config` or `set_model_config_with_yaml` to set the model configuration.
        """

        if self.data is None:
            raise NotSetDatasetException()
        if self.model is None:
            raise NotSetModelException()

    def _get_available_models(self) -> Dict[str, Any]:
        """Get available models based on the current task.

        Returns:
            Dict[str, Any]: A dictionary mapping model types to available models.
        """

        available_models = {
            "classification": CLASSIFICATION_MODELS,
            "detection": DETECTION_MODELS,
            "segmentation": SEGMENTATION_MODELS,
        }[self.task]

        return available_models

    def set_dataset_config(
        self,
        name: str,
        root_path: str,
        train_image: str = "images/train",
        train_label: str = "labels/train",
        valid_image: str = "images/valid",
        valid_label: str = "labels/valid",
        test_image: str = "images/valid",
        test_label: str = "labels/valid",
        id_mapping: Optional[Union[List[str], Dict[str, str], str]] = None,
    ):
        """Set the dataset configuration for the Trainer.

        Args:
            name (str): The name of dataset.
            root_path (str): Root directory of dataset.
            train_image (str, optional): The directory for training images. Should be relative path to root directory. Defaults to "images/train".
            train_label (str, optional): The directory for training labels. Should be relative path to root directory. Defaults to "labels/train".
            valid_image (str, optional): The directory for validation images. Should be relative path to root directory. Defaults to "images/val".
            valid_label (str, optional): The directory for validation labels. Should be relative path to root directory. Defaults to "labels/val".
            id_mapping (Union[List[str], Dict[str, str]], optional): ID mapping for the dataset. Defaults to None.
        """

        common_config = {
            "name": name,
            "path": PathConfig(
                root=root_path,
                train=ImageLabelPathConfig(image=train_image, label=train_label),
                valid=ImageLabelPathConfig(image=valid_image, label=valid_label),
                test=ImageLabelPathConfig(image=test_image, label=test_label),
            ),
            "id_mapping": id_mapping,
        }
        self.data = DATA_CONFIG_TYPE[self.task](**common_config)

    def check_paths_exist(self, base_path):
        paths = [
            "images/train",
            "images/valid",
            "id_mapping.json",
        ]

        # Check for the existence of required directories and files
        logger.info(f"Checking paths exist: {base_path}")
        for relative_path in paths:
            path = Path(base_path) / relative_path
            if not path.exists():
                if path.suffix:  # It's a file
                    raise FileNotFoundErrorException(path.as_posix())
                else:  # It's a directory
                    raise DirectoryNotFoundException(path.as_posix())

    def check_test_paths_exist(self, base_path):
        paths = [
            "images/test",
            "id_mapping.json",
        ]

        # Check for the existence of required directories and files
        for relative_path in paths:
            path = Path(base_path) / relative_path
            if not path.exists():
                if path.suffix:  # It's a file
                    raise FileNotFoundErrorException(relative_path)
                else:  # It's a directory
                    raise DirectoryNotFoundException(relative_path)

    def find_paths(self, base_path: str, search_dir, split: str) -> List[str]:
        base_dir = Path(base_path)

        if not base_dir.exists():
            raise BaseDirectoryNotFoundException(base_dir)

        result_paths = []

        dir_path = base_dir / search_dir
        if dir_path.exists() and dir_path.is_dir():
            for item in dir_path.iterdir():
                if (item.is_dir() or item.is_file()) and split in item.name:
                    result_paths.append(item.as_posix())

        return result_paths[0]

    def set_dataset(self, dataset_root_path: str, dataset_name: Optional[str] = None):
        if dataset_name is None:
            dataset_name = Path(dataset_root_path).name
        root_path = Path(dataset_root_path).resolve().as_posix()

        self.check_paths_exist(root_path)
        images_train = self.find_paths(root_path, "images", "train")
        images_valid = self.find_paths(root_path, "images", "valid")
        labels_train = self.find_paths(root_path, "labels", "train")
        labels_valid = self.find_paths(root_path, "labels", "valid")
        id_mapping = FileHandler.load_json(f"{root_path}/id_mapping.json")
        self.set_dataset_config(
            name=dataset_name,
            root_path=dataset_root_path,
            train_image=images_train,
            train_label=labels_train,
            valid_image=images_valid,
            valid_label=labels_valid,
            id_mapping=id_mapping,
        )

        train_image_path = Path(images_train)
        valid_image_path = Path(images_valid)
        train_image_count = len(list(train_image_path.glob("*.*"))) if train_image_path.is_dir() else 1
        valid_image_count = len(list(valid_image_path.glob("*.*"))) if valid_image_path.is_dir() else 1
        total_image_count = train_image_count + valid_image_count

        self.train_dataset = Dataset(
            name=dataset_name,
            path=root_path,
            id_mapping=self.data.id_mapping,
            palette=self.data.pallete,
            task_type=self.task,
            class_count=len(self.data.id_mapping),
            count=total_image_count,
            storage_location=StorageLocation.STORAGE if self.is_dataforge else StorageLocation.LOCAL,
            storage_info={"dataset_id": Path(root_path).name}
        )

    def _save_evaluation_dataset(self, evaluation_dataset):
        with get_db_session() as db:
            evaluation_dataset = evaluation_dataset_repository.save(db=db, model=evaluation_dataset)

            return evaluation_dataset

    def set_test_dataset(self, dataset_root_path: str, dataset_name: Optional[str] = None):
        if dataset_name is None:
            dataset_name = Path(dataset_root_path).name
        root_path = Path(dataset_root_path).resolve().as_posix()

        # self.check_test_paths_exist(root_path)
        images_test = self.find_paths(root_path, "images", "test")
        labels_test = self.find_paths(root_path, "labels", "test")
        id_mapping = FileHandler.load_json(f"{root_path}/id_mapping.json")

        if self.data is not None:
            self.data.path.test.image = images_test
            self.data.path.test.label = labels_test
        else:
            self.set_dataset_config(
                name=dataset_name,
                root_path=dataset_root_path,
                test_image=images_test,
                test_label=labels_test,
                id_mapping=id_mapping,
            )

        test_image_path = Path(images_test)
        test_image_count = len(list(test_image_path.glob("*.*"))) if test_image_path.is_dir() else 1

        storage_location = StorageLocation.STORAGE if self.is_dataforge else StorageLocation.LOCAL
        storage_info = {"dataset_id": Path(root_path).name} if self.is_dataforge else None

        self.test_dataset = EvaluationDataset(
            name=dataset_name,
            path=root_path,
            id_mapping=self.data.id_mapping,
            palette=self.data.pallete,
            task_type=self.task,
            class_count=len(self.data.id_mapping),
            count=test_image_count,
            storage_location=storage_location,
            storage_info=storage_info
        )
        self.test_dataset = self._save_evaluation_dataset(self.test_dataset)

    def set_test_dataset_no_create(self, dataset_root_path: str, dataset_name: Optional[str] = None):
        if dataset_name is None:
            dataset_name = Path(dataset_root_path).name
        root_path = Path(dataset_root_path).resolve().as_posix()

        # self.check_test_paths_exist(root_path)
        images_test = self.find_paths(root_path, "images", "test")
        labels_test = self.find_paths(root_path, "labels", "test")
        id_mapping = FileHandler.load_json(f"{root_path}/id_mapping.json")

        if self.data is not None:
            self.data.path.test.image = images_test
            self.data.path.test.label = labels_test
        else:
            self.set_dataset_config(
                name=dataset_name,
                root_path=dataset_root_path,
                test_image=images_test,
                test_label=labels_test,
                id_mapping=id_mapping,
            )

    def set_model_config(
        self,
        model_name: str,
        img_size: int,
        use_pretrained: bool = True,
        load_head: bool = False,
        path: Optional[str] = None,
        fx_model_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
    ):
        """Set the model configuration for the Trainer.

        Args:
            model_name (str): Name of the model.
            img_size (int): Image size for the model.
            use_pretrained (bool, optional): Whether to use a pre-trained model. Defaults to True.
            load_head (bool, optional): Whether to load the model head. Defaults to False.
            path (str, optional): Path to the model. Defaults to None.
            fx_model_path (str, optional): Path to the FX model. Defaults to None.
            optimizer_path (str, optional): Path to the optimizer. Defaults to None.

        Raises:
            ValueError: If the specified model is not supported for the current task.
        """
        self.model_name = model_name
        model = self._get_available_models().get(model_name)
        self.img_size = img_size
        self.logging.model_save_options.sample_input_size = [img_size, img_size]

        if model is None:
            raise NotSupportedModelException(
                available_models=self._get_available_models(),
                model_name=model_name,
                task=self.task,
            )

        self.model = model(
            checkpoint=CheckpointConfig(
                use_pretrained=use_pretrained,
                load_head=load_head,
                path=path,
                fx_model_path=fx_model_path,
                optimizer_path=optimizer_path,
            )
        )

    def set_fx_model(self, fx_model_path: str):
        """Set the FX model path for retraining.

        Args:
            fx_model_path (str): The path to the FX model.

        Raises:
            ValueError: If the model is not set. Please use 'set_model_config' for model setup.
        """

        if not self.model:
            raise RetrainingFunctionException()

        self.model.checkpoint.path = None
        self.model.checkpoint.fx_model_path = fx_model_path

    def set_training_config(
        self,
        optimizer,
        scheduler,
        epochs: int = 3,
        batch_size: int = 8,
    ):
        """Set the training configuration.

        Args:
            optimizer: The configuration of optimizer.
            scheduler: The configuration of learning rate scheduler.
            epochs (int, optional): The total number of epoch for training the model. Defaults to 3.
            batch_size (int, optional): The number of samples in single batch input. Defaults to 8.
        """

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training = ScheduleConfig(
            epochs=epochs,
            optimizer=optimizer.asdict(),
            scheduler=scheduler.asdict(),
        )
        self.environment.batch_size = batch_size

    def set_augmentation_config(
        self,
        train_transforms: Optional[List] = None,
        inference_transforms: Optional[List] = None,
    ):
        """Set the augmentation configuration for training.

        Args:
            train_transforms (List, optional): List of transforms for training. Defaults to None.
            inference_transforms (List, optional): List of transforms for inference. Defaults to None.
        """

        self.augmentation = AugmentationConfig(
            train=train_transforms,
            inference=inference_transforms,
        )

    def set_logging_config(
        self,
        project_id: Optional[str] = None,
        output_dir: str = "./outputs",
        tensorboard: bool = True,
        csv: bool = False,
        image: bool = True,
        stdout: bool = True,
        save_optimizer_state: bool = True,
        validation_epoch: int = 10,
        save_checkpoint_epoch: Optional[int] = None,
    ):
        """Set the logging configuration.

        Args:
            project_id (str, optional): Project name to save the experiment. If None, it is set as {task}_{model} (e.g. segmentation_segformer).
            output_dir (str, optional): Root directory for saving the experiment. Defaults to "./outputs".
            tensorboard (bool, optional): Whether to use the tensorboard. Defaults to True.
            csv (bool, optional): Whether to save the result in csv format. Defaults to False.
            image (bool, optional): Whether to save the validation results. It is ignored if the task is classification. Defaults to True.
            stdout (bool, optional): Whether to log the standard output. Defaults to True.
            save_optimizer_state (bool, optional): Whether to save optimizer state with model checkpoint to resume training. Defaults to True.
            validation_epoch (int, optional): Validation frequency in total training process. Defaults to 10.
            save_checkpoint_epoch (int, optional): Checkpoint saving frequency in total training process. Defaults to None.
        """
        model_save_options = ModelSaveOptions(
            save_optimizer_state=save_optimizer_state,
            sample_input_size=[self.img_size, self.img_size],
            validation_epoch=validation_epoch,
            save_checkpoint_epoch=save_checkpoint_epoch,
        )
        metrics = Metrics(
            classwise_analysis=False,
            metric_names=None,
        )

        self.logging = LoggingConfig(
            project_id=project_id,
            output_dir=output_dir,
            tensorboard=tensorboard,
            csv=csv,
            image=image,
            stdout=stdout,
            model_save_options=model_save_options,
            metrics=metrics,
        )

    def set_environment_config(self, seed: int = 1, num_workers: int = 4):
        """Set the environment configuration.

        Args:
            seed (int, optional): Random seed. Defaults to 1.
            num_workers (int, optional): The number of multi-processing workers to be used by the data loader. Defaults to 4.
        """

        self.environment = EnvironmentConfig(seed=seed, num_workers=num_workers)

    def _change_transforms(self, transforms: Transform):
        """Update the 'size' attribute in the given list of transforms with the specified image size.

        Args:
            transforms (List[Transform]): The list of transforms to be updated.

        Returns:
            List[Transform]: The list of transforms with the 'size' attribute updated to the specified image size.
        """

        field_name_to_check = "size"

        if transforms is None:
            return transforms

        for transform in transforms:
            field_type = transform.__annotations__.get(field_name_to_check)

            if field_type == List:
                transform.size = [self.img_size, self.img_size]
            elif isinstance(field_type, int):
                transform.size = self.img_size

        return transforms

    def _apply_img_size(self):
        """Apply the specified image size to the augmentation configurations.

        This method updates the 'img_size' attribute in the augmentation configurations, including
        'train.transforms', 'train.mix_transforms', and 'inference.transforms'.
        """

        self.augmentation.img_size = self.img_size
        self.augmentation.train = self._change_transforms(self.augmentation.train)
        self.augmentation.inference = self._change_transforms(self.augmentation.inference)

    def _get_status_by_training_summary(self, status: str) -> TaskStatus:
        status_mapping = {
            "success": TaskStatus.COMPLETED,
            "stop": TaskStatus.STOPPED,
            "error": TaskStatus.ERROR,
            "": TaskStatus.IN_PROGRESS,
        }
        return status_mapping.get(status, TaskStatus.IN_PROGRESS)

    def find_best_model_paths(self, destination_folder: Path):
        best_fx_paths_set = set()

        for pattern in ["*best_fx.pt", "*best.pt"]:
            best_fx_paths_set.update(destination_folder.glob(pattern))

        best_fx_paths = list(best_fx_paths_set)
        best_onnx_paths = list(destination_folder.glob("*best.onnx"))

        return best_fx_paths, best_onnx_paths

    def update_task_status(self, task_id: str, status: TaskStatus, error_message: Optional[str] = None) -> None:
        """
        Update the task status.

        Args:
            task_id: The ID of the task to update.
            status: The new status (Status enum value).
            error_message: The error message (optional).
        """
        logger.info(f"Updating task status: {task_id} -> {status}")
        if error_message:
            logger.error(f"Error message: {error_message}")

        with get_db_session() as session:
            training_task_repository.update_status(
                db=session,
                task_id=task_id,
                status=status,
                error_message=error_message,
            )

    def create_performance(self, training_task: TrainingTask, training_summary: Dict[str, Any]) -> TrainingTask:
        performance = Performance(
            train_losses=training_summary["train_losses"],
            valid_losses=training_summary["valid_losses"],
            train_metrics=training_summary["train_metrics"],
            valid_metrics=training_summary["valid_metrics"],
            metrics_list=training_summary["metrics_list"],
            primary_metric=training_summary["primary_metric"],
            flops=str(training_summary["flops"]),
            params=str(training_summary["params"]),
            total_train_time=training_summary["total_train_time"],
            best_epoch=training_summary["best_epoch"],
            last_epoch=training_summary["last_epoch"],
            total_epoch=training_summary["total_epoch"],
            status=training_summary["status"],
        )
        training_task.performance = performance

        return training_task

    def train(
        self,
        db,
        gpus: str,
        training_task_id: str,
        output_dir: Optional[str] = "./outputs",
    ) -> TrainingTask:
        # Validate configuration and initialize
        self._validate_config()
        self._apply_img_size()

        temp_dir = Path(tempfile.mkdtemp(prefix="training_task_"))

        # Setup logging
        self._setup_logging(output_dir, temp_dir.name)
        self.environment.gpus = gpus

        # Create training configurations
        configs = self._create_training_configs()

        training_task = training_task_repository.get_by_task_id(db=db, task_id=training_task_id)

        try:
            self._execute_training(gpus, configs)
        except Exception as e:
            self._handle_training_error(training_task, e)
        except KeyboardInterrupt:
            training_task.status = TaskStatus.STOPPED
            training_task.error_detail = FailedTrainingException(error_log="Training stopped by user").args[0]
        finally:
            # Cleanup and move files
            self._cleanup_and_move_files(configs, temp_dir)

            # Process training summary
            training_task = self._process_training_summary(db=db, training_task=training_task, destination_folder=temp_dir)

            # Upload model files if completed
            if training_task.status == TaskStatus.COMPLETED:
                calibration_dataset = self.prepare_calibration_dataset(dataset_path=self.data.path.train.image, num_dataset=100)
                self._upload_model_files(training_task, temp_dir, calibration_dataset)

        return training_task

    def _setup_logging(self, output_dir, project_id):
        """Set up the logging directory."""
        self.logging.output_dir = output_dir
        self.logging.project_id = project_id
        self.logging_dir = Path(self.logging.output_dir) / self.logging.project_id / "version_0"

    def _create_training_configs(self):
        """Create training configurations."""
        return TrainerConfigs(
            self.data,
            self.augmentation,
            self.model,
            self.training,
            self.logging,
            self.environment,
        )

    def _execute_training(self, gpus: str, configs: TrainerConfigs):
        """Execute model training."""
        from netspresso_trainer import train_with_yaml

        train_with_yaml(
            gpus=gpus,
            data=configs.data,
            augmentation=configs.augmentation,
            model=configs.model,
            training=configs.training,
            logging=configs.logging,
            environment=configs.environment,
        )

    def _handle_training_error(self, training_task: TrainingTask, error):
        """Handle training errors."""
        e = FailedTrainingException(error_log=error.args[0])
        training_task.status = TaskStatus.ERROR
        training_task.error_detail = e.args[0]

    def _cleanup_and_move_files(self, configs: TrainerConfigs, destination_folder: Path):
        """Clean up temporary files and move result files."""
        FileHandler.remove_folder(configs.temp_folder)
        logger.info(f"Removed {configs.temp_folder} folder.")

        FileHandler.move_and_cleanup_folders(source_folder=self.logging_dir, destination_folder=destination_folder)
        logger.info(f"Files in {self.logging_dir} were moved to {destination_folder}.")

    def _process_training_summary(self, db, training_task: TrainingTask, destination_folder: Path):
        """Process training summary file and update training task status."""
        summary_path = destination_folder / "training_summary.json"

        if not summary_path.exists():
            logger.error(f"Training summary file not found at {summary_path}")
            error_msg = f"Training summary file not found at {summary_path}"
            training_summary = self._create_default_error_summary(error_msg)
            training_task.status = TaskStatus.ERROR
            training_task.error_detail = FailedTrainingException(error_log=error_msg).args[0]
        else:
            try:
                training_summary = FileHandler.load_json(file_path=summary_path)
            except Exception as e:
                logger.error(f"Failed to load training summary: {e}")
                error_msg = f"Failed to load training summary: {str(e)}"
                training_summary = self._create_default_error_summary(error_msg)
                training_task.status = TaskStatus.ERROR
                training_task.error_detail = FailedTrainingException(error_log=error_msg).args[0]
                return training_task

        try:
            training_task = self.create_performance(training_task=training_task, training_summary=training_summary)
        except Exception as e:
            logger.error(f"Error creating performance record: {e}, {training_summary}")
            training_task.status = TaskStatus.ERROR
            training_task.error_detail = FailedTrainingException(error_log=f"Failed to create performance record: {str(e)}").args[0]
            return training_task

        training_task.status = self._get_status_by_training_summary(training_summary.get("status"))
        if training_task.status == TaskStatus.ERROR:
            error_stats = training_summary.get("error_stats", "")
            training_task.error_detail = FailedTrainingException(error_log=error_stats).args[0]

        training_task = training_task_repository.update(db=db, model=training_task)

        return training_task

    def _create_default_error_summary(self, error_msg):
        """Create default error summary."""
        return {
            "train_losses": {}, "valid_losses": {},
            "train_metrics": {}, "valid_metrics": {},
            "metrics_list": [], "primary_metric": "",
            "flops": "0", "params": "0",
            "total_train_time": 0, "best_epoch": 0,
            "last_epoch": 0, "total_epoch": 0, "status": "error",
            "error_stats": error_msg
        }

    def _upload_model_files(self, training_task: TrainingTask, destination_folder: Path, calibration_dataset: str):
        """Upload trained model files to storage."""
        try:
            pt_file, onnx_file = self.find_model_files(destination_folder)

            errors = []
            if not pt_file:
                errors.append("PyTorch (PT) model file not found")
            if not onnx_file:
                errors.append("ONNX model file not found")

            if errors:
                error_msg = f"Required model files missing after training: {', '.join(errors)}"
                logger.error(error_msg)
                self.update_task_status(task_id=training_task.task_id, status=TaskStatus.ERROR, error_message=FailedTrainingException(error_log=error_msg))
                return

            model: Model = training_task.model
            self._upload_file_with_retry(
                local_path=str(pt_file),
                object_path=f"{model.object_path}/model.pt",
                file_type="PT"
            )

            self._upload_file_with_retry(
                local_path=str(onnx_file),
                object_path=f"{model.object_path}/model.onnx",
                file_type="ONNX"
            )

            if Path(calibration_dataset).exists():
                self._upload_file_with_retry(
                    local_path=calibration_dataset,
                    object_path=f"{model.object_path}/{Path(calibration_dataset).name}",
                    file_type="Numpy"
                )

        except Exception as e:
            error_msg = f"Failed to upload model files to Zenko: {e}"
            logger.error(error_msg)
            self.update_task_status(task_id=training_task.task_id, status=TaskStatus.ERROR, error_message=FailedTrainingException(error_log=error_msg))

    def _upload_file_with_retry(self, local_path, object_path, file_type, max_retries=3, retry_delay=5):
        """Execute file upload with retry mechanism."""
        import time

        for attempt in range(max_retries):
            try:
                storage_handler.upload_file_to_s3(
                    bucket_name=settings.MODEL_BUCKET_NAME,
                    local_path=local_path,
                    object_path=object_path,
                )
                logger.info(f"Uploaded {file_type} file to Zenko: {object_path}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"{file_type} file upload attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"Failed to upload {file_type} file after {max_retries} attempts: {e}")

    def get_all_available_models(self) -> Dict[str, List[str]]:
        """Get all available models for each task, excluding deprecated names.

        Returns:
            Dict[str, List[str]]: A dictionary mapping each task to its available models.
        """
        all_models = {
            "classification": CLASSIFICATION_MODELS,
            "detection": DETECTION_MODELS,
            "segmentation": SEGMENTATION_MODELS,
        }
        return all_models

    def get_all_available_optimizers(self) -> Dict[str, Dict]:
        return get_supported_optimizers()

    def get_all_available_schedulers(self) -> Dict[str, Dict]:
        return get_supported_schedulers()

    def find_model_files(self, folder_path: Union[str, Path]) -> tuple[Optional[Path], Optional[Path]]:
        """Find one .pt file and one .onnx file in the given folder

        Args:
            folder_path: Path to search for model files

        Returns:
            tuple[Optional[Path], Optional[Path]]: Tuple of (pt_file_path, onnx_file_path)
            Each can be None if not found
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            return None, None

        pt_files = list(folder_path.glob("*best*.pt")) or list(folder_path.glob("*.pt"))
        onnx_files = list(folder_path.glob("*best*.onnx")) or list(folder_path.glob("*.onnx"))

        if not pt_files:
            logger.warning(f"No PyTorch model files found in {folder_path}")
            pt_file = None
        else:
            pt_file = max(pt_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found PT file: {pt_file.name} (Last modified: {pt_file.stat().st_mtime})")

        if not onnx_files:
            logger.warning(f"No ONNX model files found in {folder_path}")
            onnx_file = None
        else:
            onnx_file = max(onnx_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found ONNX file: {onnx_file.name} (Last modified: {onnx_file.stat().st_mtime})")

        return pt_file, onnx_file

    def prepare_calibration_dataset(self, dataset_path: str, num_dataset: int = 100) -> str:
        """Create a calibration dataset."""
        logger.info("Creating calibration dataset")

        preprocess_list = [
            asdict(aug) for aug in self.augmentation.train
            if aug.name != "totensor"
        ]
        logger.info(f"Using preprocess_list: {preprocess_list}")
        preprocessor = Preprocessor(preprocess_list)

        inputs_array = []

        # Support multiple image extensions
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(f"{dataset_path}/{ext}"))

        # Limit the number of images to num_dataset
        image_paths = image_paths[:num_dataset]

        if not image_paths:
            logger.warning(f"No images found in {dataset_path} with extensions {image_extensions}")
            return

        logger.info(f"Processing {len(image_paths)} images for calibration dataset")

        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocessor(img)
            inputs_array.append(img)

        if not inputs_array:
            logger.warning("No valid images were processed")
            return

        result_array = np.concatenate(inputs_array, axis=0)

        # save chunk data
        calibration_dataset_path = f"{Path(dataset_path).parts[0]}/calibration_dataset.npy"
        np.save(calibration_dataset_path, result_array, allow_pickle=True)
        logger.info(f"Calibration dataset saved to {calibration_dataset_path}")

        return calibration_dataset_path
