import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy.orm import Session

from src.api.v1.schemas.tasks.common.device import (
    HardwareTypePayload,
    PrecisionForConversionPayload,
    SoftwareVersionPayload,
    SupportedDevicePayload,
    SupportedDeviceResponse,
)
from src.api.v1.schemas.tasks.conversion.conversion_task import (
    ConversionCreate,
    TargetFrameworkPayload,
)
from src.api.v1.schemas.tasks.evaluation.evaluation_task import (
    BoundingBox,
    EvaluationCreate,
    EvaluationPayload,
    EvaluationResultsPayload,
    ImagePrediction,
    PredictionForThreshold,
)
from src.configs.settings import settings
from src.enums.conversion import EvaluationTargetFramework, SourceFramework, TargetFramework
from src.enums.device import DeviceName, SoftwareVersion
from src.enums.model import DataType
from src.enums.sort import Order, TimeSort
from src.enums.task import TaskStatus
from src.exceptions.conversion import ConversionTaskNotFoundException
from src.exceptions.evaluation import EvaluationTaskAlreadyExistsException
from src.models.conversion import ConversionTask
from src.models.evaluation import EvaluationDataset, EvaluationTask
from src.modules.clients.launcher.v2.schemas.common import DeviceInfo
from src.modules.converter.v2.converter import ConverterV2
from src.repositories.conversion import conversion_task_repository
from src.repositories.evaluation import evaluation_dataset_repository, evaluation_task_repository
from src.repositories.model import model_repository
from src.services.conversion_task import conversion_task_service
from src.services.user import user_service
from src.utils.file import FileHandler
from src.worker.evaluation_task import chain_conversion_and_evaluation, run_multiple_evaluations
from src.zenko.storage_handler import ObjectStorageHandler

storage_handler = ObjectStorageHandler()


class EvaluationTaskService:
    def __init__(self):
        self.confidence_scores = settings.EVALUATION_CONFIDENCE_SCORES

    def get_supported_devices(
        self, db: Session, framework: SourceFramework, api_key: str
    ) -> List[SupportedDeviceResponse]:
        """Get supported devices for conversion tasks.

        Args:
            db (Session): Database session
            framework (SourceFramework): Framework to get supported devices for
            api_key (str): API key for authentication

        Returns:
            List[SupportedDeviceResponse]: List of supported devices grouped by framework
        """
        converter = ConverterV2(api_key=api_key)
        supported_options = converter.get_supported_options(framework=framework)

        supported_framework = [TargetFramework.TENSORFLOW_LITE]

        return [self._create_supported_device_response(option) for option in supported_options if option.framework in supported_framework]

    def _create_supported_device_response(self, option) -> SupportedDeviceResponse:
        """Create SupportedDeviceResponse from converter option.

        Args:
            option: Converter option containing framework and devices information

        Returns:
            SupportedDeviceResponse: Response containing framework and supported devices
        """
        return SupportedDeviceResponse(
            framework=TargetFrameworkPayload(name=option.framework),
            devices=[self._create_device_payload(device) for device in option.devices],
        )

    def _create_device_payload(self, device: DeviceInfo) -> SupportedDevicePayload:
        """Create SupportedDevicePayload from device information.

        Args:
            device: Device information containing name, versions, precisions, and hardware types

        Returns:
            SupportedDevicePayload: Payload containing device information
        """
        return SupportedDevicePayload(
            name=device.device_name,
            software_versions=[
                SoftwareVersionPayload(name=version.software_version) for version in device.software_versions
            ],
            precisions=[PrecisionForConversionPayload(name=precision) for precision in device.data_types],
            hardware_types=[HardwareTypePayload(name=hardware_type) for hardware_type in device.hardware_types],
        )

    def _find_existing_conversion_task(
        self,
        db: Session,
        input_model_id: str,
        target_framework: TargetFramework,
        target_device_name: DeviceName,
        target_software_version: Optional[SoftwareVersion] = None,
        target_data_type: DataType = DataType.FP16
    ) -> Optional[ConversionTask]:
        """Find an existing conversion task that matches the given parameters.

        Args:
            input_model_id: ID of the input model
            target_framework: Target framework for conversion
            target_device_name: Target device for conversion
            target_software_version: Target software version (optional)
            target_data_type: Target data type/precision

        Returns:
            The task_id of the matching conversion task, or None if no match found
        """
        # Find conversion tasks for the input model
        conversion_tasks = conversion_task_repository.get_all_by_model_id(
            db=db,
            model_id=input_model_id
        )

        # Filter tasks by the conversion parameters
        for task in conversion_tasks:
            if (task.framework == target_framework and
                task.device_name == target_device_name and
                task.precision == target_data_type and
                (target_software_version is None or task.software_version == target_software_version) and
                task.status == TaskStatus.COMPLETED):
                return task

        raise ConversionTaskNotFoundException()

    def check_evaluation_task_status(
        self,
        db: Session,
        model_id: str,
        dataset_path: str,
        confidence_score: float
    ):
        evaluation_task = evaluation_task_repository.get_by_model_dataset_and_confidence(
            db=db,
            model_id=model_id,
            dataset_id=dataset_path,
            confidence_score=confidence_score
        )

        if evaluation_task:
            if evaluation_task.status == TaskStatus.COMPLETED:
                logger.warning(f"Evaluation task already completed: {evaluation_task.task_id}")
                raise EvaluationTaskAlreadyExistsException(task_id=evaluation_task.task_id, task_status=TaskStatus.COMPLETED.value)
            elif evaluation_task.status == TaskStatus.IN_PROGRESS:
                logger.warning(f"Evaluation task already in progress: {evaluation_task.task_id}")
                raise EvaluationTaskAlreadyExistsException(task_id=evaluation_task.task_id, task_status=TaskStatus.IN_PROGRESS.value)
            elif evaluation_task.status == TaskStatus.ERROR:
                logger.info(f"Retrying failed evaluation task: {evaluation_task.task_id}")
            else:
                # Other status (NOT_STARTED, STOPPED, etc.)
                logger.info(f"Using existing evaluation task with ID: {evaluation_task.task_id}")

    def create_evaluation_task(
        self,
        db: Session,
        evaluation_in: EvaluationCreate,
        api_key: str,
    ) -> str:
        """
        Main entry point for creating evaluation tasks.
        Delegates to specific handlers based on framework type.
        """
        if evaluation_in.conversion.framework == EvaluationTargetFramework.ONNX:
            return self._handle_onnx_evaluation(db, evaluation_in, api_key)
        else:
            return self._handle_conversion_evaluation(db, evaluation_in, api_key)

    def _handle_onnx_evaluation(
        self,
        db: Session,
        evaluation_in: EvaluationCreate,
        api_key: str,
    ) -> str:
        """Handle evaluation for ONNX models (no conversion needed)."""
        logger.info("Processing ONNX model evaluation without conversion")

        input_model = model_repository.get_by_model_id(db=db, model_id=evaluation_in.input_model_id)

        # Check for duplicate tasks
        self._check_duplicate_evaluation_tasks(
            db=db,
            model_id=input_model.model_id,
            dataset_path=evaluation_in.dataset_path
        )

        # Create evaluation tasks for all confidence scores
        evaluation_task_ids = self._create_evaluation_tasks(
            db=db,
            model_id=input_model.model_id,
            dataset_path=evaluation_in.dataset_path,
            training_task_id=evaluation_in.training_task_id,
            user_id=input_model.user_id,
            conversion_task_id=None
        )

        # Start evaluation
        self._start_evaluation_tasks(
            api_key=api_key,
            model_id=input_model.model_id,
            dataset_id=evaluation_in.dataset_path,
            training_task_id=evaluation_in.training_task_id,
            evaluation_task_ids=evaluation_task_ids
        )

        return ""

    def _handle_conversion_evaluation(
        self,
        db: Session,
        evaluation_in: EvaluationCreate,
        api_key: str,
    ) -> str:
        """Handle evaluation for models that need conversion."""
        try:
            # Try to find existing conversion task
            conversion_task = self._find_existing_conversion_task(
                db=db,
                input_model_id=evaluation_in.input_model_id,
                target_framework=evaluation_in.conversion.framework,
                target_device_name=evaluation_in.conversion.device_name,
                target_software_version=evaluation_in.conversion.software_version,
                target_data_type=evaluation_in.conversion.precision
            )

            return self._handle_existing_conversion_evaluation(
                db=db,
                evaluation_in=evaluation_in,
                api_key=api_key,
                conversion_task=conversion_task
            )

        except ConversionTaskNotFoundException:
            return self._handle_new_conversion_evaluation(
                db=db,
                evaluation_in=evaluation_in,
                api_key=api_key
            )

    def _handle_existing_conversion_evaluation(
        self,
        db: Session,
        evaluation_in: EvaluationCreate,
        api_key: str,
        conversion_task: ConversionTask
    ) -> str:
        """Handle evaluation using existing conversion task."""
        logger.info(f"Using existing conversion task: {conversion_task.task_id}")

        # Check for duplicate tasks
        self._check_duplicate_evaluation_tasks(
            db=db,
            model_id=conversion_task.model_id,
            dataset_path=evaluation_in.dataset_path
        )

        # Create evaluation tasks
        evaluation_task_ids = self._create_evaluation_tasks(
            db=db,
            model_id=conversion_task.model_id,
            dataset_path=evaluation_in.dataset_path,
            training_task_id=evaluation_in.training_task_id,
            user_id=conversion_task.user_id,
            conversion_task_id=conversion_task.task_id
        )

        # Start evaluation
        self._start_evaluation_tasks(
            api_key=api_key,
            model_id=conversion_task.model_id,
            dataset_id=evaluation_in.dataset_path,
            training_task_id=evaluation_in.training_task_id,
            evaluation_task_ids=evaluation_task_ids
        )

        return ""

    def _handle_new_conversion_evaluation(
        self,
        db: Session,
        evaluation_in: EvaluationCreate,
        api_key: str,
    ) -> str:
        """Handle evaluation by creating new conversion task and chaining."""
        logger.info("No existing conversion task found. Creating new conversion task and chaining with evaluation.")

        # Get model information
        input_model = model_repository.get_by_model_id(db=db, model_id=evaluation_in.input_model_id)

        # Create and start conversion task
        conversion_task_payload = self._create_and_start_conversion_task(
            db=db,
            evaluation_in=evaluation_in,
            api_key=api_key
        )

        evaluation_task_ids = self._create_evaluation_tasks(
            db=db,
            model_id=input_model.model_id,
            dataset_path=evaluation_in.dataset_path,
            training_task_id=evaluation_in.training_task_id,
            user_id=input_model.user_id,
            conversion_task_id=conversion_task_payload.task_id
        )

        # Start conversion + evaluation chain
        _ = chain_conversion_and_evaluation.apply_async(
            kwargs={
                "api_key": api_key,
                "input_model_id": input_model.model_id,
                "conversion_task_id": conversion_task_payload.task_id,
                "target_framework": evaluation_in.conversion.framework,
                "target_device_name": evaluation_in.conversion.device_name,
                "target_data_type": evaluation_in.conversion.precision,
                "target_software_version": evaluation_in.conversion.software_version,
                "dataset_id": evaluation_in.dataset_path,
                "training_task_id": evaluation_in.training_task_id,
                "evaluation_task_ids": evaluation_task_ids,
                "confidence_scores": self.confidence_scores,
            }
        )

        return ""

    def _check_duplicate_evaluation_tasks(
        self,
        db: Session,
        model_id: str,
        dataset_path: str
    ) -> None:
        """Check for duplicate evaluation tasks across all confidence scores."""
        for confidence_score in self.confidence_scores:
            self.check_evaluation_task_status(
                db=db,
                model_id=model_id,
                dataset_path=dataset_path,
                confidence_score=confidence_score
            )

    def _create_evaluation_tasks(
        self,
        db: Session,
        model_id: str,
        dataset_path: str,
        training_task_id: str,
        user_id: str,
        conversion_task_id: Optional[str] = None
    ) -> List[str]:
        """Create evaluation tasks for all confidence scores."""
        evaluation_task_ids = []

        for confidence_score in self.confidence_scores:
            evaluation_task = EvaluationTask(
                # dataset_id=dataset_path,
                input_model_id=model_id,
                training_task_id=training_task_id,
                conversion_task_id=conversion_task_id,
                confidence_score=confidence_score,
                status=TaskStatus.NOT_STARTED,
                user_id=user_id,
            )
            evaluation_task = evaluation_task_repository.save(db=db, model=evaluation_task)
            evaluation_task_ids.append(evaluation_task.task_id)

        return evaluation_task_ids

    def _start_evaluation_tasks(
        self,
        api_key: str,
        model_id: str,
        dataset_id: str,
        training_task_id: str,
        evaluation_task_ids: List[str]
    ) -> None:
        """Start the evaluation tasks using Celery."""
        run_multiple_evaluations.apply_async(
            kwargs={
                "api_key": api_key,
                "input_model_id": model_id,
                "dataset_id": dataset_id,
                "training_task_id": training_task_id,
                "evaluation_task_ids": evaluation_task_ids,
                "confidence_scores": self.confidence_scores,
            }
        )

    def _create_and_start_conversion_task(
        self,
        db: Session,
        evaluation_in: EvaluationCreate,
        api_key: str
    ):
        """Create and start a new conversion task."""

        conversion_in = ConversionCreate(
            input_model_id=evaluation_in.input_model_id,
            framework=evaluation_in.conversion.framework,
            device_name=evaluation_in.conversion.device_name,
            precision=evaluation_in.conversion.precision,
            software_version=evaluation_in.conversion.software_version,
        )

        conversion_task = conversion_task_service.create_conversion_task(
            db=db,
            conversion_in=conversion_in,
            api_key=api_key,
        )

        conversion_task_payload = conversion_task_service.start_conversion_task(
            conversion_in=conversion_in,
            conversion_task=conversion_task,
            api_key=api_key,
        )

        return conversion_task_payload

    def get_evaluation_tasks(
        self,
        db: Session,
        token: str,
        model_id: str,
    ) -> List[EvaluationPayload]:
        user_info = user_service.get_user_info(token=token)
        evaluation_tasks = evaluation_task_repository.get_all_by_user_id_and_model_id(
            db=db,
            user_id=user_info.user_id,
            model_id=model_id
        )

        return [EvaluationPayload.model_validate(evaluation_task) for evaluation_task in evaluation_tasks]

    def count_evaluation_task_by_user_id(
        self,
        db: Session,
        token: str,
        model_id: str,
    ) -> int:
        user_info = user_service.get_user_info(token=token)
        return evaluation_task_repository.count_by_user_id_and_model_id(
            db=db,
            user_id=user_info.user_id,
            model_id=model_id
        )

    def get_unique_datasets_by_model_id(
        self,
        db: Session,
        token: str,
        model_id: str,
    ) -> List[EvaluationDataset]:
        """Get unique dataset IDs used for evaluating a specific model.

        Args:
            db: Database session
            api_key: API key for authentication
            model_id: Model ID

        Returns:
            List[str]: List of unique dataset IDs
        """
        user_info = user_service.get_user_info(token=token)
        dataset_ids = evaluation_task_repository.get_unique_datasets_by_model_id(
            db=db,
            user_id=user_info.user_id,
            model_id=model_id
        )
        evaluation_datasets = evaluation_dataset_repository.get_by_dataset_ids(
            db=db,
            dataset_ids=dataset_ids
        )

        return evaluation_datasets

    def get_evaluation_results_by_model_and_dataset(
        self,
        db: Session,
        token: str,
        model_id: str,
        dataset_id: str,
    ) -> List[EvaluationTask]:
        """Get evaluation results for a specific model and dataset.

        Args:
            db: Database session
            api_key: API key for authentication
            model_id: Model ID
            dataset_id: Dataset ID

        Returns:
            List[EvaluationTask]: List of evaluation results
        """
        user_info = user_service.get_user_info(token=token)
        return evaluation_task_repository.get_all_by_model_and_dataset(
            db=db,
            user_id=user_info.user_id,
            model_id=model_id,
            dataset_id=dataset_id
        )

    def get_image_urls_from_s3(self, user_id: str, task_id: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Get image paths and presigned URLs from S3 for a specific task

        Args:
            user_id: User ID
            task_id: Task ID

        Returns:
            Tuple containing list of image paths and dictionary of image URLs
        """
        # Get list of image files from result_images directory
        result_images_prefix = f"{user_id}/{task_id}/result_images/"
        image_objects = storage_handler.list_objects(
            bucket_name=settings.EVALUATION_BUCKET_NAME,
            prefix=result_images_prefix
        )

        # Filter only files with '_images' in the path
        image_paths = [obj for obj in image_objects if '_images' in Path(obj).name]

        # Create presigned URLs for each image
        image_urls = {}
        for image_path in image_paths:
            # Extract image filename (e.g., "000001_images.png")
            image_filename = Path(image_path).name

            # Create presigned URL
            presigned_url = storage_handler.get_download_presigned_url(
                bucket_name=settings.EVALUATION_BUCKET_NAME,
                object_path=image_path,
                download_name=image_filename,
                expires_in=3600  # 1 hour
            )

            # Store URL with filename as key
            image_urls[image_filename] = presigned_url

        return image_paths, image_urls

    def _initialize_image_predictions(self, image_paths: List[str], image_urls: Dict[str, str]) -> Dict[str, ImagePrediction]:
        """
        Initialize ImagePrediction objects from a list of image paths

        Args:
            image_paths: List of image paths
            image_urls: Dictionary of image URLs (filename -> URL)

        Returns:
            Dictionary of ImagePrediction objects for each image
        """
        image_predictions = {}

        for image_path in image_paths:
            image_filename = Path(image_path).name
            image_url = image_urls.get(image_filename)

            if not image_url:
                logger.warning(f"No presigned URL found for image {image_filename}")
                continue

            # Create prediction entry for this image
            image_predictions[image_filename] = ImagePrediction(
                image_id=image_filename,
                image_url=image_url,
                predictions=[]
            )

        return image_predictions

    def _process_threshold_predictions(
        self,
        threshold: float,
        task: EvaluationTask,
        image_paths: List[str],
        image_predictions: Dict[str, ImagePrediction],
        temp_path: Path
    ) -> None:
        """
        Process predictions for a specific threshold value

        Args:
            threshold: Threshold value to process
            task: Evaluation task
            image_paths: List of image paths
            image_predictions: Dictionary of image predictions to update
            temp_path: Temporary file path
        """
        user_id = task.user_id
        task_id = task.task_id

        # Download predictions.json for this threshold
        predictions_object_path = f"{user_id}/{task_id}/predictions.json"
        predictions_local_path = temp_path / f"predictions_{threshold}.json"

        try:
            storage_handler.download_file_from_s3(
                bucket_name=settings.EVALUATION_BUCKET_NAME,
                object_path=predictions_object_path,
                local_path=predictions_local_path.as_posix()
            )

            # Read predictions.json
            predictions_data = FileHandler.load_json(predictions_local_path)

            # Process predictions for this threshold
            if "predictions" not in predictions_data:
                logger.warning(f"No predictions found in {predictions_local_path}")
                return

            base_predictions = predictions_data["predictions"]

            # Limit to the minimum length of both arrays to ensure 1:1 mapping
            max_items = min(len(base_predictions), len(image_paths))

            # Process each image prediction
            for i in range(max_items):
                pred = base_predictions[i]
                image_path = image_paths[i]

                # Extract filename from path
                image_filename = Path(image_path).name

                # Skip if this image was not initialized (likely due to missing URL)
                if image_filename not in image_predictions:
                    continue

                # Add predictions for this threshold to the image
                bboxes = [BoundingBox.model_validate(bbox) for bbox in pred.get("bboxes", [])]

                # Create threshold-specific prediction
                threshold_prediction = PredictionForThreshold(
                    threshold=threshold,
                    bboxes=bboxes
                )

                # Add to the image's predictions list
                image_predictions[image_filename].predictions.append(threshold_prediction)

        except Exception as e:
            logger.error(f"Error processing predictions for threshold {threshold}: {str(e)}")
            # Continue with other thresholds even if one fails

    def get_evaluation_result_details(
        self,
        db: Session,
        token: str,
        model_id: str,
        dataset_id: str,
        start: int = 0,
        size: int = 20,
    ) -> EvaluationResultsPayload:
        """Get detailed evaluation results including predictions and result images with pagination.

        Args:
            db: Database session
            api_key: API key for authentication
            model_id: Model ID
            dataset_id: Dataset ID
            start: Pagination start index
            size: Page size (number of images)

        Returns:
            EvaluationResultsPayload: Detailed evaluation results with predictions and image URLs
        """
        # Get evaluation tasks for this model and dataset
        evaluation_tasks = self.get_evaluation_results_by_model_and_dataset(
            db=db,
            token=token,
            model_id=model_id,
            dataset_id=dataset_id
        )

        if not evaluation_tasks:
            logger.warning(f"No evaluation tasks found for model {model_id} and dataset {dataset_id}")
            return EvaluationResultsPayload(
                model_id=model_id,
                dataset_id=dataset_id,
                results=[],
                result_count=0,
                total_count=0
            )

        if evaluation_tasks[0].is_dataset_deleted:
            logger.info(f"Dataset {dataset_id} is deleted. Returning empty results.")
            return EvaluationResultsPayload(
                model_id=model_id,
                dataset_id=dataset_id,
                results=[],
                result_count=0,
                total_count=0
            )

        # Create temporary directory for downloads
        temp_dir = tempfile.mkdtemp(prefix="evaluation_results_")
        temp_path = Path(temp_dir)

        logger.info(f"evaluation_tasks: {len(evaluation_tasks)}")

        try:
            # Get list of image files and presigned URLs using the first completed task
            # (image URLs should be the same for all tasks with the same dataset)
            first_task = evaluation_tasks[0]
            image_paths, image_urls = self.get_image_urls_from_s3(first_task.user_id, first_task.task_id)

            # Sort image paths to ensure consistent ordering
            image_paths.sort()

            # 1. Initialize prediction objects for all images
            image_predictions = self._initialize_image_predictions(image_paths, image_urls)

            # 2. Process each evaluation task directly
            for task in evaluation_tasks:
                # Use the actual threshold from the database
                threshold = task.confidence_score

                logger.info(f"Processing task with confidence score {threshold}")
                self._process_threshold_predictions(
                    threshold=threshold,
                    task=task,
                    image_paths=image_paths,
                    image_predictions=image_predictions,
                    temp_path=temp_path
                )

            # 3. Convert results and apply pagination
            results = list(image_predictions.values())

            total_count = len(results)

            # Check for empty results
            if total_count == 0:
                logger.warning("No image predictions were created")
                return EvaluationResultsPayload(
                    model_id=model_id,
                    dataset_id=dataset_id,
                    results=[],
                    result_count=0,
                    total_count=0
                )

            # Validate start index
            if start >= total_count:
                start = 0

            # Calculate end index
            end = min(start + size, total_count)

            # Get paginated results
            paginated_predictions = results[start:end]

            # Return combined results with pagination info
            return EvaluationResultsPayload(
                model_id=model_id,
                dataset_id=dataset_id,
                results=paginated_predictions,
                result_count=len(paginated_predictions),
                total_count=total_count
            )

        except Exception as e:
            logger.error(f"Error retrieving evaluation results: {str(e)}")
            raise e

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def delete_evaluation_task(self, db: Session, evaluation_task_id: str, api_key: str) -> EvaluationPayload:
        evaluation_task = evaluation_task_repository.get_by_task_id(db=db, task_id=evaluation_task_id)
        evaluation_task_repository.soft_delete(db=db, model=evaluation_task)

        return EvaluationPayload.model_validate(evaluation_task)

    def _get_evaluation_info(self, db: Session, model_id: str) -> tuple[Optional[str], List[str]]:
        """Get evaluation task information

        Args:
            db: Database session
            model_id: Model ID

        Returns:
            tuple: (latest_status, task_ids)
        """
        evaluation_tasks = evaluation_task_repository.get_all_by_model_id(
            db=db,
            model_id=model_id,
            order=Order.DESC,
            time_sort=TimeSort.CREATED_AT,
        )
        if not evaluation_tasks:
            return None, []

        task_ids = [task.task_id for task in evaluation_tasks]

        evaluation_task = evaluation_task_repository.get_latest_evaluation_task(
            db=db,
            model_id=model_id,
            order=Order.DESC,
            time_sort=TimeSort.UPDATED_AT,
        )
        latest_status = evaluation_task.status

        return latest_status, task_ids

    def delete_evaluation_dataset(self, db: Session, dataset_id: str, api_key: str) -> EvaluationPayload:
        evaluation_tasks = evaluation_task_repository.get_all_by_dataset_id(db=db, dataset_id=dataset_id)
        for evaluation_task in evaluation_tasks:
            evaluation_task.is_dataset_deleted = True
            evaluation_task_repository.update(db=db, model=evaluation_task)

        return EvaluationPayload.model_validate(evaluation_task)

    def get_evaluation_tasks_by_ids(self, db: Session, model_ids: List[str]) -> List[EvaluationPayload]:
        evaluation_tasks = evaluation_task_repository.get_all_by_model_ids(
            db=db, model_ids=model_ids, order=Order.DESC, time_sort=TimeSort.CREATED_AT, is_deleted=False
        )
        return [EvaluationPayload.model_validate(task) for task in evaluation_tasks]

    def get_evaluation_task(self, db: Session, task_id: str) -> EvaluationPayload:
        evaluation_task = evaluation_task_repository.get_by_task_id(db, task_id)
        return EvaluationPayload.model_validate(evaluation_task)


evaluation_task_service = EvaluationTaskService()
