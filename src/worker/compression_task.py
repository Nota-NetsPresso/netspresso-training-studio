from loguru import logger

from src.modules.compressor.v2.compressor import CompressorV2
from src.worker.celery_app import celery_app

POLLING_INTERVAL = 30  # seconds


@celery_app.task(bind=True, name='compress_model')
def compress_model(
    self,
    api_key: str,
    method: str,
    recommendation_method: str,
    ratio: float,
    options: dict,
    input_model_id: str,
    compression_task_id: str,
):
    compressor = CompressorV2(api_key=api_key)

    try:
        task_id = compressor.recommendation_compression(
            compression_method=method,
            recommendation_method=recommendation_method,
            recommendation_ratio=ratio,
            options=options,
            input_model_id=input_model_id,
            compression_task_id=compression_task_id,
        )
        result = {"task_id": task_id, "status": "completed"}
        return result
    except Exception as e:
        logger.error(f"Compression failed: {str(e)}")
        raise e
