import contextvars
import traceback
import uuid

import better_exceptions
from fastapi import status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.exceptions.base import AdditionalData, ExceptionBase

context_data_var = contextvars.ContextVar("context_data")


class ContextData(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_attributes: dict = Field(default_factory=dict)


class ContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        context_data_token = context_data_var.set(ContextData())
        response = None
        try:
            response = await call_next(request)
        except Exception as exc:
            exception_message = "".join(
                better_exceptions.format_exception(type(exc), exc, exc.__traceback__)
            )
            exception = ExceptionBase(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code="NPTS50000",
                data=AdditionalData(
                    error_log=traceback.format_exc(limit=5)
                ),
                name=exc.__class__.__name__,
                message=exc.__repr__(),
            )

            logger.error(exception_message)

            response = JSONResponse(
                status_code=exception.status_code,
                content=exception.detail,
            )
        finally:
            context_data_var.reset(context_data_token)

        return response
