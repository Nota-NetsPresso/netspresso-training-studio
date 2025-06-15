from fastapi import APIRouter

from src.api.v1.endpoints import (
    compression_task,
    conversion_task,
    evaluation_task,
    model,
    project,
    system,
    training_task,
    user,
)

api_router = APIRouter()
api_router.include_router(user.router, prefix="/users", tags=["user"])
api_router.include_router(project.router, prefix="/projects", tags=["project"])
api_router.include_router(model.router, prefix="/models", tags=["model"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(training_task.router, prefix="/tasks", tags=["training"])
api_router.include_router(evaluation_task.router, prefix="/tasks", tags=["evaluation"])
api_router.include_router(compression_task.router, prefix="/tasks", tags=["compression"])
api_router.include_router(conversion_task.router, prefix="/tasks", tags=["conversion"])
