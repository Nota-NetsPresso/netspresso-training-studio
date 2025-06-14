from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EnvironmentCreate(BaseModel):
    seed: Optional[int] = Field(default=1, description="Random seed")
    num_workers: Optional[int] = Field(default=4, description="Number of workers")
    gpus: str = Field(default="0", description="GPUs to use")
    batch_size: Optional[int] = Field(default=8, description="Batch size")
    cache_data: Optional[bool] = Field(default=True, description="Cache data")


class EnvironmentPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    gpus: str
