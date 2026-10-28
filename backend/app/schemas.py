from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

class PipelineConfig(BaseModel):
    samples: int = Field(default=320, ge=100, le=5000)
    random_seed: int = Field(default=42, ge=0, le=1_000_000)
    test_size: float = Field(default=0.25, gt=0.1, lt=0.5)
    quality_threshold: float = Field(default=0.70, ge=0, le=1)
    run_tuning: bool = True

class PipelineRunRequest(BaseModel):
    pipeline_id: str = "intelligent-classification-v1"
    config: PipelineConfig = Field(default_factory=PipelineConfig)

class ArtifactRef(BaseModel):
    id: str
    name: str
    kind: str
    download_url: str
    size_bytes: int

class NodeResult(BaseModel):
    node_id: str
    component: str
    status: Literal['succeeded', 'failed', 'skipped']
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)

class PipelineRunResponse(BaseModel):
    run_id: str
    pipeline_id: str
    status: Literal['succeeded', 'failed']
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    config: dict[str, Any]
    nodes: list[NodeResult]
    metrics: dict[str, Any]
    comparison: dict[str, Any]
    quality_gate: dict[str, Any]
    artifacts: list[ArtifactRef]
    logs: list[str]
