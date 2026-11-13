from pathlib import Path
from pydantic import BaseModel, Field
import yaml

class DataConfig(BaseModel):
    data_dir: Path = Path("data")
    data_path: Path | None = None
    target_col: str = "MedHouseVal"
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)
    random_state: int = 42

class ModelConfig(BaseModel):
    n_estimators: int = Field(default=100, gt=0)
    max_depth: int | None = None
    min_samples_split: int = Field(default=2, ge=2)
    random_state: int = 42
    n_jobs: int = -1

class ArtifactConfig(BaseModel):
    output_model_path: Path = Path("artifacts/rfr.pkl")
    output_metrics_path: Path | None = Path("artifacts/metrics.json")

class TrainingPipelineConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrainingPipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)