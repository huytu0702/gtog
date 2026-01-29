"""Configuration model for evaluation runs."""

from pathlib import Path

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    path: str = Field(description="Path to QA dataset JSON file")


class OutputConfig(BaseModel):
    """Output configuration."""

    dir: str = Field(default="eval/results", description="Output directory for results")
    save_intermediate: bool = Field(default=True, description="Save intermediate results during run")


class JudgeConfig(BaseModel):
    """LLM judge configuration."""

    model: str | None = Field(default=None, description="Model to use for judging (None = use default)")
    temperature: float = Field(default=0.0, description="Temperature for judge LLM")


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    dataset: DatasetConfig
    indexes: dict[str, str] = Field(description="Mapping of imdb_key to index root directory")
    methods: list[str] = Field(default=["tog", "local", "basic"], description="Search methods to evaluate")
    output: OutputConfig = Field(default_factory=OutputConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        """Load configuration from YAML file."""
        import yaml

        from pathlib import Path

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)
