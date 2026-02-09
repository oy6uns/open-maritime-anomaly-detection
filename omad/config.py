"""Configuration management for OMAD CLI tool using Pydantic models."""

from pathlib import Path
from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    """Path configuration for OMAD."""

    # Use current working directory as base
    project_root: Path = Path(".")
    data_dir: Path = Path("./data")

    def route_sliced_dir(self, T: int) -> Path:
        """Get route_sliced_{T} directory path."""
        return self.project_root / f"route_sliced_{T}"

    def preprocessed_csv(self, T: int) -> Path:
        """Get preprocessed CSV path for given T."""
        return self.route_sliced_dir(T) / f"routes_sliced_{T}_preprocessed.csv"

    def injected_csv(self, T: int) -> Path:
        """Get injected CSV path for given T."""
        return self.route_sliced_dir(T) / f"routes_sliced_{T}_injected.csv"

    def raw_routes_csv(self, T: int) -> Path:
        """Get raw routes CSV path for given T."""
        return self.route_sliced_dir(T) / f"routes_sliced_{T}.csv"

    def user_query_dir(self, T: int) -> Path:
        """Get user_query directory path for given T."""
        return self.route_sliced_dir(T) / "user_query"

    def indices_dir(self, T: int) -> Path:
        """Get indices directory path for given T."""
        return self.route_sliced_dir(T) / "indices"

    def injected_json_dir(self, T: int) -> Path:
        """Get injected JSON directory path for given T."""
        return self.route_sliced_dir(T) / "injected"


class LLMConfig(BaseModel):
    """LLM configuration."""

    model_id: str = "Qwen/Qwen3-8B"
    cache_dir: Path | None = None  # Will use HuggingFace default
    max_new_tokens: int = 256
    max_retries: int = -1


class InjectionConfig(BaseModel):
    """Anomaly injection configuration."""

    theta_g: float = 3.0  # A1 SED multiplier
    theta_v: float = 2.0  # A2 speed/heading multiplier
    default_seed: int = 0


class DatasetConfig(BaseModel):
    """Dataset preparation configuration."""

    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    default_seeds: list[int] = [2, 12, 22, 32, 42]
    default_ratios: list[int] = [10, 5, 3, 1]


class OmadConfig(BaseModel):
    """Master configuration for OMAD."""

    paths: PathConfig = Field(default_factory=PathConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)


# Global config instance
_config: OmadConfig | None = None


def get_config() -> OmadConfig:
    """Get global config instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = OmadConfig()
    return _config
