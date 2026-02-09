"""Configuration management for OMAD CLI tool using Pydantic models."""

from pathlib import Path
from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    """Path configuration for OMAD."""

    nas_base: Path = Path("/workspace/NAS/KRISO2026")
    local_base: Path = Path("/workspace/Local")
    shared_folder: Path = Path("/workspace/NAS/KRISO2026/shared_folder")
    west_grid_base: Path = Path("/workspace/NAS/West Grid")

    def route_sliced_dir(self, T: int) -> Path:
        """Get route_sliced_{T} directory path."""
        return self.nas_base / f"route_sliced_{T}"

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

    def indices_dir(self) -> Path:
        """Get indices directory path."""
        return self.shared_folder / "indices"

    def injected_json_dir(self, T: int) -> Path:
        """Get injected JSON directory path for given T."""
        return self.route_sliced_dir(T) / "injected"


class LLMConfig(BaseModel):
    """LLM configuration."""

    model_id: str = "Qwen/Qwen3-8B"
    cache_dir: Path = Path("/nas/home/oy6uns")
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
    default_percentages: list[str] = ["10pct", "5pct", "3pct", "1pct"]


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
