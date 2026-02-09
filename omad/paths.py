"""Path resolution utilities for OMAD stages."""

from pathlib import Path
from omad.config import OmadConfig, get_config


def resolve_stage1_paths(
    T: int,
    config: OmadConfig | None = None,
    **overrides
) -> dict[str, Path]:
    """
    Auto-derive Stage 1 paths from T parameter.

    Args:
        T: Time slice length (12, 24, 48, 72)
        config: Optional config instance (uses global if None)
        **overrides: Override specific paths

    Returns:
        Dictionary of resolved paths
    """
    if config is None:
        config = get_config()

    base_dir = config.paths.route_sliced_dir(T)

    return {
        "output_dir": overrides.get("output_dir") or base_dir,
        "raw_routes_csv": overrides.get("raw_routes_csv") or config.paths.raw_routes_csv(T),
        "preprocessed_csv": overrides.get("preprocessed_csv") or config.paths.preprocessed_csv(T),
        "indices_dir": overrides.get("indices_dir") or config.paths.indices_dir(T),
        "user_query_dir": overrides.get("user_query_dir") or config.paths.user_query_dir(T),
    }


def resolve_stage2_paths(
    T: int,
    config: OmadConfig | None = None,
    **overrides
) -> dict[str, Path]:
    """
    Auto-derive Stage 2 paths from T parameter.

    Args:
        T: Time slice length (12, 24, 48, 72)
        config: Optional config instance (uses global if None)
        **overrides: Override specific paths

    Returns:
        Dictionary of resolved paths
    """
    if config is None:
        config = get_config()

    batch_root = overrides.get("batch_root") or config.paths.route_sliced_dir(T)
    output_dir = overrides.get("output_dir") or config.paths.injected_json_dir(T)
    log_dir = overrides.get("log_dir") or (batch_root / "logs")

    return {
        "batch_root": batch_root,
        "output_dir": output_dir,
        "log_dir": log_dir,
    }


def resolve_stage3_paths(
    T: int,
    config: OmadConfig | None = None,
    **overrides
) -> dict[str, Path]:
    """
    Auto-derive Stage 3 paths from T parameter.

    Args:
        T: Time slice length (12, 24, 48, 72)
        config: Optional config instance (uses global if None)
        **overrides: Override specific paths

    Returns:
        Dictionary of resolved paths
    """
    if config is None:
        config = get_config()

    base = config.paths.route_sliced_dir(T)

    return {
        "input_csv": overrides.get("input_csv") or config.paths.preprocessed_csv(T),
        "track_csv": overrides.get("track_csv") or config.paths.raw_routes_csv(T),
        "injected_dir": overrides.get("injected_dir") or config.paths.injected_json_dir(T),
        "output_csv": overrides.get("output_csv") or config.paths.injected_csv(T),
    }


def resolve_stage4_paths(
    base_dir: str | Path,
    T: int | None = None,
    config: OmadConfig | None = None,
    **overrides
) -> dict[str, Path]:
    """
    Auto-derive Stage 4 paths.

    Args:
        base_dir: Base directory (usually shared_folder)
        T: Optional time slice length
        config: Optional config instance (uses global if None)
        **overrides: Override specific paths

    Returns:
        Dictionary of resolved paths
    """
    if config is None:
        config = get_config()

    base_path = Path(base_dir)

    paths = {
        "base_dir": base_path,
        "output_dir": overrides.get("output_dir") or (base_path / "data"),
        "indices_dir": overrides.get("indices_dir") or (base_path / "indices"),
    }

    if T is not None:
        paths["injected_csv"] = overrides.get("injected_csv") or config.paths.injected_csv(T)

    return paths
