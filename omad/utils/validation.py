"""Input validation utilities for OMAD CLI."""

from pathlib import Path
from typing import List


def validate_T_parameter(T: int) -> bool:
    """
    Validate T (time slice) parameter.

    Args:
        T: Time slice length

    Returns:
        True if valid

    Raises:
        ValueError: If T is not in valid range
    """
    valid_values = [12, 24, 48, 72]
    if T not in valid_values:
        raise ValueError(
            f"Invalid T parameter: {T}. Must be one of {valid_values}"
        )
    return True


def validate_percentages(percentages: List[str]) -> bool:
    """
    Validate percentage values.

    Args:
        percentages: List of percentage strings

    Returns:
        True if valid

    Raises:
        ValueError: If any percentage is invalid
    """
    valid_pcts = {"10pct", "5pct", "3pct", "1pct"}
    invalid = set(percentages) - valid_pcts
    if invalid:
        raise ValueError(
            f"Invalid percentages: {invalid}. Must be one of {valid_pcts}"
        )
    return True


def validate_split_ratios(train: float, valid: float, test: float) -> bool:
    """
    Validate train/valid/test split ratios.

    Args:
        train: Train ratio
        valid: Validation ratio
        test: Test ratio

    Returns:
        True if valid

    Raises:
        ValueError: If ratios don't sum to 1.0 or are out of range
    """
    if not (0 < train < 1 and 0 < valid < 1 and 0 < test < 1):
        raise ValueError("All ratios must be between 0 and 1")

    total = train + valid + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0 (got {total:.6f}). "
            f"train={train}, valid={valid}, test={test}"
        )

    return True


def validate_file_exists(path: str | Path, description: str = "File") -> bool:
    """
    Validate that a file exists.

    Args:
        path: File path
        description: Description for error message

    Returns:
        True if file exists

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}. "
            f"Please ensure the file exists or run previous stages first."
        )
    return True


def validate_directory_exists(path: str | Path, description: str = "Directory") -> bool:
    """
    Validate that a directory exists.

    Args:
        path: Directory path
        description: Description for error message

    Returns:
        True if directory exists

    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    path_obj = Path(path)
    if not path_obj.is_dir():
        raise FileNotFoundError(
            f"{description} not found: {path}. "
            f"Please ensure the directory exists or run previous stages first."
        )
    return True


def validate_mode(mode: str) -> bool:
    """
    Validate anomaly mode parameter.

    Args:
        mode: Anomaly mode (a1, a2, a3)

    Returns:
        True if valid

    Raises:
        ValueError: If mode is invalid
    """
    valid_modes = {"a1", "a2", "a3"}
    if mode.lower() not in valid_modes:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of {valid_modes}"
        )
    return True
