from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def infer_anomaly_type_from_filename(fp: Path) -> Optional[str]:
    """Infer anomaly type (A1/A2/A3) from a query filename."""
    # Expected: "..._A1.txt" / "..._A2.txt" / "..._A3.txt"
    m = re.search(r"(?:^|[_\-])(A[123])(?:$|[_\-])", fp.stem)
    if m:
        return m.group(1)
    m2 = re.search(r"(A[123])", fp.name)
    return m2.group(1) if m2 else None


def infer_expected_T_from_path(fp: Path) -> Optional[int]:
    """Infer expected segment length T from the directory name (e.g., route_sliced_12)."""
    # e.g., ".../route_sliced_12/user_query/route_123_A3.txt"
    for part in fp.parts:
        m = re.match(r"route_sliced_(\d+)$", part)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None

