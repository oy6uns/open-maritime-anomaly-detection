from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_arg_value(flag: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch a simple CLI flag value from sys.argv (no argparse dependency)."""
    if flag not in sys.argv:
        return default
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except Exception:
        return default


def log(line: str, log_fh=None) -> None:
    """Write a line to stdout and optionally to an open log file handle."""
    print(line, flush=True)
    if log_fh is not None:
        log_fh.write(line + "\n")
        log_fh.flush()


def sanitize_user_query_text(text: str) -> str:
    """Normalize prompt text loaded from disk (e.g., remove surrounding triple quotes)."""
    # Some query files may store the prompt as a Python triple-quoted literal: """..."""
    t = text.strip()
    if t.startswith('"""'):
        t = t[3:]
    if t.endswith('"""'):
        t = t[:-3]
    return t.strip()


def iter_query_files(query_dir: str) -> List[Path]:
    """List .txt files in a query directory in a deterministic order."""
    p = Path(query_dir)
    files = list(p.glob("*.txt"))

    def _key(x: Path):
        stem = x.stem
        if stem.isdigit():
            return (0, int(stem), stem)
        return (1, stem, stem)

    return sorted(files, key=_key)


def save_output_json(
    answer_text: str,
    *,
    out_dir: str,
    source_name: Optional[str] = None,
    prefix: str = "qwen_output",
) -> str:
    """Persist model output as JSON (best-effort parse; otherwise wrap parse_error/raw_output)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    name_prefix = f"{prefix}_{source_name}_" if source_name else f"{prefix}_"
    target = out_path / f"{name_prefix}{ts}.json"

    start = answer_text.find("{")
    end = answer_text.rfind("}")
    extracted = answer_text[start : end + 1] if (start != -1 and end != -1 and end > start) else None
    if extracted is not None:
        try:
            payload: Dict[str, Any] = json.loads(extracted)
        except Exception as e:
            payload = {"parse_error": str(e), "raw_output": answer_text}
    else:
        payload = {"parse_error": "No JSON object found in model output", "raw_output": answer_text}

    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(target)

