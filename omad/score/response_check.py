"""
Review (and optionally fix) saved outputs.

Usage examples:

  # Validate all outputs in out_dir; print summary
  python qwen_review.py --batch-root /workspace/NAS/KRISO2026 --out-dir /workspace/Local/outputs

  # Fix invalid ones by re-running model and saving new JSONs with prefix qwen_review
  python qwen_review.py --batch-root /workspace/NAS/KRISO2026 --out-dir /workspace/Local/outputs --fix
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from infer import infer_anomaly_type_from_filename, infer_expected_T_from_path
from llm_core import generate_json_validated, load_model, set_hf_cache, validate_score_payload
from logging_utils import get_arg_value, sanitize_user_query_text, save_output_json

CACHE_DIR = "/nas/home/oy6uns"
MODEL_ID = "Qwen/Qwen3-8B"
set_hf_cache(CACHE_DIR)

def _parse_source_name_from_output_filename(name: str, prefix: str) -> Optional[str]:
    """Extract the original query basename from an output filename."""
    # Filename format: "{prefix}_{source_name}_{YYYYMMDD}-{HHMMSS}-{ffffff}.json"
    # source_name itself can include underscores, so we capture minimally up to timestamp.
    m = re.match(rf"^{re.escape(prefix)}_(.+)_(\d{{8}}-\d{{6}}-\d{{6}})\.json$", name)
    if not m:
        return None
    return m.group(1)


def _find_query_file(batch_root: str, source_name: str) -> Optional[Path]:
    """Find the matching query .txt file under batch_root by basename."""
    root = Path(batch_root)
    hits = list(root.glob(f"**/user_query/{source_name}.txt"))
    if not hits:
        return None
    # Prefer shortest path if multiple.
    return sorted(hits, key=lambda p: len(str(p)))[0]


def _load_json(path: Path):
    """Load a JSON file, returning None on failure."""
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def main() -> int:
    """CLI entrypoint to validate outputs and optionally re-run invalid ones."""
    batch_root = get_arg_value("--batch-root", None)
    out_dir = get_arg_value("--out-dir", None)
    prefix = get_arg_value("--prefix", "qwen_output") or "qwen_output"
    fix = "--fix" in sys.argv

    max_new_tokens_s = get_arg_value("--max-new-tokens", None)
    max_new_tokens = int(max_new_tokens_s) if (max_new_tokens_s and max_new_tokens_s.isdigit()) else 2048
    max_retries_s = get_arg_value("--max-retries", None)
    max_retries = int(max_retries_s) if (max_retries_s and max_retries_s.isdigit()) else 2

    if out_dir is None:
        print("[오류] --out-dir 는 필수입니다.", flush=True)
        return 2
    if fix and batch_root is None:
        print("[오류] --fix 를 쓰려면 --batch-root 가 필요합니다.", flush=True)
        return 2

    outp = Path(out_dir)
    files = sorted(outp.glob(f"{prefix}_*.json"))
    print(f"[리뷰] out_dir={out_dir} prefix={prefix} files={len(files)} fix={fix}", flush=True)
    tokenizer = None
    model = None
    if fix:
        tokenizer, model = load_model(MODEL_ID, CACHE_DIR)

    ok_cnt = 0
    bad_cnt = 0
    fixed_cnt = 0
    missing_query_cnt = 0

    for fp in files:
        payload = _load_json(fp)
        if payload is None:
            bad = True
            reason = "cannot load json"
        else:
            # qwen output may be wrapper with parse_error/raw_output.
            ok, reason = validate_score_payload(payload, expected_anomaly_type=None, expected_T=None)
            bad = not ok

        if not bad:
            ok_cnt += 1
            continue

        bad_cnt += 1
        source_name = _parse_source_name_from_output_filename(fp.name, prefix=prefix)
        print(f"[BAD] {fp.name} reason={reason} source={source_name}", flush=True)

        if not fix:
            continue
        if source_name is None:
            continue
        qf = _find_query_file(batch_root=batch_root, source_name=source_name)
        if qf is None:
            missing_query_cnt += 1
            print(f"  [SKIP] query not found for {source_name}", flush=True)
            continue

        raw = qf.read_text(encoding="utf-8", errors="replace")
        user_query = sanitize_user_query_text(raw)
        if not user_query:
            print(f"  [SKIP] empty query file: {qf}", flush=True)
            continue

        anomaly_type = infer_anomaly_type_from_filename(qf)
        expected_T = infer_expected_T_from_path(qf)
        if anomaly_type is None:
            print(f"  [SKIP] cannot infer anomaly type from: {qf.name}", flush=True)
            continue

        answer, payload2, status, attempts = generate_json_validated(
            tokenizer=tokenizer,
            model=model,
            user_query=user_query,
            anomaly_type=anomaly_type,
            expected_T=expected_T,
            max_new_tokens=max_new_tokens,
            max_retries=max_retries,
        )
        saved = save_output_json(answer, out_dir=out_dir, source_name=source_name, prefix="response_check")
        print(f"  [FIXED] status={status} attempts={attempts} saved={saved}", flush=True)
        fixed_cnt += 1

    print(
        f"[요약] ok={ok_cnt} bad={bad_cnt} fixed={fixed_cnt} missing_query={missing_query_cnt}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

