from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer


def set_hf_cache(cache_dir: str) -> None:
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)


def load_model(model_id: str, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        device_map="auto",
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def extract_json_text(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_score_payload(answer_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    extracted = extract_json_text(answer_text)
    if extracted is None:
        return None, "No JSON object found in model output"
    try:
        payload = json.loads(extracted)
    except Exception as exc:
        return None, f"JSON parse error: {exc}"
    if not isinstance(payload, dict):
        return None, "Parsed JSON is not an object"
    return payload, "ok"


def validate_score_payload(
    payload: Any,
    *,
    expected_anomaly_type: Optional[str],
    expected_T: Optional[int],
) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload is not a JSON object"

    # Schema:
    # - A1/A2 require K (anomaly count) and its allowed values depend on T
    #   (for slices: K ∈ {T/4, T/2, 3T/4} where T ∈ {12,24,48,72})
    # - A3 does NOT require K
    # - If expected_anomaly_type is None (e.g., review mode), accept either schema
    if expected_anomaly_type in {"A1", "A2"}:
        required_keys: Optional[set[str]] = {"T", "anomaly_type", "K", "scores"}
    elif expected_anomaly_type == "A3":
        required_keys = {"T", "anomaly_type", "scores"}
    else:
        required_keys = None  # accept either (with or without K)
    keys = set(payload.keys())
    if required_keys is not None and keys != required_keys:
        extra = sorted(list(keys - required_keys))
        missing = sorted(list(required_keys - keys))
        return False, f"keys mismatch (missing={missing}, extra={extra})"
    if required_keys is None and keys not in ({"T", "anomaly_type", "scores"}, {"T", "anomaly_type", "K", "scores"}):
        return False, 'keys mismatch (expected {T, anomaly_type, scores} or {T, anomaly_type, K, scores})'

    T = payload.get("T")
    anomaly_type = payload.get("anomaly_type")
    scores = payload.get("scores")
    K = payload.get("K", None)

    if not isinstance(T, int):
        return False, '"T" must be int'
    if T <= 0:
        return False, '"T" must be positive'
    if expected_T is not None and T != expected_T:
        return False, f'"T" mismatch (expected {expected_T}, got {T})'

    if anomaly_type not in {"A1", "A2", "A3"}:
        return False, '"anomaly_type" must be one of A1/A2/A3'
    if expected_anomaly_type is not None and anomaly_type != expected_anomaly_type:
        return False, f'"anomaly_type" mismatch (expected {expected_anomaly_type}, got {anomaly_type})'

    # Validate K if present or required
    if ("K" in keys) or (expected_anomaly_type in {"A1", "A2"}):
        if not isinstance(K, int):
            return False, '"K" must be int'
        base_T = expected_T if expected_T is not None else T
        if isinstance(base_T, int) and base_T > 0 and base_T % 4 == 0:
            allowed = {base_T // 4, base_T // 2, (3 * base_T) // 4}
            if K not in allowed:
                return False, f'"K" must be one of {sorted(list(allowed))} for T={base_T}'

    if not isinstance(scores, list):
        return False, '"scores" must be a list'
    if len(scores) != T:
        return False, f'"scores" length mismatch (expected {T}, got {len(scores)})'
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    zero_count = 0
    for i, x in enumerate(scores):
        if not isinstance(x, (int, float)):
            return False, f'"scores[{i}]" is not a number'
        xf = float(x)
        if xf < 0 or xf > 1:
            return False, f'"scores[{i}]" out of range [0,1]'
        if abs(xf) < 1e-12:
            zero_count += 1
        min_score = xf if (min_score is None or xf < min_score) else min_score
        max_score = xf if (max_score is None or xf > max_score) else max_score

    # Reject degenerate outputs (e.g., all zeros / all constants) which break downstream top-K selection.
    # We only enforce this for A1/A2 where a meaningful ranking is required.
    if expected_anomaly_type in {"A1", "A2"} and (min_score is not None) and (max_score is not None):
        if (max_score - min_score) < 1e-9:
            return False, '"scores" are degenerate (all values are equal); produce a meaningful ranking'
        # If there are more zeros than (T-K), then fewer than K timestamps have non-zero scores.
        # This tends to collapse the downstream top-K selection.
        if isinstance(K, int) and isinstance(T, int) and (zero_count > (T - K)):
            return (
                False,
                f'"scores" have too many zeros: zero_count={zero_count} > (T-K)={T-K} (T={T}, K={K}); produce non-zero scores for at least K timestamps',
            )

    return True, "ok"


def build_repair_user_message(
    *,
    invalid_answer: str,
    reason: str,
    anomaly_type: str,
    expected_T: Optional[int],
) -> str:
    expT = expected_T if expected_T is not None else "(unknown)"
    required_keys = "T, anomaly_type, K, scores" if anomaly_type in {"A1", "A2"} else "T, anomaly_type, scores"
    return (
        "Your previous output was invalid.\n"
        f"- reason: {reason}\n"
        f'- required anomaly_type: "{anomaly_type}"\n'
        f"- required T: {expT}\n\n"
        f"Return JSON ONLY with exactly these keys: {required_keys}.\n"
        "Do not add any other keys. Do not include any explanation.\n\n"
        "CRITICAL: The length of \"scores\" MUST be exactly T.\n"
        "Here was your previous output (invalid):\n"
        "<<<\n"
        f"{invalid_answer}\n"
        ">>>\n\n"
        "Now return the corrected JSON."
    )


def generate_text(*, tokenizer, model, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False
    )
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()


def generate_json_validated(
    *,
    tokenizer,
    model,
    user_query: str,
    anomaly_type: str,
    expected_T: Optional[int],
    max_new_tokens: int,
    max_retries: int,
    log_fn=None,
) -> Tuple[str, Optional[Dict[str, Any]], str, int]:
    from prompts import build_system_prompt

    system_prompt = build_system_prompt(anomaly_type=anomaly_type)
    base_messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    last_answer = ""
    last_reason = ""
    attempt = 0
    while True:
        attempt += 1
        answer = generate_text(messages=base_messages, tokenizer=tokenizer, model=model, max_new_tokens=max_new_tokens)
        last_answer = answer

        payload, parse_reason = parse_score_payload(answer)
        if payload is not None:
            ok, reason = validate_score_payload(
                payload,
                expected_anomaly_type=anomaly_type,
                expected_T=expected_T,
            )
            if ok:
                return answer, payload, "ok", attempt
            last_reason = reason
        else:
            last_reason = parse_reason

        # max_retries < 0 means unlimited retries
        if max_retries >= 0 and attempt > max_retries:
            break

        repair_user = build_repair_user_message(
            invalid_answer=answer,
            reason=last_reason,
            anomaly_type=anomaly_type,
            expected_T=expected_T,
        )
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            {"role": "user", "content": repair_user},
        ]
        if log_fn is not None:
            log_fn(f"[VALIDATION FAILED] attempt={attempt} reason={last_reason} -> retry")

    return last_answer, None, last_reason or "unknown error", attempt
