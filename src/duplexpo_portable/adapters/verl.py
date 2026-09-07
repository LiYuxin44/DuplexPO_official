"""Dictionary bridge for verl's data/reward and group-index conventions."""

from __future__ import annotations

from ..schema import Window


def to_verl_row(window: Window, prompt="", data_source: str = "duplexpo") -> dict:
    return {
        "data_source": data_source,
        "prompt": prompt,
        "reward_model": {"style": "rule", "ground_truth": window.to_dict()},
        "extra_info": {
            "index": window.window_id,
            "window_id": window.window_id,
            "conversation_id": window.conversation_id,
        },
    }


def attach_verl_group_index(batch: dict, records: list[dict]) -> dict:
    """Add the event ID vector consumed by verl's grouped advantage estimator."""

    result = dict(batch)
    non_tensor = dict(result.get("non_tensor_batch", {}))
    non_tensor["uid"] = [record["group_id"] for record in records]
    non_tensor["duplexpo_record"] = records
    result["non_tensor_batch"] = non_tensor
    return result
