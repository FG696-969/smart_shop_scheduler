"""Data loading utilities for the Smart Shop Scheduler.

The project supports the custom 5x4 teaching case and JSPLIB JSSP files
(ft06, la01, la02). All datasets are converted to:

    jobs = [[(machine_id, processing_time), ...], ...]

Machine IDs are stored internally from 0, but displayed as M1, M2, ...
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

Jobs = List[List[Tuple[int, int]]]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CUSTOM_5X4: Jobs = [
    [(0, 4), (1, 3), (3, 5), (2, 2)],
    [(1, 2), (2, 6), (0, 3), (3, 4)],
    [(2, 5), (0, 4), (3, 3), (1, 6)],
    [(3, 3), (1, 5), (2, 4), (0, 2)],
    [(0, 6), (3, 2), (1, 4), (2, 3)],
]

FT06_FALLBACK_TEXT = """# JSPLIB ft06 fallback data
# Fisher and Thompson 6x6 instance, optimum 55
6 6
2 1 0 3 1 6 3 7 5 3 4 6
1 8 2 5 4 10 5 10 0 10 3 4
2 5 3 4 5 8 0 9 1 1 4 7
1 5 0 5 2 5 3 3 4 8 5 9
2 9 1 3 4 5 5 4 0 3 3 1
1 3 3 3 5 9 0 10 4 4 2 1
"""

DEFAULT_METADATA: Dict[str, Dict[str, object]] = {
    "Custom 5x4": {"jobs": 5, "machines": 4, "optimum": None, "description": "Self-built teaching case"},
    "FT06": {"jobs": 6, "machines": 6, "optimum": 55, "description": "Fisher & Thompson 6x6"},
    "LA01": {"jobs": 10, "machines": 5, "optimum": 666, "description": "Lawrence 10x5 instance 1"},
    "LA02": {"jobs": 10, "machines": 5, "optimum": 655, "description": "Lawrence 10x5 instance 2"},
}


def parse_jssp_instance(text: str) -> Tuple[Jobs, int, int]:
    """Parse a JSPLIB-style JSSP instance.

    Expected core block:
        n_jobs n_machines
        machine processing_time machine processing_time ...
    """
    no_comment_lines = [line.split("#", 1)[0] for line in text.splitlines()]
    numbers = [int(v) for v in re.findall(r"-?\d+", "\n".join(no_comment_lines))]
    if len(numbers) < 2:
        numbers = [int(v) for v in re.findall(r"-?\d+", text)]

    for start_idx in range(max(1, len(numbers) - 1)):
        n_jobs = numbers[start_idx]
        n_machines = numbers[start_idx + 1] if start_idx + 1 < len(numbers) else -1
        if not (1 <= n_jobs <= 500 and 1 <= n_machines <= 200):
            continue
        data_start = start_idx + 2
        required = n_jobs * n_machines * 2
        if data_start + required > len(numbers):
            continue
        raw = numbers[data_start : data_start + required]
        jobs: Jobs = []
        cursor = 0
        valid = True
        for _ in range(n_jobs):
            ops: List[Tuple[int, int]] = []
            for _ in range(n_machines):
                machine_id = raw[cursor]
                processing_time = raw[cursor + 1]
                cursor += 2
                if not (0 <= machine_id < n_machines and processing_time > 0):
                    valid = False
                    break
                ops.append((machine_id, processing_time))
            if not valid:
                break
            jobs.append(ops)
        if valid:
            return jobs, n_jobs, n_machines
    raise ValueError("No valid JSSP numeric block found.")


def load_metadata() -> Dict[str, Dict[str, object]]:
    metadata = dict(DEFAULT_METADATA)
    path = DATA_DIR / "instances.json"
    if path.exists():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
            for entry in entries:
                name = str(entry.get("name", "")).upper()
                if name in {"FT06", "LA01", "LA02"}:
                    metadata[name] = {
                        "jobs": entry.get("jobs"),
                        "machines": entry.get("machines"),
                        "optimum": entry.get("optimum"),
                        "description": entry.get("path", "JSPLIB instance"),
                    }
        except Exception:
            pass
    return metadata


def load_dataset(dataset_name: str) -> Tuple[Jobs, Dict[str, object]]:
    """Load dataset by display name."""
    if dataset_name == "Custom 5x4":
        jobs = [list(job) for job in CUSTOM_5X4]
        return jobs, DEFAULT_METADATA["Custom 5x4"]

    normalized = dataset_name.lower()
    path = DATA_DIR / f"{normalized}.txt"
    if not path.exists() and normalized == "ft06":
        text = FT06_FALLBACK_TEXT
    elif path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise FileNotFoundError(f"Missing dataset file: {path}")

    jobs, n_jobs, n_machines = parse_jssp_instance(text)
    metadata = load_metadata().get(dataset_name, {})
    metadata = {**metadata, "jobs": n_jobs, "machines": n_machines}
    return jobs, metadata


def available_datasets() -> List[str]:
    return ["Custom 5x4", "FT06", "LA01", "LA02"]
