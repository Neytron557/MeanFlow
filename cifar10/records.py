"""Recover only an uncommitted attempt-log tail, preserving its original bytes."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def recover_attempt_tail(path):
    """Keep all newline-terminated records; archive a crash-truncated final row.

    Call only while the experiment owns the log and no writer is active. A
    valid final JSON object missing only its newline is completed, not removed.
    Returns recovery metadata or None; complete records never change.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as stream:
        stream.seek(-1, 2)
        if stream.read(1) == b"\n":
            return None
        # Attempt rows are small; grow backwards to the preceding newline so
        # a long experiment log is never read fully just to recover its tail.
        end = stream.tell()
        start = max(0, end - 8192)
        while True:
            stream.seek(start)
            chunk = stream.read(end - start)
            separator = chunk.rfind(b"\n")
            if separator >= 0 or start == 0:
                tail_start = start + separator + 1
                tail = chunk[separator + 1:]
                break
            start = max(0, start - 8192)
    stamp = time.time_ns()
    artifact = path.with_name(f"{path.name}.partial-{stamp}.bin")
    artifact.write_bytes(tail)
    try:
        value = json.loads(tail)
        valid = isinstance(value, dict)
    except (ValueError, UnicodeDecodeError):
        valid = False
    if valid:
        with path.open("ab") as stream:
            stream.write(b"\n")
        action = "completed_missing_newline"
    else:
        with path.open("r+b") as stream:
            stream.truncate(tail_start)
        action = "archived_incomplete_tail"
    event = {"event": "attempt_log_tail_recovery", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "action": action, "original_size": end, "tail_start": tail_start,
             "artifact": str(artifact), "tail_sha256": hashlib.sha256(tail).hexdigest()}
    with path.with_name("attempts_recovery.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    return event
