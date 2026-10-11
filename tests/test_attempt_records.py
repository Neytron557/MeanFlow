import json
from pathlib import Path

from cifar10.records import recover_attempt_tail


def test_partial_tail_archived_without_changing_complete_records(tmp_path):
    path = tmp_path / "attempts.jsonl"
    complete = b'{"committed":true,"step":1}\n'
    tail = b'{"committed":tr'
    path.write_bytes(complete + tail)
    event = recover_attempt_tail(path)
    assert path.read_bytes() == complete
    assert Path(event["artifact"]).read_bytes() == tail
    with path.open("a") as stream:
        stream.write('{"committed":true,"step":2}\n')
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [1, 2]
    assert recover_attempt_tail(path) is None


def test_complete_json_missing_newline_is_retained(tmp_path):
    path = tmp_path / "attempts.jsonl"
    content = b'{"committed":true,"step":1}'
    path.write_bytes(content)
    event = recover_attempt_tail(path)
    assert event["action"] == "completed_missing_newline"
    assert path.read_bytes() == content + b"\n"
