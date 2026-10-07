"""Real coordinator/filesystem fencing; process evidence is not engine evidence."""

import os
from pathlib import Path

import pytest

from cochem.orchestration.sqlite_queue import LeaseLostError, SQLiteTaskQueue


def test_specific_task_claim_and_publication_fence(tmp_path: Path):
    queue = SQLiteTaskQueue(tmp_path / "authority.sqlite")
    first = queue.enqueue_task("file_publication", {"output": "first"})
    second = queue.enqueue_task("file_publication", {"output": "second"})
    lease = queue.lease_task(os.getpid(), "actual-test-process", task_id=second)
    assert lease.task_id == second
    assert queue.get_task(first).state == "PENDING"
    queue.reclaim_orphaned_tasks(timeout_grace_sec=0)
    destination = tmp_path / "published"

    def publish():
        destination.write_text("real file bytes created by the current attempt")

    with pytest.raises(LeaseLostError):
        queue.complete_task_with_publication(
            second,
            {"output": str(destination)},
            publish,
            lease_token=lease.lease_token,
            lease_generation=lease.lease_generation,
        )
    assert not destination.exists()
    current = queue.lease_task(os.getpid(), "actual-test-process", task_id=second)
    queue.complete_task_with_publication(
        second,
        {"output": str(destination)},
        publish,
        lease_token=current.lease_token,
        lease_generation=current.lease_generation,
    )
    assert destination.is_file() and queue.get_task(second).state == "COMPLETED"
    queue.close()


def test_failed_publication_never_claims_completed(tmp_path: Path):
    queue = SQLiteTaskQueue(tmp_path / "authority.sqlite")
    task_id = queue.enqueue_task("exclusive_file", {})
    lease = queue.lease_task(os.getpid(), "actual-test-process", task_id=task_id)
    destination = tmp_path / "already-present"
    destination.write_text("original bytes")

    def publish():
        with destination.open("x") as stream:
            stream.write("cannot overwrite original")

    with pytest.raises(FileExistsError):
        queue.complete_task_with_publication(
            task_id,
            {"output": str(destination)},
            publish,
            lease_token=lease.lease_token,
            lease_generation=lease.lease_generation,
        )
    assert queue.get_task(task_id).state == "RUNNING"
    assert destination.read_text() == "original bytes"
    queue.close()
