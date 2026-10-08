"""Audited train/dev access boundary; ordinary code has no test-loading path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .canonical import canonical_sha256
from .manifests import EVOLUTION_PROTOCOL


@dataclass(frozen=True)
class TaskRef:
    domain: str
    task_id: str


@dataclass(frozen=True)
class AccessAudit:
    split: str
    purpose: str
    phase: str
    manifest_sha256: str
    task_refs_sha256: str
    task_count: int


def _task_identity(task: object) -> TaskRef:
    if isinstance(task, tuple) and len(task) >= 2:
        return TaskRef(str(task[0]), str(task[1]))
    if isinstance(task, dict):
        domain = task.get("domain")
        task_id = task.get("task_id", task.get("id"))
    else:
        domain = getattr(task, "domain", None)
        task_id = getattr(task, "task_id", getattr(task, "id", None))
    if not isinstance(domain, str) or not isinstance(task_id, str):
        raise ValueError("native loader result does not expose domain and task identity")
    return TaskRef(domain, task_id)


class OrdinaryDatasetAccess:
    """Expose train and restricted dev; sealed test is intentionally absent."""

    def __init__(self, manifest: dict, loader: Callable[[str, Iterable[str]], list[object]]):
        if manifest.get("protocol") != EVOLUTION_PROTOCOL:
            raise ValueError("evolution access requires the 144/34 protocol")
        self._manifest = manifest
        self._loader = loader
        self.audits: list[AccessAudit] = []

    def load(
        self,
        *,
        split: str = "train",
        purpose: str = "development",
        phase: str,
        round_index: int | None = None,
    ) -> tuple[list[object], AccessAudit]:
        if split == "train":
            if purpose not in {"development", "evolution"} or round_index not in (0, 1, 2):
                raise PermissionError("train requires evolution/development and round 0, 1, or 2")
            refs = [
                TaskRef(domain, task_id)
                for domain, values in self._manifest["domains"].items()
                for task_id in values["rounds"][round_index]
            ]
        elif split == "dev":
            if purpose != "skill_ab_validation" or round_index is not None:
                raise PermissionError("dev is restricted to paired Skill A/B validation")
            refs = [
                TaskRef(domain, task_id)
                for domain, values in self._manifest["domains"].items()
                for task_id in values["dev"]
            ]
        else:
            raise PermissionError("ordinary access cannot load sealed test")
        audit = AccessAudit(
            split=split,
            purpose=purpose,
            phase=phase,
            manifest_sha256=self._manifest["manifest_sha256"],
            task_refs_sha256=canonical_sha256(
                [{"domain": ref.domain, "task_id": ref.task_id} for ref in refs]
            ),
            task_count=len(refs),
        )
        # The permission decision and audit object exist before content is returned.
        tasks: list[object] = []
        for domain in dict.fromkeys(ref.domain for ref in refs):
            ids = [ref.task_id for ref in refs if ref.domain == domain]
            tasks.extend(self._loader(domain, ids))
        if len(tasks) != len(refs):
            raise ValueError("native loader did not return the exact selected task count")
        actual_refs = [_task_identity(task) for task in tasks]
        if actual_refs != refs:
            raise ValueError("native loader did not return exact manifest-ordered task identities")
        self.audits.append(audit)
        return tasks, audit
