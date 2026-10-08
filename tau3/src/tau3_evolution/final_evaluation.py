"""Frozen one-time Vanilla/G000/G003 evaluation plan for official tau3 test."""

from __future__ import annotations

import fcntl
from dataclasses import dataclass
from pathlib import Path

from . import DOMAINS, EMBEDDING_MODEL, QWEN_MODEL, SYSTEM_ORDER, USER_SIMULATOR_MODEL
from .canonical import atomic_write_json, canonical_sha256, read_json
from .generation_store import generation_state_from_document, validate_generation_state
from .manifests import EVOLUTION_PROTOCOL, validate_manifest
from .model_boundary import OutputLimitCalibration
from .seed_library import verify_seed_library

VANILLA_NO_GENERATION_SHA256 = canonical_sha256({"kind": "no_generation", "generation_id": None})


@dataclass(frozen=True)
class TaskTrialCluster:
    domain: str
    task_id: str
    simulator_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS or not self.task_id:
            raise ValueError("invalid task cluster")
        if not self.simulator_seeds or any(type(seed) is not int for seed in self.simulator_seeds):
            raise ValueError("at least one integer simulator seed is required")
        if len(set(self.simulator_seeds)) != len(self.simulator_seeds):
            raise ValueError("duplicate simulator seed within task cluster")


@dataclass(frozen=True)
class ExecutionBinding:
    output_limit_calibration_sha256: str
    seed_library_sha256: str
    qwen_model: str
    qwen_max_tokens: int
    qwen_decoding_sha256: str
    user_simulator_model: str
    user_simulator_config_sha256: str
    embedding_model: str
    embedding_config_sha256: str
    native_runtime_sha256: str
    evaluator_sha256: str
    vanilla_generation_sha256: str
    generation_0_sha256: str
    updated_generation_sha256: str

    def __post_init__(self) -> None:
        if self.qwen_model != QWEN_MODEL or self.qwen_max_tokens <= 0:
            raise ValueError(
                "final binding requires fixed Qwen/Qwen3-32B and calibrated max tokens"
            )
        expected_qwen = canonical_sha256(
            {
                "model": QWEN_MODEL,
                "temperature": 0.0,
                "seed": 0,
                "max_tokens": self.qwen_max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        if self.qwen_decoding_sha256 != expected_qwen:
            raise ValueError("final Qwen decoding binding mismatch")
        if self.user_simulator_model != USER_SIMULATOR_MODEL:
            raise ValueError("final user-simulator model mismatch")
        if self.embedding_model != EMBEDDING_MODEL:
            raise ValueError("final embedding model mismatch")
        expected_embedding = canonical_sha256(
            {"model": EMBEDDING_MODEL, "encoding_format": "float", "fallback": None}
        )
        if self.embedding_config_sha256 != expected_embedding:
            raise ValueError("final embedding configuration mismatch")
        if self.vanilla_generation_sha256 != VANILLA_NO_GENERATION_SHA256:
            raise ValueError("final Vanilla binding requires the no-generation sentinel")
        for value in (
            self.output_limit_calibration_sha256,
            self.seed_library_sha256,
            self.user_simulator_config_sha256,
            self.native_runtime_sha256,
            self.evaluator_sha256,
            self.vanilla_generation_sha256,
            self.generation_0_sha256,
            self.updated_generation_sha256,
        ):
            if not value.startswith("sha256:"):
                raise ValueError("final environment/generation bindings require content hashes")

    @property
    def binding_sha256(self) -> str:
        return canonical_sha256(self.__dict__)


@dataclass(frozen=True)
class FinalEvaluationPlan:
    evolution_manifest_sha256: str
    sealed_test_manifest_sha256: str
    generation_0_id: str
    updated_generation_id: str
    systems: tuple[str, ...]
    task_clusters: tuple[TaskTrialCluster, ...]
    execution_binding: ExecutionBinding

    def __post_init__(self) -> None:
        if (
            self.generation_0_id != "g000"
            or self.updated_generation_id != "g003"
            or self.systems != SYSTEM_ORDER
        ):
            raise ValueError("final comparison is fixed to Vanilla/G000/G003")

    @property
    def plan_sha256(self) -> str:
        return canonical_sha256(
            {
                "evolution_manifest_sha256": self.evolution_manifest_sha256,
                "sealed_test_manifest_sha256": self.sealed_test_manifest_sha256,
                "generation_0_id": self.generation_0_id,
                "updated_generation_id": self.updated_generation_id,
                "systems": list(self.systems),
                "task_clusters": [
                    {
                        "domain": cluster.domain,
                        "task_id": cluster.task_id,
                        "simulator_seeds": list(cluster.simulator_seeds),
                    }
                    for cluster in self.task_clusters
                ],
                "execution_binding_sha256": self.execution_binding.binding_sha256,
            }
        )


def _verified_generation_chain(
    documents: tuple[dict, dict, dict, dict], seed_sha256: str
) -> tuple[str, str]:
    parent = None
    states = []
    for index, document in enumerate(documents):
        state = generation_state_from_document(document)
        expected_id = f"g{index:03d}"
        if state.generation_id != expected_id:
            raise ValueError("final generation chain ID mismatch")
        if state.seed_library_sha256 != seed_sha256:
            raise ValueError("final generation does not carry the verified seed library")
        validate_generation_state(state, parent=parent)
        states.append(state)
        parent = state
    return states[0].state_sha256, states[-1].state_sha256


def build_final_plan(
    manifest: dict,
    sealed: dict,
    *,
    calibration: OutputLimitCalibration,
    seed_library: dict,
    user_simulator_config_sha256: str,
    native_runtime_sha256: str,
    evaluator_sha256: str,
    generation_0: dict,
    generation_1: dict,
    generation_2: dict,
    updated_generation: dict,
    simulator_seeds: tuple[int, ...] = (0,),
) -> FinalEvaluationPlan:
    validate_manifest(manifest, sealed)
    calibration.validate_for_manifest(manifest)
    verify_seed_library(seed_library)
    if manifest["protocol"] != EVOLUTION_PROTOCOL:
        raise ValueError("final comparison requires the evolution144/dev34 protocol")
    generation_0_sha256, updated_generation_sha256 = _verified_generation_chain(
        (generation_0, generation_1, generation_2, updated_generation),
        seed_library["library_sha256"],
    )
    clusters = tuple(
        TaskTrialCluster(domain, task_id, simulator_seeds)
        for domain in DOMAINS
        for task_id in sealed["domains"][domain]
    )
    if len(clusters) != 100 or len({(row.domain, row.task_id) for row in clusters}) != 100:
        raise ValueError("final plan requires the exact 100 official test tasks")
    return FinalEvaluationPlan(
        evolution_manifest_sha256=manifest["manifest_sha256"],
        sealed_test_manifest_sha256=sealed["manifest_sha256"],
        generation_0_id="g000",
        updated_generation_id="g003",
        systems=SYSTEM_ORDER,
        task_clusters=clusters,
        execution_binding=ExecutionBinding(
            output_limit_calibration_sha256=calibration.receipt_sha256,
            seed_library_sha256=seed_library["library_sha256"],
            qwen_model=QWEN_MODEL,
            qwen_max_tokens=calibration.chosen_max_tokens,
            qwen_decoding_sha256=calibration.qwen_config_sha256,
            user_simulator_model=USER_SIMULATOR_MODEL,
            user_simulator_config_sha256=user_simulator_config_sha256,
            embedding_model=EMBEDDING_MODEL,
            embedding_config_sha256=canonical_sha256(
                {"model": EMBEDDING_MODEL, "encoding_format": "float", "fallback": None}
            ),
            native_runtime_sha256=native_runtime_sha256,
            evaluator_sha256=evaluator_sha256,
            vanilla_generation_sha256=VANILLA_NO_GENERATION_SHA256,
            generation_0_sha256=generation_0_sha256,
            updated_generation_sha256=updated_generation_sha256,
        ),
    )


class FinalTestOnceGuard:
    """Global sealed-test claim; one execution owner may resume but no other may rerun."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        if lock_path.is_symlink() or self.path.is_symlink():
            raise ValueError("final-test ledger paths cannot be symlinks")
        return lock_path.open("a+", encoding="utf-8")

    def claim(self, plan: FinalEvaluationPlan) -> None:
        """Legacy strict claim used by protocol checks; it never resumes."""
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists() and read_json(self.path).get("status") in {
                "claimed",
                "running",
                "complete",
            }:
                raise RuntimeError("final test has already been claimed")
            atomic_write_json(
                self.path,
                {
                    "plan_sha256": plan.plan_sha256,
                    "system_order": list(SYSTEM_ORDER),
                    "status": "claimed",
                },
            )

    def claim_or_resume(self, plan: FinalEvaluationPlan, execution_owner_sha256: str) -> str:
        if not execution_owner_sha256.startswith("sha256:"):
            raise ValueError("final execution owner requires a durable fingerprint")
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            expected = {
                "plan_sha256": plan.plan_sha256,
                "system_order": list(SYSTEM_ORDER),
                "execution_owner_sha256": execution_owner_sha256,
            }
            if self.path.exists():
                existing = read_json(self.path)
                if any(existing.get(key) != value for key, value in expected.items()):
                    raise RuntimeError("final test has already been claimed by another execution")
                if existing.get("status") not in {"claimed", "running", "complete"}:
                    raise RuntimeError("invalid final-test guard state")
                return existing["status"]
            atomic_write_json(self.path, {**expected, "status": "claimed"})
            return "claimed"

    def mark_running(self, plan: FinalEvaluationPlan, execution_owner_sha256: str) -> None:
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = read_json(self.path)
            expected = {
                "plan_sha256": plan.plan_sha256,
                "system_order": list(SYSTEM_ORDER),
                "execution_owner_sha256": execution_owner_sha256,
            }
            if any(existing.get(key) != value for key, value in expected.items()):
                raise RuntimeError("final-test guard does not belong to this execution")
            if existing.get("status") == "complete":
                return
            atomic_write_json(self.path, {**expected, "status": "running"})

    def complete_resumable(
        self, plan: FinalEvaluationPlan, execution_owner_sha256: str, report_sha256: str
    ) -> None:
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = read_json(self.path)
            expected = {
                "plan_sha256": plan.plan_sha256,
                "system_order": list(SYSTEM_ORDER),
                "execution_owner_sha256": execution_owner_sha256,
            }
            if any(existing.get(key) != value for key, value in expected.items()):
                raise RuntimeError("final-test guard does not belong to this execution")
            complete = {**expected, "report_sha256": report_sha256, "status": "complete"}
            if existing.get("status") == "complete" and existing != complete:
                raise RuntimeError("conflicting completed final-test report")
            atomic_write_json(self.path, complete)

    def validate_completed(
        self,
        plan: FinalEvaluationPlan,
        execution_owner_sha256: str,
        report_sha256: str,
    ) -> None:
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = read_json(self.path)
            expected = {
                "plan_sha256": plan.plan_sha256,
                "system_order": list(SYSTEM_ORDER),
                "execution_owner_sha256": execution_owner_sha256,
                "report_sha256": report_sha256,
                "status": "complete",
            }
            if existing != expected:
                raise RuntimeError("completed final-test guard report mismatch")

    def complete(self, plan: FinalEvaluationPlan, report_sha256: str) -> None:
        """Complete a legacy strict claim."""
        existing = read_json(self.path)
        if existing != {
            "plan_sha256": plan.plan_sha256,
            "system_order": list(SYSTEM_ORDER),
            "status": "claimed",
        }:
            raise RuntimeError("final test ledger is not the matching claimed plan")
        atomic_write_json(
            self.path,
            {
                "plan_sha256": plan.plan_sha256,
                "system_order": list(SYSTEM_ORDER),
                "report_sha256": report_sha256,
                "status": "complete",
            },
        )
