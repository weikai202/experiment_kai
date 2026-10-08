"""Fail-closed model request boundary shared by every pipeline role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import DOMAINS, EMBEDDING_MODEL, QWEN_MODEL, USER_SIMULATOR_MODEL
from .canonical import canonical_sha256, verify_content_hash
from .manifests import EVOLUTION_PROTOCOL


@dataclass(frozen=True)
class ModelProfile:
    qwen_model: str = QWEN_MODEL
    user_simulator_model: str = USER_SIMULATOR_MODEL
    embedding_model: str = EMBEDDING_MODEL
    temperature: float = 0.0
    seed: int = 0

    def validate(self) -> None:
        if self.qwen_model != QWEN_MODEL:
            raise ValueError("pipeline model must be Qwen/Qwen3-32B")
        if self.user_simulator_model != USER_SIMULATOR_MODEL:
            raise ValueError("user simulator model mismatch")
        if self.embedding_model != EMBEDDING_MODEL:
            raise ValueError("embedding model mismatch; no fallback is allowed")
        if self.temperature != 0.0 or self.seed != 0:
            raise ValueError("formal model profile requires temperature=0 and seed=0")


@dataclass(frozen=True)
class CalibrationSample:
    domain: str
    task_id: str
    observed_output_tokens: int
    finish_reason: str


@dataclass(frozen=True)
class OutputLimitCalibration:
    source_kind: str
    evolution_manifest_sha256: str
    train_task_refs: tuple[tuple[str, str], ...]
    samples: tuple[CalibrationSample, ...]
    chosen_max_tokens: int
    model: str
    runtime_sha256: str

    def validate(self) -> None:
        if self.source_kind != "real_train_smoke":
            raise ValueError("formal max tokens require a real train-only smoke receipt")
        if not self.evolution_manifest_sha256.startswith("sha256:"):
            raise ValueError("calibration must bind the evolution manifest")
        if self.model != QWEN_MODEL or not self.runtime_sha256.startswith("sha256:"):
            raise ValueError("calibration model/runtime mismatch")
        if not self.train_task_refs or len(set(self.train_task_refs)) != len(self.train_task_refs):
            raise ValueError("calibration requires unique train task references")
        allowed = set(self.train_task_refs)
        if not self.samples:
            raise ValueError("calibration requires observed train-smoke samples")
        for row in self.samples:
            if (row.domain, row.task_id) not in allowed or row.domain not in DOMAINS:
                raise ValueError("calibration sample is outside its declared train-only tasks")
            if type(row.observed_output_tokens) is not int or row.observed_output_tokens < 0:
                raise ValueError("invalid observed output token count")
            if row.finish_reason not in {"stop", "length", "tool", "other"}:
                raise ValueError("unknown calibration finish reason")
        sample_refs = [(row.domain, row.task_id) for row in self.samples]
        if len(sample_refs) != len(set(sample_refs)) or set(sample_refs) != allowed:
            raise ValueError("calibration requires exactly one sample per declared train task")
        if any(row.finish_reason == "length" for row in self.samples):
            raise ValueError("truncated calibration samples must be rerun before formal use")
        observed_max = max(row.observed_output_tokens for row in self.samples)
        headroom = max(64, (observed_max + 4) // 5)
        expected_limit = ((observed_max + headroom + 63) // 64) * 64
        if type(self.chosen_max_tokens) is not int or self.chosen_max_tokens != expected_limit:
            raise ValueError(
                "chosen max tokens must equal the deterministic observed-max headroom rule"
            )

    def validate_for_manifest(self, manifest: dict[str, Any]) -> None:
        self.validate()
        verify_content_hash(manifest)
        if manifest.get("protocol") != EVOLUTION_PROTOCOL:
            raise ValueError("calibration requires the evolution144/dev34 protocol")
        if self.evolution_manifest_sha256 != manifest.get("manifest_sha256"):
            raise ValueError("calibration receipt uses a different evolution manifest")
        train_refs = {
            (domain, task_id)
            for domain in DOMAINS
            for shard in manifest["domains"][domain]["rounds"]
            for task_id in shard
        }
        if not set(self.train_task_refs) <= train_refs:
            raise ValueError("calibration includes a Dev, test, or unknown task reference")

    @property
    def qwen_config_sha256(self) -> str:
        self.validate()
        return canonical_sha256(
            {
                "model": QWEN_MODEL,
                "temperature": 0.0,
                "seed": 0,
                "max_tokens": self.chosen_max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return canonical_sha256(
            {
                "source_kind": self.source_kind,
                "evolution_manifest_sha256": self.evolution_manifest_sha256,
                "train_task_refs": [list(row) for row in self.train_task_refs],
                "samples": [
                    {
                        "domain": row.domain,
                        "task_id": row.task_id,
                        "observed_output_tokens": row.observed_output_tokens,
                        "finish_reason": row.finish_reason,
                    }
                    for row in self.samples
                ],
                "chosen_max_tokens": self.chosen_max_tokens,
                "model": self.model,
                "runtime_sha256": self.runtime_sha256,
                "qwen_config_sha256": self.qwen_config_sha256,
            }
        )


def qwen_request(messages: list[dict[str, Any]], *, max_tokens: int) -> dict[str, Any]:
    if not messages or max_tokens <= 0:
        raise ValueError("non-empty messages and positive max_tokens required")
    return {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def validate_qwen_request(request: dict[str, Any]) -> None:
    if request.get("model") != QWEN_MODEL:
        raise ValueError("wrong Qwen model")
    if request.get("chat_template_kwargs") != {"enable_thinking": False}:
        raise ValueError("Qwen thinking must be explicitly disabled")
    if request.get("temperature") != 0.0 or request.get("seed") != 0:
        raise ValueError("formal Qwen requests require temperature=0 and seed=0")
    if "tools" in request or "tool_choice" in request:
        raise ValueError("native model tool calling is outside the pipeline contract")
