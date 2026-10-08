from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .canonical import sha256_json

QWEN_MODEL = "Qwen/Qwen3-32B"
CALIBRATION_PROTOCOL = "bfcl_train_smoke_max_tokens_v1"


def _base_request_config_sha256() -> str:
    return sha256_json({
        "model": QWEN_MODEL,
        "temperature": 0.0,
        "seed": 0,
        "enable_thinking": False,
    })


@dataclass(frozen=True)
class MaxTokenObservation:
    case_id: str
    output_tokens: int
    finish_reason: str


@dataclass(frozen=True)
class MaxTokenCalibrationReceipt:
    protocol: str
    evidence_source: str
    dataset_manifest_sha256: str
    model: str
    base_request_config_sha256: str
    train_case_ids: tuple[str, ...]
    observations: tuple[MaxTokenObservation, ...]
    chosen_max_tokens: int
    receipt_sha256: str

    @staticmethod
    def choose_limit(observations: Iterable[MaxTokenObservation]) -> int:
        rows = tuple(observations)
        if not rows:
            raise ValueError("Token-limit calibration requires observations")
        if any(row.output_tokens < 0 or row.finish_reason not in {"stop", "tool_calls", "length"} for row in rows):
            raise ValueError("Calibration observation is invalid")
        maximum = max(row.output_tokens for row in rows)
        multiplier = 2.0 if any(row.finish_reason == "length" for row in rows) else 1.25
        return max(64, int(math.ceil(maximum * multiplier / 64.0) * 64))

    @classmethod
    def build(
        cls,
        *,
        evidence_source: str,
        dataset_manifest_sha256: str,
        train_case_ids: tuple[str, ...],
        observations: Iterable[MaxTokenObservation],
    ) -> "MaxTokenCalibrationReceipt":
        if evidence_source not in {"live_train_smoke", "synthetic_test"}:
            raise ValueError("Calibration evidence source is invalid")
        rows = tuple(observations)
        if not train_case_ids or tuple(row.case_id for row in rows) != train_case_ids:
            raise ValueError("Calibration observations must follow the declared train-case order")
        core = {
            "protocol": CALIBRATION_PROTOCOL,
            "evidence_source": evidence_source,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "model": QWEN_MODEL,
            "base_request_config_sha256": _base_request_config_sha256(),
            "train_case_ids": train_case_ids,
            "observations": [asdict(row) for row in rows],
            "chosen_max_tokens": cls.choose_limit(rows),
        }
        return cls(
            protocol=core["protocol"],
            evidence_source=evidence_source,
            dataset_manifest_sha256=dataset_manifest_sha256,
            model=QWEN_MODEL,
            base_request_config_sha256=core["base_request_config_sha256"],
            train_case_ids=train_case_ids,
            observations=rows,
            chosen_max_tokens=core["chosen_max_tokens"],
            receipt_sha256=sha256_json(core),
        )

    def validate(self, *, dataset_manifest_sha256: str, allowed_train_case_ids: set[str], require_live: bool) -> None:
        core = {
            "protocol": self.protocol,
            "evidence_source": self.evidence_source,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "model": self.model,
            "base_request_config_sha256": self.base_request_config_sha256,
            "train_case_ids": self.train_case_ids,
            "observations": [asdict(row) for row in self.observations],
            "chosen_max_tokens": self.chosen_max_tokens,
        }
        if self.receipt_sha256 != sha256_json(core):
            raise ValueError("Calibration receipt hash is invalid")
        if self.protocol != CALIBRATION_PROTOCOL or self.model != QWEN_MODEL:
            raise ValueError("Calibration protocol or model is invalid")
        if self.base_request_config_sha256 != _base_request_config_sha256():
            raise ValueError("Calibration base request configuration is invalid")
        if self.dataset_manifest_sha256 != dataset_manifest_sha256:
            raise ValueError("Calibration dataset manifest does not match")
        if tuple(row.case_id for row in self.observations) != self.train_case_ids or len(set(self.train_case_ids)) != len(self.train_case_ids):
            raise ValueError("Calibration observations must bind unique ordered train cases")
        if not set(self.train_case_ids) <= allowed_train_case_ids:
            raise ValueError("Calibration contains non-train case IDs")
        if self.chosen_max_tokens != self.choose_limit(self.observations):
            raise ValueError("Calibrated max_tokens does not follow the frozen rule")
        if require_live and self.evidence_source != "live_train_smoke":
            raise ValueError("Formal execution requires a live train-smoke calibration receipt")


def calibration_to_payload(receipt: MaxTokenCalibrationReceipt) -> dict[str, Any]:
    receipt.validate(
        dataset_manifest_sha256=receipt.dataset_manifest_sha256,
        allowed_train_case_ids=set(receipt.train_case_ids),
        require_live=False,
    )
    return asdict(receipt)


def calibration_from_payload(payload: Mapping[str, Any]) -> MaxTokenCalibrationReceipt:
    row = dict(payload)
    row["train_case_ids"] = tuple(row["train_case_ids"])
    row["observations"] = tuple(MaxTokenObservation(**item) for item in row["observations"])
    receipt = MaxTokenCalibrationReceipt(**row)
    receipt.validate(
        dataset_manifest_sha256=receipt.dataset_manifest_sha256,
        allowed_train_case_ids=set(receipt.train_case_ids),
        require_live=False,
    )
    return receipt


@dataclass(frozen=True)
class QwenRequestConfig:
    max_tokens: int
    model: str = QWEN_MODEL
    temperature: float = 0.0
    seed: int = 0

    def as_openai_kwargs(self, calibration: MaxTokenCalibrationReceipt) -> dict[str, Any]:
        if self.model != QWEN_MODEL or self.temperature != 0.0 or self.seed != 0:
            raise ValueError("Formal Qwen identity and deterministic decoding are fixed")
        if self.max_tokens <= 0 or self.max_tokens != calibration.chosen_max_tokens:
            raise ValueError("max_tokens must equal the bound calibration receipt")
        return {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }


def verify_nonthinking_request(kwargs: dict[str, Any]) -> None:
    if kwargs.get("model") != QWEN_MODEL:
        raise ValueError("Unexpected model")
    if kwargs.get("extra_body", {}).get("chat_template_kwargs") != {"enable_thinking": False}:
        raise ValueError("Qwen non-thinking template was not requested")
