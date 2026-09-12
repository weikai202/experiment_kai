"""Strict provisional and calibrated selections; no per-request adaptation."""
from pathlib import Path
from typing import Annotated, Literal
from pydantic import Field, model_validator
from toolsandbox_pipeline.schemas.memory import Count, Digest, FrozenRecord, Identifier, unique
from toolsandbox_pipeline.schemas.runtime import RoleTokenLimitConfig
from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256

ROLES = ("policy", "critic", "revision")
BOOTSTRAP = (256, 384, 256)


class RoleLimit(FrozenRecord):
    role: Literal["policy", "critic", "revision"]
    output_model_name: Literal["ActionEnvelope", "CriticOutput"]
    max_tokens: Annotated[int, Field(gt=0)]


class ProvisionalTokenLimits(FrozenRecord):
    schema_version: Literal[1]
    status: Literal["provisional"]
    roles: tuple[RoleLimit, ...]

    @model_validator(mode="after")
    def exact(self):
        if tuple((r.role, r.output_model_name, r.max_tokens) for r in self.roles) != tuple(zip(ROLES, ("ActionEnvelope", "CriticOutput", "ActionEnvelope"), BOOTSTRAP)):
            raise ValueError("bootstrap role/schema/ceiling mismatch")
        return self


class DevelopmentTokenLimits(FrozenRecord):
    """Explicit smoke ceilings; no claim of measured formal calibration."""
    schema_version: Literal[1]
    status: Literal["calibration"]
    development_only: Literal[True]
    roles: tuple[RoleLimit, ...]

    @model_validator(mode="after")
    def layout(self):
        if tuple((r.role, r.output_model_name) for r in self.roles) != tuple(zip(ROLES, ("ActionEnvelope", "CriticOutput", "ActionEnvelope"))):
            raise ValueError("exact smoke role/schema order required")
        if any(r.max_tokens % 64 for r in self.roles):
            raise ValueError("smoke ceilings must be multiples of 64")
        return self


class RoleCalibration(FrozenRecord):
    role: Literal["policy", "critic", "revision"]
    bootstrap_max_tokens: Annotated[int, Field(gt=0)]
    attempted_max_tokens: tuple[Annotated[int, Field(gt=0)], ...]
    observed_max_completion_tokens: Count
    recommended_max_tokens: Annotated[int, Field(ge=64)]
    stop_count: Count
    length_count: Count
    request_count: Annotated[int, Field(ge=1)]
    strict_valid_count: Count
    usage_complete: Literal[True]
    prompt_sha256: Digest
    output_schema_sha256: Digest

    @model_validator(mode="after")
    def invariants(self):
        if self.bootstrap_max_tokens != BOOTSTRAP[ROLES.index(self.role)] or not self.attempted_max_tokens:
            raise ValueError("invalid calibration ceiling history")
        if self.recommended_max_tokens % 64 or self.recommended_max_tokens not in self.attempted_max_tokens:
            raise ValueError("final ceiling must be tested and divisible by 64")
        if self.stop_count + self.length_count != self.request_count or self.strict_valid_count != self.stop_count:
            raise ValueError("inconsistent finish/validation counts")
        if self.recommended_max_tokens < recommended_limit(self.observed_max_completion_tokens):
            raise ValueError("insufficient headroom")
        return self


class CalibratedTokenLimits(FrozenRecord):
    schema_version: Literal[1]
    status: Literal["calibrated"]
    calibration_protocol_version: Literal["online-token-calibration-v1"]
    data_seed: Literal[0]
    train_manifest_sha256: Digest
    ordered_calibration_scenario_ids: tuple[Identifier, ...]
    runtime_inputs_sha256: Digest
    qwen_config_sha256: Digest
    calibration_artifact_sha256: Digest
    roles: tuple[RoleCalibration, ...]

    @model_validator(mode="after")
    def invariants(self):
        unique(self.ordered_calibration_scenario_ids)
        if not self.ordered_calibration_scenario_ids or tuple(r.role for r in self.roles) != ROLES:
            raise ValueError("ordered samples and roles required")
        return self


def recommended_limit(observed):
    if type(observed) is not int or observed < 0:
        raise ValueError("actual non-negative completion tokens required")
    return max(64, (((5 * observed + 3) // 4 + 63) // 64) * 64)


def load_token_limits(path, *, expected_sha256):
    from toolsandbox_pipeline.retrieval.index import file_hash
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("absolute non-symlink token config required")
    raw = path.read_bytes()
    if file_hash(raw) != expected_sha256:
        raise ValueError("token config hash mismatch")
    import json
    status = json.loads(raw).get("status")
    model = {"provisional": ProvisionalTokenLimits, "calibration": DevelopmentTokenLimits, "calibrated": CalibratedTokenLimits}.get(status)
    if model is None:
        raise ValueError("unknown token config")
    return model.model_validate_json(raw)


def select_limit(config, role, *, mode, qwen_config_sha256, prompt_sha256, schema_sha256):
    if mode not in ("offline", "calibration", "formal"):
        raise ValueError("invalid role mode")
    if type(config) is DevelopmentTokenLimits:
        if mode == "formal":
            raise ValueError("formal runs require promoted calibration")
        entry = config.roles[ROLES.index(role)]
        return RoleTokenLimitConfig(version="online-smoke-v1", role=role, stage="calibration",
                                   max_tokens=entry.max_tokens,
                                   evidence_manifest_identity=canonical_sha256(config.model_dump(mode="json")))
    if type(config) is ProvisionalTokenLimits:
        if mode == "formal":
            raise ValueError("formal runs require promoted calibration")
        entry = config.roles[ROLES.index(role)]
        return RoleTokenLimitConfig(version="online-bootstrap-v1", role=role, stage="bootstrap", max_tokens=entry.max_tokens)
    if type(config) is not CalibratedTokenLimits:
        raise TypeError("validated token config required")
    entry = config.roles[ROLES.index(role)]
    if (config.qwen_config_sha256, entry.prompt_sha256, entry.output_schema_sha256) != (qwen_config_sha256, prompt_sha256, schema_sha256):
        raise ValueError("calibration invalidated by runtime/prompt/schema change")
    return RoleTokenLimitConfig(version=config.calibration_protocol_version, role=role, stage="calibrated",
                               max_tokens=entry.recommended_max_tokens, evidence_manifest_identity=config.calibration_artifact_sha256)
