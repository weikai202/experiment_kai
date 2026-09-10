"""Deterministic train sample and ceiling search over captured real requests.

Episode execution, sealing the corpus, and promotion remain coordinator-owned.
Callbacks supply prepared ledger identities and one-call runners; this module
never loads a scenario, creates a client, or executes a tool.
"""
from hashlib import sha256
from time import monotonic
from typing import Literal
from pydantic import model_validator
from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.memory import Digest, FrozenRecord, Identifier
from toolsandbox_pipeline.schemas.runtime import RoleTokenLimitConfig
from .prompt_builder import fingerprint
from .prompt_contracts import PreparedRoleRequest, RoleCallResult
from .token_limits import BOOTSTRAP, ROLES, RoleCalibration, CalibratedTokenLimits, recommended_limit

PROTOCOL = "online-token-calibration-v1"
VARIANTS = ("no_distraction", "three_distraction_tools", "ten_distraction_tools", "all_tools",
            "three_distraction_tool_description_scrambled", "three_distraction_argument_type_scrambled",
            "three_distraction_argument_description_scrambled", "three_distraction_tool_name_scrambled")


class CalibrationScenario(FrozenRecord):
    scenario_id: Identifier
    variant: str
    split: Literal["train"]

    @model_validator(mode="after")
    def known_variant(self):
        if self.variant not in VARIANTS:
            raise ValueError("unknown calibration variant")
        return self


class CalibrationSample(FrozenRecord):
    train_manifest_sha256: Digest
    initial: tuple[Identifier, ...]
    reserve: tuple[Identifier, ...]

    @model_validator(mode="after")
    def counts(self):
        if len(self.initial) != 32 or len(self.reserve) != 32 or len(set(self.initial + self.reserve)) != 64:
            raise ValueError("exactly 32 initial and 32 disjoint reserve scenarios required")
        return self


class CalibrationRuntimeInputs(FrozenRecord):
    generation_manifest_sha256: Digest
    embedding_config_sha256: Digest
    user_simulator_model: Literal["gpt-4o-mini-2024-07-18"]
    user_simulator_prompt_sha256: Digest
    controller_config_sha256: Digest
    tool_schema_manifest_sha256: Digest
    fixture_store_manifest_sha256: Digest
    world_clock_config_sha256: Digest
    online_orchestrator_version: Identifier


class CapturedCalibrationRequest(FrozenRecord):
    scenario_id: Identifier
    scenario_family_id: Identifier
    prepared: PreparedRoleRequest


def select_calibration_sample(records: tuple[CalibrationScenario, ...], train_manifest_sha256):
    if any(type(r) is not CalibrationScenario for r in records) or len({r.scenario_id for r in records}) != len(records):
        raise ValueError("unique validated train records required")
    initial, reserve = set(), set()
    for variant in VARIANTS:
        pool = [r.scenario_id for r in records if r.variant == variant]
        pool.sort(key=lambda sid: (sha256(("0\0" + PROTOCOL + "\0" + sid).encode()).digest(), sid.encode()))
        if len(pool) < 8:
            raise ValueError("each variant requires eight train scenarios")
        initial.update(pool[:4])
        reserve.update(pool[4:8])
    return CalibrationSample(train_manifest_sha256=train_manifest_sha256,
        initial=tuple(r.scenario_id for r in records if r.scenario_id in initial),
        reserve=tuple(r.scenario_id for r in records if r.scenario_id in reserve))


def with_ceiling(prepared, ceiling, qwen, corpus_identity):
    if type(prepared) is not PreparedRoleRequest or type(ceiling) is not int or ceiling <= 0:
        raise ValueError("prepared request and positive ceiling required")
    if qwen.output_limit is not None and ceiling > qwen.output_limit:
        raise ValueError("calibration exceeds server hard ceiling")
    selection = RoleTokenLimitConfig(version=PROTOCOL, role=prepared.role, stage="calibration", max_tokens=ceiling,
                                     evidence_manifest_identity=corpus_identity)
    data = prepared.model_dump()
    data.update(max_tokens=ceiling, selected_limit=selection, token_limit_config_status="calibration",
        token_limit_config_sha256=canonical_sha256(selection.model_dump(mode="json")),
        canonical_input_fingerprint=fingerprint(prepared.role, prepared.messages, prepared.output_schema_sha256, ceiling, qwen))
    return PreparedRoleRequest.model_validate(data)


def calibrate_role(corpus, *, qwen, corpus_identity, hard_ceiling, invoke):
    if type(corpus) is not tuple or len(corpus) < 32 or any(type(p) is not PreparedRoleRequest for p in corpus):
        raise ValueError("sealed corpus requires at least 32 captured role requests")
    role = corpus[0].role
    if any((p.role, p.prompt_sha256, p.output_schema_sha256, p.generation_id) !=
           (role, corpus[0].prompt_sha256, corpus[0].output_schema_sha256, corpus[0].generation_id) for p in corpus):
        raise ValueError("mixed role corpus identity")
    if len({p.canonical_input_fingerprint for p in corpus}) != len(corpus):
        raise ValueError("duplicate captured request")
    if type(hard_ceiling) is not int or hard_ceiling < BOOTSTRAP[ROLES.index(role)]:
        raise ValueError("invalid hard ceiling")
    results, attempted = [], []
    logical_ids, attempt_ids = set(), set()
    started = monotonic()

    def call(request, ceiling):
        if ceiling > hard_ceiling:
            raise ValueError("calibration hard ceiling exceeded")
        prepared = with_ceiling(request, ceiling, qwen, corpus_identity)
        result = invoke(prepared)
        if type(result) is not RoleCallResult or result.prepared_request != prepared:
            raise ValueError("calibration result identity mismatch")
        attempt = result.attempt
        if attempt.context.logical_request_id in logical_ids or attempt.context.attempt_id in attempt_ids:
            raise ValueError("every calibration dispatch needs a fresh request and attempt identity")
        logical_ids.add(attempt.context.logical_request_id)
        attempt_ids.add(attempt.context.attempt_id)
        if not attempt.metrics.usage.usage_complete or attempt.metrics.usage.output_tokens is None or attempt.finish_reason not in ("stop", "length"):
            raise ValueError("complete actual usage and explicit finish required")
        results.append(result)
        attempted.append(ceiling)
        return result

    ceiling = BOOTSTRAP[ROLES.index(role)]
    observed = 0
    for request in corpus:
        while True:
            result = call(request, ceiling)
            if not result.truncated:
                observed = max(observed, result.attempt.metrics.usage.output_tokens)
                break
            ceiling = ((2 * ceiling + 63) // 64) * 64
    recommendation = recommended_limit(observed)
    while True:
        replay = [call(request, recommendation) for request in corpus]
        if all(not result.truncated for result in replay):
            break
        recommendation += 64
    summary = RoleCalibration(role=role, bootstrap_max_tokens=BOOTSTRAP[ROLES.index(role)], attempted_max_tokens=tuple(attempted),
        observed_max_completion_tokens=observed, recommended_max_tokens=recommendation,
        stop_count=sum(not r.truncated for r in results), length_count=sum(r.truncated for r in results),
        request_count=len(results), strict_valid_count=sum(not r.truncated for r in results), usage_complete=True,
        prompt_sha256=corpus[0].prompt_sha256, output_schema_sha256=corpus[0].output_schema_sha256)
    return summary, tuple(results), monotonic() - started


def calibrate_and_write(captured, *, sample, executed_scenario_ids, runtime_inputs, qwen,
                        hard_ceiling, invoke, output_directory):
    """Replay a sealed train corpus and emit a recommendation, without promotion.

    The integrated pilot supplies every captured request and its scenario binding.
    This function cannot extend a deficient corpus with generated probes.
    """
    import os
    from pathlib import Path
    from toolsandbox_pipeline.retrieval.index import file_hash
    if type(sample) is not CalibrationSample or type(runtime_inputs) is not CalibrationRuntimeInputs:
        raise TypeError("validated sample and runtime bindings required")
    if type(captured) is not tuple or any(type(r) is not CapturedCalibrationRequest for r in captured):
        raise TypeError("sealed captured request corpus required")
    executed = tuple(executed_scenario_ids)
    if executed[:32] != sample.initial or executed[32:] != sample.reserve[:len(executed) - 32] or len(set(executed)) != len(executed):
        raise ValueError("pilot must execute initial set then a prefix of the reserve")
    if not captured or {r.scenario_id for r in captured} != set(executed):
        raise ValueError("corpus must cover exactly executed pilot scenarios")
    positions = {sid: i for i, sid in enumerate(executed)}
    order = [positions[r.scenario_id] for r in captured]
    if order != sorted(order):
        raise ValueError("captured requests must preserve pilot scenario order")
    payload = dict(protocol_version=PROTOCOL, sample=sample.model_dump(mode="json"),
        executed_scenario_ids=list(executed), runtime_inputs=runtime_inputs.model_dump(mode="json"),
        captured_requests=[r.model_dump(mode="json") for r in captured])
    corpus_identity = canonical_sha256(payload)
    summaries, attempts = [], []
    started = monotonic()
    for role in ROLES:
        corpus = tuple(r.prepared for r in captured if r.prepared.role == role)
        summary, results, _ = calibrate_role(corpus, qwen=qwen, corpus_identity=corpus_identity,
                                           hard_ceiling=hard_ceiling, invoke=invoke)
        summaries.append(summary)
        attempts.extend(results)
    payload.update(corpus_sha256=corpus_identity, role_summaries=[s.model_dump(mode="json") for s in summaries],
        attempts=[r.model_dump(mode="json") for r in attempts], total_running_time_seconds=monotonic() - started,
        total_tokens=sum(r.attempt.metrics.usage.total_tokens for r in attempts), usage_complete=True)
    raw_artifact = canonical_json_bytes(payload)
    config = CalibratedTokenLimits(schema_version=1, status="calibrated", calibration_protocol_version=PROTOCOL, data_seed=0,
        train_manifest_sha256=sample.train_manifest_sha256, ordered_calibration_scenario_ids=executed,
        runtime_inputs_sha256=canonical_sha256(runtime_inputs.model_dump(mode="json")),
        qwen_config_sha256=canonical_sha256(qwen.model_dump(mode="json")),
        calibration_artifact_sha256=file_hash(raw_artifact), roles=tuple(summaries))
    blobs = {"calibration.json": raw_artifact, "online_token_limits.recommended.json": canonical_json_bytes(config.model_dump(mode="json"))}
    root = Path(output_directory)
    if not root.is_absolute() or ".." in root.parts or any(p.is_symlink() for p in (root, *root.parents)) or root.exists():
        raise ValueError("new explicit private calibration output directory required")
    root.mkdir(mode=0o700)
    for name, raw in blobs.items():
        fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return config, {name: file_hash(raw) for name, raw in blobs.items()}
