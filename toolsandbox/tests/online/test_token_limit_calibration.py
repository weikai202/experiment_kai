import pytest
from toolsandbox_pipeline.online.token_limit_calibration import CalibrationScenario, VARIANTS, select_calibration_sample, with_ceiling
from toolsandbox_pipeline.schemas.runtime import QwenConfig
from tests.online.test_prompt_builder import contexts, prepared


def test_sample_order_and_variant_minimum():
    records = tuple(CalibrationScenario(scenario_id=f"scenario-{i:03d}-{j}", variant=v, split="train") for i, v in enumerate(VARIANTS) for j in range(10))
    sample = select_calibration_sample(records, "sha256:" + "a" * 64)
    assert len(sample.initial) == len(sample.reserve) == 32
    assert sample == select_calibration_sample(records, "sha256:" + "a" * 64)
    positions = {r.scenario_id: i for i, r in enumerate(records)}
    assert [positions[i] for i in sample.initial] == sorted(positions[i] for i in sample.initial)
    with pytest.raises(ValueError):
        select_calibration_sample(records[:5], "sha256:" + "a" * 64)


def test_ceiling_changes_fingerprint_and_enforces_hard_limit(tmp_path):
    request = prepared(contexts(tmp_path)[0], 0)
    qwen = QwenConfig(structured_output_wire_mode="guided_json", output_limit=512)
    changed = with_ceiling(request, 512, qwen, "sha256:" + "a" * 64)
    assert changed.messages == request.messages
    assert changed.canonical_input_fingerprint != request.canonical_input_fingerprint
    with pytest.raises(ValueError):
        with_ceiling(request, 513, qwen, "sha256:" + "a" * 64)


def test_full_corpus_search_only_retries_length(tmp_path):
    import json
    from toolsandbox_pipeline.online.prompt_contracts import InitialPolicyContext
    from toolsandbox_pipeline.online.token_limit_calibration import calibrate_role
    from toolsandbox_pipeline.online.qwen_roles import InitialPolicyRunner
    from toolsandbox_pipeline.providers.qwen import QwenGateway
    from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
    from tests.retrieval.test_queries import state
    from tests.online.test_qwen_roles import Chat, MANIFEST, request_context
    initial = contexts(tmp_path)[0]
    corpus = []
    for i in range(32):
        current = state(f"synthetic request {i}")
        envelope = json.loads(initial.user_envelope)
        envelope["state"] = current.model_dump(mode="json", by_alias=True)
        identity = initial.identity_payload()
        identity.update(state_id=current.state_id, user_envelope=envelope)
        context = InitialPolicyContext(**{**initial.model_dump(), "state_id": current.state_id,
            "user_envelope": canonical_json_bytes(envelope).decode(), "policy_context_hash": canonical_sha256(identity)})
        corpus.append(prepared(context, 0))
    class LengthChat(Chat):
        def create(self, **kwargs):
            self.finish = "length" if kwargs["max_tokens"] < 128 else "stop"
            return super().create(**kwargs)
    transport = LengthChat('{"action":{"type":"assistant_message","content":"Please clarify"}}', tokens=10)
    qwen = QwenConfig(structured_output_wire_mode="guided_json", output_limit=512)
    runner = InitialPolicyRunner(QwenGateway(qwen, transport=transport), manifest_identity=MANIFEST, mode="calibration")
    def invoke(request):
        return runner.run(request, request_context(request, len(transport.calls)))
    summary, results, duration = calibrate_role(tuple(corpus), qwen=qwen, corpus_identity="sha256:" + "a" * 64, hard_ceiling=512, invoke=invoke)
    assert summary.recommended_max_tokens == 128
    assert summary.length_count == 32
    assert summary.strict_valid_count == 64
    assert len({r.attempt.context.logical_request_id for r in results}) == len(results)
    assert duration >= 0


def test_calibration_artifact_contains_all_roles_and_hashes(tmp_path):
    import json
    from toolsandbox_pipeline.online.prompt_contracts import InitialPolicyContext, CriticContext, RevisionContext
    from toolsandbox_pipeline.online.token_limit_calibration import (CalibrationRuntimeInputs, CapturedCalibrationRequest, calibrate_and_write)
    from toolsandbox_pipeline.online.qwen_roles import InitialPolicyRunner, CriticRunner, RevisionRunner
    from toolsandbox_pipeline.providers.qwen import QwenGateway
    from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
    from toolsandbox_pipeline.retrieval.index import file_hash
    from tests.retrieval.test_queries import state
    from tests.online.test_qwen_roles import Chat, MANIFEST, request_context
    initial, critic, revision = contexts(tmp_path)
    records = tuple(CalibrationScenario(scenario_id=f"sample-{i}-{j}", variant=v, split="train") for i, v in enumerate(VARIANTS) for j in range(8))
    sample = select_calibration_sample(records, "sha256:" + "a" * 64)
    captured = []
    for sid in sample.initial:
        current = state(sid)
        envelope = json.loads(initial.user_envelope)
        envelope["state"] = current.model_dump(mode="json", by_alias=True)
        identity = initial.identity_payload()
        identity.update(state_id=current.state_id, user_envelope=envelope)
        first = InitialPolicyContext(**{**initial.model_dump(), "state_id": current.state_id,
            "user_envelope": canonical_json_bytes(envelope).decode(), "policy_context_hash": canonical_sha256(identity)})
        second = CriticContext(**{**critic.model_dump(), "initial": first})
        third = RevisionContext(initial=first, critic=second, critic_feedback_json=revision.critic_feedback_json)
        captured.extend(CapturedCalibrationRequest(scenario_id=sid, scenario_family_id="synthetic-family", prepared=prepared(c, i)) for i, c in enumerate((first, second, third)))
    bindings = CalibrationRuntimeInputs(**{name: "sha256:" + "a" * 64 for name in CalibrationRuntimeInputs.model_fields if name.endswith("sha256")},
        user_simulator_model="gpt-4o-mini-2024-07-18", online_orchestrator_version="synthetic-v1")
    chat = Chat("")
    qwen = QwenConfig(structured_output_wire_mode="guided_json")
    gateway = QwenGateway(qwen, transport=chat)
    runners = {cls.role: cls(gateway, manifest_identity=MANIFEST, mode="calibration") for cls in (InitialPolicyRunner, CriticRunner, RevisionRunner)}
    def invoke(request):
        chat.content = revision.critic_feedback_json if request.role == "critic" else '{"action":{"type":"assistant_message","content":"Please clarify"}}'
        return runners[request.role].run(request, request_context(request, len(chat.calls)))
    output = tmp_path / "calibrated"
    config, hashes = calibrate_and_write(tuple(captured), sample=sample, executed_scenario_ids=sample.initial,
        runtime_inputs=bindings, qwen=qwen, hard_ceiling=1024, invoke=invoke, output_directory=output)
    artifact = json.loads((output / "calibration.json").read_bytes())
    assert len(artifact["attempts"]) == 192
    assert artifact["total_tokens"] == 192 * 30
    assert all(r.recommended_max_tokens == 64 for r in config.roles)
    assert config.calibration_artifact_sha256 == hashes["calibration.json"]
    for name, digest in hashes.items():
        assert file_hash((output / name).read_bytes()) == digest
