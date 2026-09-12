import json
from pathlib import Path

from toolsandbox_pipeline.offline.memory_prompts import load_memory_prompts
from toolsandbox_pipeline.offline.memory_roles import (
    MemoryRoleRunner,
    load_token_limits,
    prepare_candidate_request,
)
from toolsandbox_pipeline.providers.contracts import ProviderRole, RequestContext, TransportResponse
from toolsandbox_pipeline.providers.qwen import QwenGateway
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.offline_memory import PolicyMemoryCandidateDecision, PolicyTrajectoryProjection
from toolsandbox_pipeline.schemas.runtime import QwenConfig


ROOT = Path(__file__).parents[2]
DIGEST = "sha256:" + "a" * 64


def projection():
    return PolicyTrajectoryProjection(
        trajectory_id=DIGEST,
        manifest_position=0,
        visible_states=(),
        retrieved_policy_memory=(),
        retrieved_skills=(),
        proposed_actions=(ActionEnvelope(action={"type": "assistant_message", "content": "draft"}),),
        final_actions=(ActionEnvelope(action={"type": "assistant_message", "content": "final"}),),
        controller_codes=(),
        visible_tool_outcomes=(),
        native_similarity=1.0,
        fully_successful=True,
        host_attribution="successful",
    )


class Transport:
    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        content = {
            "result": "CANDIDATE",
            "role": "policy",
            "candidate": {
                "scope": "Confirm prerequisites",
                "applicability": [],
                "action_guidance": "Validate required arguments before using a tool",
                "avoid": [],
            },
        }
        data = {
            "model": "Qwen/Qwen3-32B",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16},
        }
        return TransportResponse(raw_body=json.dumps(data).encode(), data=data)


class Seam:
    def __init__(self):
        self.response = None
        self.persisted = 0

    def load_completed_response(self, context):
        return self.response

    def persist_completed_response(self, response):
        self.response = response
        self.persisted += 1


def context(prepared, attempt="attempt-1"):
    return RequestContext(
        logical_request_id="logical-1",
        attempt_id=attempt,
        role=ProviderRole.MEMORY_CANDIDATE,
        phase="offline",
        unit_reference=prepared.unit_reference,
        input_fingerprint=prepared.canonical_input_fingerprint,
        replayed_after_unknown_outcome=False,
        manifest_identity=DIGEST,
    )


def test_candidate_role_dispatches_once_with_frozen_qwen_fields_and_reuses_response():
    prompts = load_memory_prompts(ROOT.resolve(), (ROOT / "prompts/offline/memory_manifest.json").resolve())
    limits, limits_sha = load_token_limits((ROOT / "configs/offline_memory_token_limits.provisional.json").resolve())
    prepared = prepare_candidate_request(
        projection(), unit_reference="memory-unit-1", prompts=prompts, limits=limits,
        limits_sha256=limits_sha, structured_output_wire_mode="guided_json",
    )
    transport = Transport()
    seam = Seam()
    gateway = QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"), transport=transport)
    runner = MemoryRoleRunner(gateway, manifest_identity=DIGEST, durability_seam=seam)
    first = runner.run(prepared, context(prepared))
    second = runner.run(prepared, context(prepared, "attempt-recovery"))
    assert type(first.output) is PolicyMemoryCandidateDecision
    assert second.output == first.output
    assert len(transport.requests) == 1 and seam.persisted == 1
    request = transport.requests[0]
    assert request["model"] == "Qwen/Qwen3-32B"
    assert request["temperature"] == 0.0 and request["seed"] == 0
    assert request["max_tokens"] == 512 and "top_p" not in request
    assert request["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_packed_v2_request_preserves_original_and_dispatches_with_new_identity():
    from toolsandbox_pipeline.offline.reflection_packing import unintern, unpack, VERSION
    from toolsandbox_pipeline.reproducibility import canonical_sha256
    original = projection()
    original_json = original.model_dump_json()
    prompts = load_memory_prompts(ROOT.resolve(), (ROOT / 'prompts/offline/memory_manifest.json').resolve())
    limits, digest = load_token_limits((ROOT / 'configs/offline_memory_token_limits.provisional.json').resolve())
    args = dict(unit_reference='memory-unit-v2', prompts=prompts, limits=limits,
                limits_sha256=digest, structured_output_wire_mode='guided_json')
    old = prepare_candidate_request(original, **args)
    new = prepare_candidate_request(original, **args, input_representation='packed-v2')
    envelope = json.loads(new.messages[1].content)
    restored = unpack(unintern(envelope['trajectory_encoding']))
    assert canonical_sha256(restored) == envelope['original_projection_sha256']
    assert restored == original.model_dump(mode='json')
    assert original.model_dump_json() == original_json
    assert envelope['input_representation_version'] == VERSION
    assert new.prompt_version == 'v2' and old.prompt_version == 'v1'
    assert new.canonical_input_fingerprint != old.canonical_input_fingerprint
    assert new.prompt_sha256 != old.prompt_sha256
    transport = Transport()
    runner = MemoryRoleRunner(QwenGateway(QwenConfig(structured_output_wire_mode='guided_json'), transport=transport), manifest_identity=DIGEST)
    result = runner.run(new, context(new))
    assert result.output.decision().result == 'CANDIDATE'
    assert len(transport.requests) == 1
    assert json.loads(transport.requests[0]['messages'][1]['content']) == envelope


def test_candidate_wire_shapes_roundtrip_and_reject_null_payloads():
    import pytest
    from jsonschema import Draft202012Validator
    from pydantic import ValidationError
    from toolsandbox_pipeline.schemas.offline_memory import WorldMemoryCandidateDecision
    for model, role, content in (
        (PolicyMemoryCandidateDecision, 'policy', dict(scope='Prerequisites', applicability=[], action_guidance='Verify arguments', avoid=[])),
        (WorldMemoryCandidateDecision, 'world', dict(action_pattern='Tool execution', state_conditions=[], schema_conditions=[], likely_error_codes=[], outcome_calibration='Check prerequisites', correction_principle='Verify arguments')),
    ):
        validator = Draft202012Validator(model.model_json_schema())
        valid = [{'result': 'NONE'}, {'result': 'CANDIDATE', 'role': role, 'candidate': content}]
        invalid = [dict(result='NONE', role=None), dict(result='NONE', candidate=None),
                   dict(result='CANDIDATE', role=None, candidate=content),
                   dict(result='CANDIDATE', role=role, candidate=None)]
        for wire in valid:
            validator.validate(wire)
            parsed = model.model_validate_json(json.dumps(wire))
            assert parsed.model_dump(mode='json') == wire
            assert model.model_validate_json(parsed.model_dump_json()) == parsed
            assert parsed.decision().result == wire['result']
        for wire in invalid:
            assert list(validator.iter_errors(wire))
            with pytest.raises(ValidationError):
                model.model_validate_json(json.dumps(wire))
