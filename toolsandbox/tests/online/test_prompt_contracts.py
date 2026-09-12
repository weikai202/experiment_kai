import pytest
from toolsandbox_pipeline.online.prompt_contracts import PromptSafeControllerDecision
from toolsandbox_pipeline.online.prompt_builder import safe_controller
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256


def test_safe_evidence_hash():
    data = dict(code="INVALID_FUNCTION", source_kind="schema", source_ref="private_tool")
    decision = ControllerDecision.model_validate(dict(blocking_codes=["INVALID_FUNCTION"], critic_trigger_codes=[], evidence=[data]))
    safe = safe_controller(decision)
    assert safe.evidence[0].evidence_ref == canonical_sha256(data)
    assert "private_tool" not in safe.model_dump_json()
    assert "source_ref" not in safe.model_dump_json()
    with pytest.raises(ValueError):
        PromptSafeControllerDecision(blocking_codes=("INVALID_FUNCTION",), critic_trigger_codes=(), evidence=())


def _argument_decision(ref, kind="state"):
    return ControllerDecision.model_validate(dict(blocking_codes=["UNGROUNDED_ARGUMENT"], critic_trigger_codes=[],
        evidence=[dict(code="UNGROUNDED_ARGUMENT", source_kind=kind, source_ref=ref)]))


def _calls(arguments, batch=None):
    from toolsandbox_pipeline.schemas.action import ActionEnvelope
    call = dict(call_id="c0", selected_skill_id=None, name="scrambled", arguments=arguments)
    body = dict(type="function_call", **call) if batch is None else dict(type="parallel_batch", calls=[dict(call, call_id=f"c{i}") for i in range(batch)])
    return ActionEnvelope.model_validate(dict(action=body))


def test_visible_grounding_location_retains_hash_and_old_serialization():
    decision = _argument_decision("action://arguments/days")
    action = _calls(dict(timestamp=123, days=-1))
    old = safe_controller(decision)
    located = safe_controller(decision, action)
    assert "action_pointer" not in old.model_dump(mode="json")["evidence"][0]
    assert located.evidence[0].action_pointer == "/action/arguments/days"
    assert located.evidence[0].evidence_ref == old.evidence[0].evidence_ref
    assert "timestamp" not in located.model_dump_json()
    assert "action://" not in located.model_dump_json()
    assert PromptSafeControllerDecision.model_validate_json(old.model_dump_json()) == old


@pytest.mark.parametrize("ref,kind", [
    ("metadata:canonical_hidden:days", "tool_metadata"),
    ("action://arguments/absent", "state"),
    ("action://arguments/days/../../hidden", "state"),
    ("action://arguments/days", "tool_metadata"),
    ("action://arguments/%64ays", "state"),
    ("action://arguments/days~2", "state"),
])
def test_unverified_host_refs_never_become_locations(ref, kind):
    result = safe_controller(_argument_decision(ref, kind), _calls(dict(days=-1)))
    assert result.evidence[0].action_pointer is None
    assert ref not in result.model_dump_json()


def test_batch_call_location_is_exact_and_ambiguous_ref_is_omitted():
    action = _calls(dict(days=-1), batch=2)
    assert safe_controller(_argument_decision("action:/calls/1/arguments/days"), action).evidence[0].action_pointer == "/action/calls/1/arguments/days"
    assert safe_controller(_argument_decision("action://arguments/days"), action).evidence[0].action_pointer is None
    assert safe_controller(_argument_decision("action:/calls/2/arguments/days"), action).evidence[0].action_pointer is None
    assert safe_controller(_argument_decision("action://arguments/days"), _calls(dict(days=-1), batch=1)).evidence[0].action_pointer == "/action/calls/0/arguments/days"


def test_nested_unicode_and_escaped_property_locations():
    action = _calls({"a/b~": [{"日期": -1}]})
    pointer = safe_controller(_argument_decision("action://arguments/a~1b~0/0/日期"), action).evidence[0].action_pointer
    assert pointer == "/action/arguments/a~1b~0/0/日期"


def test_critic_grammar_fingerprint_distinct_and_legacy_roles_unchanged():
    from toolsandbox_pipeline.online.prompt_builder import fingerprint
    from toolsandbox_pipeline.online.prompt_contracts import PromptMessage
    from toolsandbox_pipeline.schemas.runtime import QwenConfig
    from toolsandbox_pipeline.providers.critic_grammar import critic_grammar_sha256
    messages = (PromptMessage(role='system', content='System'), PromptMessage(role='user', content='{}'))
    plain = QwenConfig(structured_output_wire_mode='structured_outputs_json', structured_output_backend='xgrammar')
    grammar = QwenConfig(structured_output_wire_mode='structured_outputs_json', structured_output_backend='xgrammar',
        critic_structured_output_mode='ordered_unique_grammar_v1', critic_grammar_sha256=critic_grammar_sha256())
    schema_hash = 'sha256:' + 'a' * 64
    for role in ('policy', 'critic', 'revision'):
        original_payload = dict(provider=plain.provider, model=plain.model, role=role,
            messages=[message.model_dump(mode='json') for message in messages],
            output_schema_sha256=schema_hash, max_tokens=768, temperature=0.0, seed=0,
            top_p='omitted', enable_thinking=False, structured_output_wire_mode=plain.structured_output_wire_mode)
        legacy = canonical_sha256(original_payload)
        assert fingerprint(role, messages, schema_hash, 768, plain) == legacy
        actual = fingerprint(role, messages, schema_hash, 768, grammar)
        if role == 'critic':
            assert actual != legacy
            assert actual == canonical_sha256(dict(original_payload,
                critic_structured_output_mode='ordered_unique_grammar_v1', critic_grammar_sha256=critic_grammar_sha256()))
        else:
            assert actual == legacy
