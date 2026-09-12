import copy
import pytest
from toolsandbox_pipeline.providers import QwenGateway
from toolsandbox_pipeline.schemas.runtime import QwenConfig, RoleTokenLimitConfig
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.critic import CriticOutput
from toolsandbox_pipeline.providers.contracts import (
    ProviderRequestError,
    ProviderRole,
    ROLE_BOOTSTRAP_MAX_TOKENS,
)
from tests.providers.test_request_identity import context, FakeTransport, chat_response


@pytest.mark.parametrize("mode", ["guided_json", "structured_outputs_json"])
def test_exact_qwen_wire(mode):
    transport = FakeTransport(chat_response())
    result = QwenGateway(QwenConfig(structured_output_wire_mode=mode), transport=transport).generate(
        context(), [{"role": "user", "content": "fixed input"}], ActionEnvelope, max_tokens=256)
    request, = transport.calls
    assert set(request) == {"model", "messages", "temperature", "seed", "max_tokens", "extra_body"}
    assert request["temperature"] == 0.0 and request["seed"] == 0 and request["max_tokens"] == 256
    extra = request["extra_body"]
    assert extra["chat_template_kwargs"] == {"enable_thinking": False}
    key = "guided_json" if mode == "guided_json" else "structured_outputs"
    assert set(extra) == {"chat_template_kwargs", key}
    assert extra[key] == (ActionEnvelope.model_json_schema() if mode == "guided_json" else {"json": ActionEnvelope.model_json_schema()})
    assert result.value.action.content == "ok"
    assert result.attempt.finish_reason == "stop"
    assert result.attempt.metrics.usage.total_tokens == 8


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(model="wrong"),
    lambda r: r.update(choices=[]),
    lambda r: r["choices"].append(copy.deepcopy(r["choices"][0])),
    lambda r: r["choices"][0]["message"].update(content=""),
    lambda r: r["choices"][0]["message"].update(content=None),
    lambda r: r["choices"][0]["message"].update(content="not JSON"),
    lambda r: r["choices"][0]["message"].update(content='{"action":{"type":"assistant_message","content":12}}'),
    lambda r: r["choices"][0]["message"].update(content='{"action":{"type":"assistant_message","content":"ok"},"extra":1}'),
    lambda r: r["choices"][0]["message"].update(content='{"action":null,"action":null}'),
    lambda r: r["choices"][0]["message"].update(reasoning_content="secret-sentinel"),
    lambda r: r["choices"][0]["message"].update(reasoning="secret-sentinel"),
    lambda r: r["choices"][0]["message"].update(content="<think>hidden</think>{}"),
    lambda r: r["choices"][0]["message"].update(tool_calls=[{}]),
])
def test_bad_qwen_output_fails_once(mutation):
    raw = chat_response()
    mutation(raw)
    transport = FakeTransport(raw)
    with pytest.raises(ProviderRequestError) as caught:
        QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"), transport=transport).generate(
            context(), [{"role": "user", "content": "input"}], ActionEnvelope, max_tokens=256)
    assert len(transport.calls) == 1
    assert caught.value.attempt.status.value == "failed"
    assert caught.value.attempt.response_hash
    assert caught.value.attempt.metrics.usage.total_tokens == 8
    assert "secret-sentinel" not in repr(caught.value.attempt)


@pytest.mark.parametrize("role,limit", list(ROLE_BOOTSTRAP_MAX_TOKENS.items()))
def test_bootstrap_and_calibrated_role_limits(role, limit):
    model = CriticOutput if role is ProviderRole.CRITIC else ActionEnvelope
    content = '{"verdict":"accept","predicted_outcome":"success","predicted_effect":"effect","error_codes":[],"correction":""}' if role is ProviderRole.CRITIC else '{"action":{"type":"assistant_message","content":"ok"}}'
    transport = FakeTransport(chat_response(content))
    gateway = QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"), transport=transport)
    gateway.generate(context(role), [{"role": "user", "content": "x"}], model, max_tokens=limit)
    assert transport.calls[0]["max_tokens"] == limit
    gateway.generate(context(role, attempt_id="attempt-2"), [{"role": "user", "content": "x"}], model, max_tokens=limit + 64,
                     token_limit_config=RoleTokenLimitConfig(version="v1", role=role.value, stage="calibration",
                                                            max_tokens=limit + 64,
                                                            evidence_manifest_identity="sha256:" + "d" * 64))
    assert transport.calls[1]["max_tokens"] == limit + 64


@pytest.mark.parametrize("limit", [0, -1, True, "256"])
def test_invalid_role_limit_rejected_before_dispatch(limit):
    transport = FakeTransport()
    with pytest.raises(ProviderRequestError) as caught:
        QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"), transport=transport).generate(
            context(), [{"role": "user", "content": "x"}], ActionEnvelope, max_tokens=limit)
    assert caught.value.attempt.status.value == "rejected_before_dispatch"
    assert not transport.calls


def test_deployment_output_limit_enforced_before_dispatch():
    transport = FakeTransport()
    gateway = QwenGateway(
        QwenConfig(structured_output_wire_mode="guided_json", output_limit=320),
        transport=transport,
    )
    with pytest.raises(ProviderRequestError):
        gateway.generate(context(), [{"role": "user", "content": "x"}], ActionEnvelope, max_tokens=384)
    assert not transport.calls


def test_explicit_length_finish_is_typed_and_preserves_actual_usage():
    raw = chat_response('{"action":{"type":"assistant_message"')
    raw["choices"][0]["finish_reason"] = "length"
    transport = FakeTransport(raw)
    with pytest.raises(ProviderRequestError) as caught:
        QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"), transport=transport).generate(
            context(), [{"role": "user", "content": "x"}], ActionEnvelope, max_tokens=256)
    attempt = caught.value.attempt
    assert attempt.exception_class == "OutputTruncated"
    assert attempt.finish_reason == "length"
    assert attempt.response_hash
    assert attempt.metrics.usage.output_tokens == 3
    assert len(transport.calls) == 1


def test_actual_critic_contract_and_no_usage_estimation():
    raw = chat_response('{"verdict":"accept","predicted_outcome":"success","predicted_effect":"A possible effect","error_codes":[],"correction":""}')
    raw.pop("usage")
    result = QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"), transport=FakeTransport(raw)).generate(
        context(ProviderRole.CRITIC), [{"role": "user", "content": "x"}], CriticOutput, max_tokens=384)
    assert not result.attempt.metrics.usage.usage_complete
    assert result.attempt.metrics.usage.total_tokens is None


@pytest.mark.parametrize("messages", [[], None, [{"role": "tool", "content": "x"}], [{"role": "user", "content": 1}]])
def test_invalid_messages_before_dispatch(messages):
    transport = FakeTransport()
    with pytest.raises(ProviderRequestError):
        QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"), transport=transport).generate(
            context(), messages, ActionEnvelope, max_tokens=256)
    assert not transport.calls


def test_selected_token_config_must_match_role_and_limit():
    transport = FakeTransport(chat_response())
    gateway = QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"), transport=transport)
    for selected in (
        RoleTokenLimitConfig(version="v1", role="critic", stage="bootstrap", max_tokens=384),
        RoleTokenLimitConfig(version="v1", role="policy", stage="bootstrap", max_tokens=256),
    ):
        with pytest.raises(ProviderRequestError):
            gateway.generate(context(), [{"role": "user", "content": "x"}], ActionEnvelope,
                             max_tokens=384, token_limit_config=selected)
    assert not transport.calls


def test_formal_limit_is_not_a_request_override():
    transport = FakeTransport(chat_response())
    gateway = QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"), transport=transport)
    selected = RoleTokenLimitConfig(version="v1", role="policy", stage="calibrated", max_tokens=512,
                                    evidence_manifest_identity="sha256:" + "c" * 64)
    result = gateway.generate(context(), [{"role": "user", "content": "x"}], ActionEnvelope,
                              max_tokens=512, token_limit_config=selected)
    assert result.attempt.finish_reason == "stop"
    assert transport.calls[0]["max_tokens"] == 512
    with pytest.raises(ProviderRequestError):
        gateway.generate(context(attempt_id="a2"), [{"role": "user", "content": "x"}], ActionEnvelope,
                         max_tokens=1024, token_limit_config=selected)
    assert len(transport.calls) == 1


@pytest.mark.parametrize('role', [ProviderRole.CRITIC, ProviderRole.POLICY, ProviderRole.REVISION, ProviderRole.MEMORY_CANDIDATE])
def test_critic_grammar_is_used_only_for_critic(role):
    from toolsandbox_pipeline.providers.critic_grammar import build_critic_grammar
    from tests.providers.test_runtime_config import grammar_config
    critic = role is ProviderRole.CRITIC
    content = ('{"verdict":"accept","predicted_outcome":"success","predicted_effect":"effect","error_codes":[],"correction":""}'
               if critic else '{"action":{"type":"assistant_message","content":"ok"}}')
    transport = FakeTransport(chat_response(content))
    model = CriticOutput if critic else ActionEnvelope
    result = QwenGateway(grammar_config(), transport=transport).generate(
        context(role), [{'role': 'user', 'content': 'fixed input'}], model,
        max_tokens=ROLE_BOOTSTRAP_MAX_TOKENS[role])
    extra = transport.calls[0]['extra_body']
    assert extra['structured_outputs'] == ({'grammar': build_critic_grammar()} if critic else {'json': model.model_json_schema()})
    assert set(extra) == {'chat_template_kwargs', 'structured_outputs'}
    assert result.attempt.finish_reason == 'stop'


def test_critic_grammar_drift_fails_before_transport_and_never_falls_back(monkeypatch):
    from toolsandbox_pipeline.providers import critic_grammar
    from tests.providers.test_runtime_config import grammar_config
    config = grammar_config()
    monkeypatch.setattr(critic_grammar, 'build_critic_grammar', lambda: 'changed grammar')
    transport = FakeTransport()
    with pytest.raises(ProviderRequestError) as caught:
        QwenGateway(config, transport=transport).generate(
            context(ProviderRole.CRITIC), [{'role': 'user', 'content': 'x'}], CriticOutput, max_tokens=384)
    assert caught.value.attempt.status.value == 'rejected_before_dispatch'
    assert not transport.calls


def test_critic_grammar_still_locally_rejects_duplicate_codes():
    from tests.providers.test_runtime_config import grammar_config
    content = '{"verdict":"revise","predicted_outcome":"failure","predicted_effect":"effect","error_codes":["UNGROUNDED_ARGUMENT","UNGROUNDED_ARGUMENT"],"correction":"Ground the argument."}'
    transport = FakeTransport(chat_response(content))
    with pytest.raises(ProviderRequestError) as caught:
        QwenGateway(grammar_config(), transport=transport).generate(
            context(ProviderRole.CRITIC), [{'role': 'user', 'content': 'x'}], CriticOutput, max_tokens=384)
    assert len(transport.calls) == 1
    assert caught.value.attempt.status.value == 'failed'


def test_skill_candidate_strict_tuple_fields_accept_json_arrays():
    from toolsandbox_pipeline.schemas.offline_skill import SkillContent, SkillContentCandidate
    from tests.skills.test_records import skill
    candidate = SkillContentCandidate(candidate=SkillContent.from_record(skill()))
    transport = FakeTransport(chat_response(candidate.model_dump_json()))
    result = QwenGateway(QwenConfig(structured_output_wire_mode='guided_json'), transport=transport).generate(
        context(ProviderRole.SKILL_CANDIDATE), [{'role': 'user', 'content': 'synthetic candidate'}], SkillContentCandidate, max_tokens=2048)
    assert result.value == candidate
    assert isinstance(result.value.candidate.expected_outputs, tuple)


@pytest.mark.parametrize('content', [
    '{"candidate":{},"candidate":{}}',
    '{"candidate":{"required_inputs":NaN}}',
    '{"candidate":{"required_inputs":Infinity}}',
])
def test_skill_candidate_json_still_rejects_duplicate_and_nonfinite(content):
    from toolsandbox_pipeline.schemas.offline_skill import SkillContentCandidate
    with pytest.raises(ProviderRequestError):
        QwenGateway(QwenConfig(structured_output_wire_mode='guided_json'),
            transport=FakeTransport(chat_response(content))).generate(
                context(ProviderRole.SKILL_CANDIDATE), [{'role': 'user', 'content': 'synthetic'}], SkillContentCandidate, max_tokens=2048)
