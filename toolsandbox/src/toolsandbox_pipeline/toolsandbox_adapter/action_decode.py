"""Assign action call identity before strict execution-contract validation.

Model call_id strings are untrusted labels. The private wire reader preserves
all original constraints except label uniqueness; the public ActionEnvelope
continues requiring unique IDs. Native execution identity remains dispatch-only.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Annotated, Literal

from pydantic import Field
from toolsandbox_pipeline.providers.contracts import RequestContext
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.action import (ActionEnvelope, ActionType, FunctionCall,
    FunctionCallAction, AssistantMessageAction)
from toolsandbox_pipeline.schemas.base import StrictModel

ACTION_DECODE_VERSION = 'host-action-call-identity-v1'


class _ModelBatch(StrictModel):
    type: Literal[ActionType.PARALLEL_BATCH]
    calls: list[FunctionCall] = Field(min_length=1)


class _ModelEnvelope(StrictModel):
    action: Annotated[FunctionCallAction | _ModelBatch | AssistantMessageAction, Field(discriminator='type')]


@dataclass(frozen=True)
class ActionCallIdentity:
    position: int
    model_call_id: str
    host_action_call_id: str


@dataclass(frozen=True)
class DecodedAction:
    action: ActionEnvelope
    call_identities: tuple[ActionCallIdentity, ...]


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def decode_action(content: str, context: RequestContext) -> DecodedAction:
    if type(content) is not str or type(context) is not RequestContext:
        raise TypeError('original JSON content and exact request context required')
    json.loads(content, object_pairs_hook=_unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    source = _ModelEnvelope.model_validate_json(content)
    payload = source.model_dump(mode='json')
    action = payload['action']
    if action['type'] == 'assistant_message':
        return DecodedAction(ActionEnvelope.model_validate_json(content), ())
    calls = action['calls'] if action['type'] == 'parallel_batch' else [action]
    identities = []
    for position, call in enumerate(calls):
        label = call['call_id']
        identifier = 'hostcall_' + canonical_sha256({
            'version': ACTION_DECODE_VERSION, 'logical_request_id': context.logical_request_id,
            'unit_reference': context.unit_reference, 'input_fingerprint': context.input_fingerprint,
            'role': context.role.value, 'position': position,
        })[7:]
        call['call_id'] = identifier
        identities.append(ActionCallIdentity(position, label, identifier))
    return DecodedAction(ActionEnvelope.model_validate_json(json.dumps(payload)), tuple(identities))


def action_decoding_audit(*, content: str, raw_response_body: bytes,
                          context: RequestContext, action: ActionEnvelope,
                          allow_legacy: bool = False) -> dict | None:
    """Verify material against raw content; never rewrite legacy committed IDs."""
    if type(action) is not ActionEnvelope or type(raw_response_body) is not bytes:
        raise TypeError('exact action and immutable provider response bytes required')
    decoded = decode_action(content, context)
    if decoded.action != action:
        if allow_legacy and ActionEnvelope.model_validate_json(content) == action:
            return None
        raise ValueError('stored action differs from original decoded provider output')
    if not decoded.call_identities:
        return None
    return {'version': ACTION_DECODE_VERSION, 'logical_request_id': context.logical_request_id,
        'source_attempt_id': context.attempt_id,
        'raw_response_sha256': 'sha256:' + sha256(raw_response_body).hexdigest(),
        'decoded_action_sha256': canonical_sha256(action.model_dump(mode='json')),
        'calls': [{'position': item.position, 'model_call_id': item.model_call_id,
                   'host_action_call_id': item.host_action_call_id} for item in decoded.call_identities]}


def audit_provider_action(response, *, allow_legacy=False):
    """Read only the agent's output content, not another role or hidden state."""
    if type(response.value) is not ActionEnvelope or isinstance(response.value.action, AssistantMessageAction):
        return None
    body = json.loads(response.raw_response_body, object_pairs_hook=_unique)
    content = body['choices'][0]['message']['content']
    return action_decoding_audit(content=content, raw_response_body=response.raw_response_body,
        context=response.attempt.context, action=response.value, allow_legacy=allow_legacy)
