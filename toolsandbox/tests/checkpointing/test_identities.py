from toolsandbox_pipeline.checkpointing import attempt_id, logical_request_id
from toolsandbox_pipeline.providers.contracts import ProviderRole
from toolsandbox_pipeline.schemas.checkpoint import LogicalLLMRequestIdentity


HASH = "sha256:" + "1" * 64


def identity(**changes):
    values = dict(
        run_id="run-1",
        role=ProviderRole.POLICY,
        phase="online",
        unit_reference="state-1",
        input_fingerprint=HASH,
        model="Qwen/Qwen3-32B",
        decoding_configuration_sha256="sha256:" + "2" * 64,
        output_schema_sha256="sha256:" + "3" * 64,
    )
    values.update(changes)
    return LogicalLLMRequestIdentity(**values)


def test_logical_identity_is_stable_and_sensitive():
    first = logical_request_id(identity())
    assert first == logical_request_id(identity())
    assert first != logical_request_id(identity(phase="offline"))
    assert first != logical_request_id(identity(unit_reference="state-2"))
    assert first != logical_request_id(identity(output_schema_sha256=None))
    assert attempt_id(first, 1) == f"attempt-{first[4:]}-00000001"
    assert attempt_id(first, 2) != attempt_id(first, 1)
