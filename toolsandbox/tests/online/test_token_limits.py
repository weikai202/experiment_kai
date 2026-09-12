from pathlib import Path
import pytest
from toolsandbox_pipeline.online.token_limits import ProvisionalTokenLimits, recommended_limit, select_limit


@pytest.mark.parametrize("observed,expected", [(0,64), (1,64), (51,64), (52,128), (256,320), (257,384)])
def test_rounding(observed, expected):
    assert recommended_limit(observed) == expected


def test_no_provisional_formal_run():
    root = Path(__file__).resolve().parents[2]
    config = ProvisionalTokenLimits.model_validate_json((root / "configs/online_token_limits.provisional.json").read_bytes())
    with pytest.raises(ValueError):
        select_limit(config, "policy", mode="formal", qwen_config_sha256="", prompt_sha256="", schema_sha256="")


def test_development_limits_are_versioned_and_never_formal():
    from toolsandbox_pipeline.online.token_limits import load_token_limits
    from toolsandbox_pipeline.retrieval.index import file_hash
    path = Path(__file__).resolve().parents[2] / "configs/online_token_limits.smoke.json"
    config = load_token_limits(path, expected_sha256=file_hash(path.read_bytes()))
    selected = [select_limit(config, role, mode="calibration", qwen_config_sha256="", prompt_sha256="", schema_sha256="")
                for role in ("policy", "critic", "revision")]
    assert [r.max_tokens for r in selected] == [512, 768, 512]
    assert all(r.stage == "calibration" and r.evidence_manifest_identity for r in selected)
    with pytest.raises(ValueError, match="formal"):
        select_limit(config, "critic", mode="formal", qwen_config_sha256="", prompt_sha256="", schema_sha256="")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_token_limits(path, expected_sha256="sha256:" + "0" * 64)


def test_development_limits_reject_missing_roles_and_false_provenance():
    import json
    from toolsandbox_pipeline.online.token_limits import DevelopmentTokenLimits
    path = Path(__file__).resolve().parents[2] / "configs/online_token_limits.smoke.json"
    payload = json.loads(path.read_bytes())
    for changed in ({**payload, "roles": payload["roles"][:2]}, {**payload, "development_only": False},
                    {**payload, "status": "calibrated"}):
        with pytest.raises(ValueError):
            DevelopmentTokenLimits.model_validate(changed)


def test_smoke_limits_reach_requests_and_preserve_context_guard(tmp_path):
    from toolsandbox_pipeline.online.token_limits import load_token_limits
    from toolsandbox_pipeline.online.prompt_builder import prepare_request
    from toolsandbox_pipeline.online.prompt_loader import load_prompts
    from toolsandbox_pipeline.retrieval.index import file_hash
    from toolsandbox_pipeline.schemas.runtime import QwenConfig
    from tests.online.test_prompt_builder import contexts, MAPPING
    root = Path(__file__).resolve().parents[2]
    path = root / "configs/online_token_limits.smoke.json"
    digest = file_hash(path.read_bytes())
    config = load_token_limits(path, expected_sha256=digest)
    prompts = load_prompts(root)
    for context, prompt, limit in zip(contexts(tmp_path), prompts, (512, 768, 512)):
        kwargs = dict(prompt=prompt, token_limits=config, token_limit_config_sha256=digest,
                      qwen_config=QwenConfig(structured_output_wire_mode="guided_json", context_limit=2048),
                      canonical_to_agent=MAPPING, mode="calibration")
        request = prepare_request(context, **kwargs, count_prompt_tokens=lambda messages: 1000)
        assert request.max_tokens == request.selected_limit.max_tokens == limit
        assert request.token_limit_config_status == "calibration"
        with pytest.raises(ValueError, match="context limit"):
            prepare_request(context, **kwargs, count_prompt_tokens=lambda messages: 2048 - limit + 1)
