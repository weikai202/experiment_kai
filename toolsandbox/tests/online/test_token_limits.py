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
