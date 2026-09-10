import json

import pytest

from toolsandbox_pipeline.online.tool_metadata import load_tool_metadata
from toolsandbox_pipeline.schemas.tool_metadata import ControllerToolMetadata, ToolEffect, ToolRisk


def test_manifest_rejects_tamper_unsorted_duplicate_and_extra(tmp_path):
    source = load_tool_metadata()
    manifest = json.loads(open("configs/tool_metadata/manifest.json", encoding="utf-8").read())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for records in ((source[1], source[0], *source[2:]), (*source, source[-1])):
        path = tmp_path / "metadata.jsonl"
        path.write_text("\n".join(x.model_dump_json() for x in records) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_tool_metadata(path, manifest_path)


def test_future_external_write_contract_is_fail_closed():
    record = ControllerToolMetadata(canonical_tool_name="future", effect=ToolEffect.EXTERNAL_WRITE, risk=ToolRisk.HIGH, prerequisites=(), parallel_safe=False, read_resources=(), write_resources=(), critic_required=True, agent_forbidden=False)
    assert record.effect.value == "external_write"
    with pytest.raises(ValueError):
        ControllerToolMetadata(canonical_tool_name="future", effect=ToolEffect.EXTERNAL_WRITE, risk=ToolRisk.MEDIUM, prerequisites=(), parallel_safe=True, read_resources=(), write_resources=(), critic_required=False, agent_forbidden=False)


def test_invalid_resource_template_fails_loader(tmp_path):
    records = [item.model_dump(mode="json") for item in load_tool_metadata()]
    records[0]["read_resources"] = ["bad:{value}"]
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text("\n".join(json.dumps(x) for x in records) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(open("configs/tool_metadata/manifest.json", encoding="utf-8").read(), encoding="utf-8")
    with pytest.raises(ValueError):
        load_tool_metadata(metadata_path, manifest_path)
