from collections import Counter

from toolsandbox_pipeline.online.tool_metadata import EXPECTED_EFFECT_COUNTS, UPSTREAM_COMMIT, installed_upstream_commit, load_tool_metadata, public_tool_inventory


def test_exact_inventory_commit_order_and_effects():
    records = load_tool_metadata()
    names = tuple(x.canonical_tool_name for x in records)
    assert len(names) == 34
    assert names == tuple(sorted(set(names))) == public_tool_inventory()
    assert installed_upstream_commit() == UPSTREAM_COMMIT
    assert Counter(x.effect for x in records) == Counter(EXPECTED_EFFECT_COUNTS)


def test_representative_effects_and_mutation_families():
    records = {x.canonical_tool_name: x for x in load_tool_metadata()}
    assert records["search_contacts"].effect.value == "sandbox_read"
    assert records["convert_currency"].effect.value == "external_read"
    assert records["end_conversation"].agent_forbidden
    for name in ("add_contact", "modify_contact", "remove_contact", "send_message_with_phone_number", "add_reminder", "modify_reminder", "remove_reminder", "set_wifi_status"):
        assert records[name].effect.value == "sandbox_write"
