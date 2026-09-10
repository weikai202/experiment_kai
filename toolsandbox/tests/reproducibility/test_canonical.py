from copy import deepcopy
from datetime import datetime
from decimal import Decimal
import math

import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.reproducibility import (
    canonical_json_bytes,
    canonical_sha256,
    compute_state_id,
    verify_state_id,
)


def test_canonical_json_exact_bytes_unicode_nested_and_immutable():
    payload = {"z": [True, None, {"β": 2.5}], "a": "\u96ea"}
    original = deepcopy(payload)
    assert canonical_json_bytes(payload) == '{"a":"\u96ea","z":[true,null,{"β":2.5}]}'.encode()
    assert payload == original


def test_fixed_hash_fixture():
    payload = {"a": 1}
    assert canonical_json_bytes(payload) == b'{"a":1}'
    assert canonical_sha256(payload) == "sha256:015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"


def test_mapping_insertion_order_does_not_change_id():
    assert compute_state_id({"a": 1, "b": 2}) == compute_state_id({"b": 2, "a": 1})


@pytest.mark.parametrize(
    "payload",
    [
        {"x": math.nan}, {"x": math.inf}, {"x": -math.inf},
        {1: "bad"}, {"x": b"bad"}, {"x": Decimal("1")},
        {"x": datetime(2026, 1, 1)}, {"x": object()}, {"x": (1, 2)},
    ],
)
def test_unsupported_values_fail(payload):
    with pytest.raises((ValidationError, TypeError, ValueError)):
        canonical_json_bytes(payload)


def test_state_id_is_non_self_referential_and_verification_is_immutable():
    payload = {"episode_id": "e1", "nested": {"value": 1}}
    original = deepcopy(payload)
    state_id = compute_state_id(payload)
    assert payload == original
    with pytest.raises(ValueError):
        compute_state_id({**payload, "state_id": state_id})

    full = {**payload, "state_id": state_id}
    full_original = deepcopy(full)
    assert verify_state_id(full) is True
    assert full == full_original
    full["nested"]["value"] = 2
    assert verify_state_id(full) is False


@pytest.mark.parametrize("payload", [{}, {"state_id": None}, {"state_id": 1}])
def test_verify_requires_string_state_id(payload):
    with pytest.raises(ValueError):
        verify_state_id(payload)
