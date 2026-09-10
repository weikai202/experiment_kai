"""Family-first split; upstream variant IDs are generated, never parsed."""
import random
from toolsandbox_pipeline.schemas.dataset import FamilyRecord, REGISTRIES, SUFFIXES


def build_family_registry(factories, backend):
    if tuple(factories) != REGISTRIES:
        raise ValueError("four ordered registry factories required")
    families = []
    seen = set()
    for label, factory in factories.items():
        mapping = factory(preferred_tool_backend=backend)
        if type(mapping) is not dict:
            raise ValueError("ordered registry mapping required")
        for name in mapping:
            if type(name) is not str or not name or name in seen:
                raise ValueError("invalid or duplicate family ID")
            name.encode("utf-8")
            seen.add(name)
            families.append((name, label))
    if len(families) != 129:
        raise ValueError("pinned registry requires 129 families")
    return tuple(families)


def assign_splits(families):
    if len(families) != 129 or len({f[0] for f in families}) != 129:
        raise ValueError("exact unique family registry required")
    labels = dict(families)
    ordered = sorted(labels, key=lambda value: value.encode("utf-8"))
    random.Random(0).shuffle(ordered)
    output = {}
    for split, names in (("test", ordered[:25]), ("dev", ordered[25:50]), ("train", ordered[50:])):
        output[split] = tuple(FamilyRecord(
            family_id=name, source_registry=labels[name], split=split,
            train_shard=(0 if index < 27 else 1 if index < 53 else 2) if split == "train" else None,
            ordered_variant_scenario_ids=tuple(name + suffix for suffix in SUFFIXES),
        ) for index, name in enumerate(names))
    return output


def build_augmented_registry(families, factory, backend):
    saved = random.getstate()
    try:
        random.seed(0)
        mapping = factory(preferred_tool_backend=backend)
    finally:
        random.setstate(saved)
    expected = tuple(name for name, _ in families) + tuple(name + suffix for name, _ in families for suffix in SUFFIXES[1:])
    if type(mapping) is not dict or tuple(mapping) != expected:
        raise ValueError("upstream variant count/identity/order mismatch")
    return mapping
