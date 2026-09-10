import random
import pytest
from toolsandbox_pipeline.schemas.dataset import REGISTRIES, SUFFIXES
from toolsandbox_pipeline.reproducibility.splits import assign_splits, build_family_registry, build_augmented_registry


def families():
    return tuple((f"family-{i:03d}", REGISTRIES[i % 4]) for i in range(129))


def test_family_split_counts_and_order():
    result = assign_splits(families())
    assert [len(result[s]) for s in ("train", "dev", "test")] == [79, 25, 25]
    assert [sum(f.train_shard == shard for f in result["train"]) for shard in range(3)] == [27, 26, 26]
    assert len({f.family_id for group in result.values() for f in group}) == 129
    assert tuple(f.family_id for f in result["test"][:5]) == ("family-048", "family-003", "family-023", "family-114", "family-022")


def test_factories_before_variants_and_random_restoration():
    events = []
    def factory(label):
        def create(**kwargs):
            events.append(label)
            return {name: None for name, source in families() if source == label}
        return create
    registry = build_family_registry({label: factory(label) for label in REGISTRIES}, "DEFAULT")
    assert events == list(REGISTRIES)
    expected = [name for name, _ in registry] + [name + suffix for name, _ in registry for suffix in SUFFIXES[1:]]
    saved = random.getstate()
    def augmented(**kwargs):
        random.random()
        return dict.fromkeys(expected)
    assert tuple(build_augmented_registry(registry, augmented, "DEFAULT")) == tuple(expected)
    assert random.getstate() == saved
    with pytest.raises(ValueError):
        build_augmented_registry(registry, lambda **kw: dict.fromkeys(reversed(expected)), "DEFAULT")
    assert random.getstate() == saved
