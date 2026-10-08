from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import sha256_json

BFCL_VARIANTS = ("base", "miss_func", "miss_param", "long_context")
EXPECTED_FAMILY_COUNT = 200


@dataclass(frozen=True)
class VariantCase:
    case_id: str
    variant: str
    source_index: int
    path: tuple[str, ...]
    involved_classes: tuple[str, ...]
    excluded_functions: tuple[str, ...]
    function_surface_sha256: str | None
    function_names: tuple[str, ...]
    delayed_function_names: tuple[str, ...]
    turn_count: int
    record_sha256: str

    @classmethod
    def from_record(cls, variant: str, record: dict[str, Any]) -> "VariantCase":
        if variant not in BFCL_VARIANTS:
            raise ValueError(f"Unsupported BFCL variant: {variant}")
        prefix = f"multi_turn_{variant}_"
        case_id = record.get("id")
        if not isinstance(case_id, str) or not case_id.startswith(prefix):
            raise ValueError("Case ID does not match its declared variant")
        suffix = case_id[len(prefix):]
        if not suffix.isdigit() or str(int(suffix)) != suffix:
            raise ValueError("BFCL case index must be canonical decimal")
        path = record.get("path")
        classes = record.get("involved_classes")
        questions = record.get("question")
        if not isinstance(path, list) or not path or not all(isinstance(x, str) for x in path):
            raise ValueError("Case path must be a non-empty string list")
        if not isinstance(classes, list) or not classes or not all(isinstance(x, str) for x in classes):
            raise ValueError("Case involved_classes must be a non-empty string list")
        if not isinstance(questions, list) or not questions:
            raise ValueError("Case must contain at least one user turn")
        excluded = record.get("excluded_function", [])
        if not isinstance(excluded, list) or not all(isinstance(x, str) for x in excluded):
            raise ValueError("excluded_function must be a string list")
        functions = record.get("function", [])
        if not isinstance(functions, list) or any(not isinstance(x, dict) or not isinstance(x.get("name"), str) for x in functions):
            raise ValueError("function must be a list of named BFCL schemas")
        function_names = tuple(x["name"] for x in functions)
        if len(set(function_names)) != len(function_names):
            raise ValueError("BFCL function names must be unique")
        delayed = record.get("missed_function", {})
        if not isinstance(delayed, dict):
            raise ValueError("missed_function must be a turn map")
        delayed_names = tuple(
            schema["name"]
            for turn in sorted(delayed, key=int)
            for schema in delayed[turn]
            if isinstance(schema, dict) and isinstance(schema.get("name"), str)
        )
        return cls(
            case_id=case_id,
            variant=variant,
            source_index=int(suffix),
            path=tuple(path),
            involved_classes=tuple(classes),
            excluded_functions=tuple(excluded),
            function_surface_sha256=sha256_json(record["function"]) if "function" in record else None,
            function_names=function_names,
            delayed_function_names=delayed_names,
            turn_count=len(questions),
            record_sha256=sha256_json(record),
        )


@dataclass(frozen=True)
class Family:
    family_id: str
    source_index: int
    variants: tuple[VariantCase, ...]
    provenance_sha256: str

    @property
    def tool_dependencies(self) -> frozenset[str]:
        return frozenset(item.rsplit(".", 1)[-1] for item in self.variants[0].path)


def build_families(records_by_variant: dict[str, Iterable[dict[str, Any]]]) -> tuple[Family, ...]:
    if set(records_by_variant) != set(BFCL_VARIANTS):
        raise ValueError("Exactly the four BFCL multi-turn variants are required")
    parsed = {v: [VariantCase.from_record(v, row) for row in rows] for v, rows in records_by_variant.items()}
    for variant, rows in parsed.items():
        if len(rows) != EXPECTED_FAMILY_COUNT:
            raise ValueError(f"{variant} must contain exactly 200 cases")
        indices = [row.source_index for row in rows]
        if len(set(indices)) != EXPECTED_FAMILY_COUNT or set(indices) != set(range(EXPECTED_FAMILY_COUNT)):
            raise ValueError(f"{variant} indices must be exactly 0..199")
    by_variant = {v: {row.source_index: row for row in rows} for v, rows in parsed.items()}
    families: list[Family] = []
    for index in range(EXPECTED_FAMILY_COUNT):
        variants = tuple(by_variant[v][index] for v in BFCL_VARIANTS)
        # The suffix is only an index lookup. Family membership is accepted only if
        # independent benchmark structure agrees across all four source records.
        structure = {
            "source_index": index,
            "path": list(variants[0].path),
            "involved_classes": list(variants[0].involved_classes),
            "excluded_functions": list(variants[0].excluded_functions),
            "function_surface_sha256_by_variant": {case.variant: case.function_surface_sha256 for case in variants},
        }
        for case in variants[1:]:
            if case.path != variants[0].path:
                raise ValueError(f"Family {index} has inconsistent tool path")
            if case.involved_classes != variants[0].involved_classes:
                raise ValueError(f"Family {index} has inconsistent involved classes")
            if case.excluded_functions != variants[0].excluded_functions:
                raise ValueError(f"Family {index} has inconsistent excluded functions")
        base, miss_func, miss_param, long_context = variants
        if any(case.function_surface_sha256 is not None for case in variants):
            if any(case.function_surface_sha256 is None for case in variants):
                raise ValueError(f"Family {index} has incomplete function surfaces")
            base_names = set(base.function_names)
            if set(miss_param.function_names) != base_names or set(long_context.function_names) != base_names:
                raise ValueError(f"Family {index} has inconsistent function names")
            if set(miss_func.function_names) & set(miss_func.delayed_function_names):
                raise ValueError(f"Family {index} exposes delayed functions too early")
            if set(miss_func.function_names) | set(miss_func.delayed_function_names) != base_names:
                raise ValueError(f"Family {index} has inconsistent delayed function surface")
        families.append(Family(f"bfcl_family_{index:03d}", index, variants, sha256_json(structure)))
    return tuple(families)
