"""Pure validation of a complete, ordered Skill history."""
from . import views
from toolsandbox_pipeline.schemas.skill import SkillRecord, validate_skill_inventory


def validate_skills(records: tuple[SkillRecord, ...], inventory: tuple[str, ...]):
    keys = [(r.skill_id.encode("utf-8"), int(r.version[3:])) for r in records]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError("unordered or duplicate Skill versions")
    groups = {}
    for record in records:
        validate_skill_inventory(record, inventory)
        groups.setdefault(record.skill_id, []).append(record)
    for history in groups.values():
        if history[-1].status != "active" or any(r.status != "deprecated" for r in history[:-1]):
            raise ValueError("exactly the latest Skill version must be active")
    return tuple(history[-1] for history in groups.values())
