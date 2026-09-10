"""All-or-nothing Skill mapping without canonical names in Policy views."""
from typing import Annotated
from pydantic import Field
from toolsandbox_pipeline.online.controller import evaluate_predicate
from toolsandbox_pipeline.online.controller_inputs import RetrievedSkillControllerView, SkillStatus
from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
from toolsandbox_pipeline.schemas.memory import FrozenRecord, GenerationId, Identifier
from toolsandbox_pipeline.schemas.skill import SkillApplicability, SkillCostProfile, SkillRiskProfile, SkillStatePredicate
from toolsandbox_pipeline.schemas.tool_metadata import MetadataPredicate, PredicateOp
from toolsandbox_pipeline.schemas.state import CompactVerifiedState


class RetrievedSkillPolicyView(FrozenRecord):
    rank: Annotated[int, Field(ge=1, le=3)] = 1
    skill_id: Identifier
    version: Identifier
    name: Identifier
    description: Identifier
    applicability: SkillApplicability
    required_inputs: tuple[SkillStatePredicate, ...]
    expected_outputs: tuple[Identifier, ...]
    tool_dependencies: tuple[Identifier, ...]
    success_criteria: tuple[Identifier, ...]
    cost_profile: SkillCostProfile
    risk_profile: SkillRiskProfile
    instruction: Identifier


def controller_view(skill, generation_id):
    def predicates(group, values):
        return tuple(MetadataPredicate.model_validate({
            **p.model_dump(mode="json"),
            "op": PredicateOp(p.op),
            "code": canonical_sha256([skill.skill_id, group, i]),
        }) for i, p in enumerate(values))
    return RetrievedSkillControllerView(
        skill_id=skill.skill_id, version=skill.version, status=SkillStatus(skill.status),
        generation_id=generation_id, tool_dependencies=skill.tool_dependencies,
        applicability_required=predicates("required", skill.applicability.required_state),
        applicability_forbidden=predicates("forbidden", skill.applicability.forbidden_state),
        required_inputs=predicates("inputs", skill.required_inputs),
    )


def skill_views(skill, generation_id: str, state: CompactVerifiedState, canonical_to_agent: dict[str, str]):
    if type(state) is not CompactVerifiedState:
        raise TypeError("CompactVerifiedState required")
    available = {tool.name for tool in state.available_tools}
    if len(set(canonical_to_agent.values())) != len(canonical_to_agent):
        raise ValueError("non-bijective mapping")
    names = tuple(canonical_to_agent.get(name) for name in skill.tool_dependencies)
    if skill.status != "active" or any(name not in available for name in names):
        return None
    controller = controller_view(skill, generation_id)
    payload = state.model_dump(mode="json", by_alias=True)
    if not all(evaluate_predicate(payload, p) for p in controller.applicability_required + controller.required_inputs):
        return None
    if any(evaluate_predicate(payload, p) for p in controller.applicability_forbidden):
        return None
    allowed = set(RetrievedSkillPolicyView.model_fields)
    data = {k: v for k, v in skill.model_dump(mode="json").items() if k in allowed}
    data["tool_dependencies"] = list(names)
    from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes
    policy = RetrievedSkillPolicyView.model_validate_json(canonical_json_bytes(data))
    return policy, controller
