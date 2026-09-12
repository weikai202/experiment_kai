"""Non-secret, frozen provider identities; deployment values are never guessed."""
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, field_validator, model_serializer, model_validator
from toolsandbox_pipeline.schemas.base import StrictModel

NonEmpty = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]


def validate_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme in ("http", "https") and parsed.hostname
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment and parsed.port != 0)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("invalid non-secret endpoint")
    return value


class RuntimeConfig(StrictModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    timeout_seconds: Annotated[float, Field(gt=0)] = 60.0

    def validate_external(self) -> None:
        """Validate deployment before any credential is retrieved."""


class QwenConfig(RuntimeConfig):
    provider: Literal["vllm_openai_compatible"] = "vllm_openai_compatible"
    model: Literal["Qwen/Qwen3-32B"] = "Qwen/Qwen3-32B"
    temperature: Literal[0.0] = 0.0
    seed: Literal[0] = 0
    enable_thinking: Literal[False] = False
    structured_output_wire_mode: Literal["guided_json", "structured_outputs_json"]
    critic_structured_output_mode: Literal["json_schema", "ordered_unique_grammar_v1"] = "json_schema"
    critic_grammar_sha256: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] | None = None
    base_url_env: Literal["QWEN_BASE_URL"] = "QWEN_BASE_URL"
    api_key_env: Literal["QWEN_API_KEY"] = "QWEN_API_KEY"
    allow_empty_local_key: bool = False
    expected_vllm_version: NonEmpty | None = None
    container_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] | None = None
    served_model_id: NonEmpty | None = None
    structured_output_backend: NonEmpty | None = None
    generation_config_policy: NonEmpty | None = None
    server_launch_configuration: NonEmpty | None = None
    context_limit: PositiveInt | None = None
    output_limit: PositiveInt | None = None

    @field_validator("temperature", "seed", "enable_thinking", mode="before")
    @classmethod
    def strict_constants(cls, value, info):
        expected = {"temperature": float, "seed": int, "enable_thinking": bool}
        if type(value) is not expected[info.field_name]:
            raise ValueError("invalid decoding type")
        return value

    @model_validator(mode="after")
    def limits(self):
        if self.context_limit and self.output_limit and self.output_limit > self.context_limit:
            raise ValueError("output limit exceeds context limit")
        self.validate_critic_structured_output()
        return self

    def validate_critic_structured_output(self) -> None:
        if self.critic_structured_output_mode == "json_schema":
            if self.critic_grammar_sha256 is not None:
                raise ValueError("JSON-schema Critic mode cannot bind a grammar hash")
            return
        if (self.critic_structured_output_mode != "ordered_unique_grammar_v1"
                or self.structured_output_wire_mode != "structured_outputs_json"
                or self.structured_output_backend != "xgrammar"):
            raise ValueError("Critic grammar requires explicit xgrammar structured outputs")
        from toolsandbox_pipeline.providers.critic_grammar import critic_grammar_sha256
        if self.critic_grammar_sha256 != critic_grammar_sha256():
            raise ValueError("Critic grammar hash mismatch")

    @model_serializer(mode="wrap")
    def historical_json_schema_identity(self, handler):
        payload = handler(self)
        if self.critic_structured_output_mode == "json_schema" and self.critic_grammar_sha256 is None:
            payload.pop("critic_structured_output_mode", None)
            payload.pop("critic_grammar_sha256", None)
        return payload

    def validate_external(self) -> None:
        self.validate_critic_structured_output()
        if any(getattr(self, name) is None for name in (
            "expected_vllm_version", "container_digest", "served_model_id",
            "structured_output_backend", "generation_config_policy",
            "server_launch_configuration", "context_limit", "output_limit",
        )):
            raise ValueError("unresolved Qwen deployment")


class EmbeddingConfig(RuntimeConfig):
    provider: Literal["openai"] = "openai"
    model: Literal["text-embedding-3-small"] = "text-embedding-3-small"
    base_url: str = Field(default="https://api.openai.com/v1", repr=False)
    base_url_manifest_identity: NonEmpty | None = None
    api_key_env: Literal["OPENAI_API_KEY"] = "OPENAI_API_KEY"
    encoding_format: Literal["float"] = "float"
    expected_dimension: PositiveInt | None = None

    _endpoint = field_validator("base_url")(validate_endpoint)

    @model_validator(mode="after")
    def recorded_endpoint(self):
        if self.base_url != "https://api.openai.com/v1" and not self.base_url_manifest_identity:
            raise ValueError("alternate endpoint requires manifest identity")
        return self


class UserSimulatorConfig(RuntimeConfig):
    provider: Literal["openai"] = "openai"
    model: Literal["gpt-4o-mini-2024-07-18"] = "gpt-4o-mini-2024-07-18"
    base_url: Literal["https://api.openai.com/v1"] = "https://api.openai.com/v1"
    api_key_env: Literal["OPENAI_API_KEY"] = "OPENAI_API_KEY"


class RoleTokenLimitConfig(StrictModel):
    """Versioned selection, not a calibration engine or request-time adaptation.

    The calibration controller owns the evidence manifest and must verify its
    contents before selecting a calibrated value for a formal run.
    """
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    version: NonEmpty
    role: Literal["vanilla", "policy", "critic", "revision", "memory_candidate", "memory_review",
                  "failure_mode_update", "skill_candidate"]
    stage: Literal["bootstrap", "calibration", "calibrated"]
    max_tokens: PositiveInt
    evidence_manifest_identity: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] | None = None

    @model_validator(mode="after")
    def selected_limit(self):
        from toolsandbox_pipeline.providers.contracts import ROLE_BOOTSTRAP_MAX_TOKENS, ProviderRole
        if self.stage == "bootstrap":
            if self.max_tokens != ROLE_BOOTSTRAP_MAX_TOKENS[ProviderRole(self.role)]:
                raise ValueError("bootstrap ceiling mismatch")
        elif self.evidence_manifest_identity is None:
            raise ValueError("calibration manifest required")
        if self.stage == "calibrated" and self.max_tokens % 64:
            raise ValueError("calibrated ceiling must be a multiple of 64")
        return self
