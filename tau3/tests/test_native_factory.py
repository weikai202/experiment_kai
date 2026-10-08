import sys
import types

from tau3_evolution.native_adapter import AssistantAction
from tau3_evolution.native_runtime import make_native_agent


class FakeHalfDuplexAgent:
    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, tools, domain_policy):
        self.tools = tools
        self.domain_policy = domain_policy


class FakeAssistantMessage:
    def __init__(self, role, content=None, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []


class FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.name = name
        self.arguments = arguments


class Tool:
    openai_schema = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }


class Policy:
    def __init__(self):
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed

    def respond(self, **kwargs):
        return AssistantAction(content="done")


def test_lazy_factory_builds_real_half_duplex_subclass_and_orchestrator_shape(monkeypatch):
    modules = {
        "tau2": types.ModuleType("tau2"),
        "tau2.agent": types.ModuleType("tau2.agent"),
        "tau2.agent.base_agent": types.ModuleType("tau2.agent.base_agent"),
        "tau2.data_model": types.ModuleType("tau2.data_model"),
        "tau2.data_model.message": types.ModuleType("tau2.data_model.message"),
    }
    modules["tau2.agent.base_agent"].HalfDuplexAgent = FakeHalfDuplexAgent
    modules["tau2.data_model.message"].AssistantMessage = FakeAssistantMessage
    modules["tau2.data_model.message"].ToolCall = FakeToolCall
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    policy = Policy()
    agent = make_native_agent([Tool()], "domain policy", policy)
    assert isinstance(agent, FakeHalfDuplexAgent)
    agent.set_seed(17)
    state = agent.get_init_state()
    response, state = agent.generate_next_message({"role": "user", "content": "hello"}, state)
    assert response.role == "assistant" and response.content == "done"
    assert not agent.is_stop(response)
    agent.stop(response, state)
    assert policy.seed == 17
