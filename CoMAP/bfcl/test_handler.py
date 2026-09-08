import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from .handler import CoMAPHandler, action_from_message

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mkdir",
            "description": "Create directory",
            "parameters": {
                "type": "object",
                "properties": {"dir_name": {"type": "string"}},
                "required": ["dir_name"],
            },
        },
    }
]


def completion(content=None, calls=None):
    return ChatCompletion.model_validate(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": calls,
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def tool_completion(name="mkdir", arguments=None):
    return completion(
        calls=[
            {
                "id": "draft-id",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments or {"dir_name": "wrong"}),
                },
            }
        ]
    )


def review(decision="REVISE", probability=0.9, name="mkdir"):
    return completion(
        json.dumps(
            {
                "reflection": "Use the requested directory.",
                "decision": decision,
                "revise_probability": probability,
                "final_action": {
                    "content": "",
                    "tool_calls": [{"name": name, "arguments": {"dir_name": "right"}}],
                },
            }
        )
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return next(self.responses)


@pytest.fixture
def handler(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test")
    monkeypatch.setenv("COMAP_WM_MODEL", "world-model")
    agent = CoMAPHandler("policy", 0, "CoMAP-FC")
    agent.client.close()
    agent.wm_client.close()
    return agent


def run_step(handler, reflection):
    handler.client = FakeClient([tool_completion(), reflection])
    handler.wm_client = FakeClient([completion("Hypothetical directory created.")])
    data = {"message": [{"role": "user", "content": "Create right"}], "tools": TOOLS}
    original = deepcopy(data)
    response, latency = handler._query_FC(data)
    assert data["message"] == original["message"]
    assert data["tools"] == original["tools"]
    assert latency >= 0
    return response


def test_revision_and_accounting(handler):
    response = run_step(handler, review())
    parsed = handler._parse_query_response_FC(response)
    assert handler.decode_execute(parsed["model_responses"], False) == [
        "mkdir(dir_name='right')"
    ]
    assert parsed["input_token"] == 30
    assert parsed["output_token"] == 15
    assert parsed["tool_call_ids"][0] != "draft-id"
    assert json.loads(parsed["reasoning_content"])["used_revised_action"]


def test_qwen_options_reach_all_three_calls(handler):
    options = {"chat_template_kwargs": {"enable_thinking": False}}
    handler.policy_extra_body = options
    handler.wm_extra_body = options
    response = run_step(handler, review())
    for request in handler.client.requests + handler.wm_client.requests:
        assert request["extra_body"] == options
    assert response.trace["policy_extra_body"] == options
    assert response.trace["wm_extra_body"] == options


@pytest.mark.parametrize(
    "reflection",
    [
        review("KEEP"),
        review(probability=0.5),
        review(name="unknown"),
        review(probability=float("nan")),
        review(probability=True),
        completion("not JSON"),
        completion("[]"),
        completion("{}"),
    ],
)
def test_keep_and_invalid_reflection_fall_back(handler, reflection):
    response = run_step(handler, reflection)
    assert not response.trace["used_revised_action"]
    assert response.response.choices[0].message.tool_calls[0].id == "draft-id"


def test_text_reply_skips_prediction(handler):
    handler.client = FakeClient([completion("Which directory?")])
    handler.wm_client = FakeClient([])
    response, _ = handler._query_FC({"message": [], "tools": TOOLS})
    assert response.input_tokens == 10
    assert not handler.wm_client.requests
    assert (
        handler._parse_query_response_FC(response)["model_responses"]
        == "Which directory?"
    )


def test_parallel_actions(handler):
    reflection = review()
    content = json.loads(reflection.choices[0].message.content)
    content["final_action"]["tool_calls"].append(
        {"name": "mkdir", "arguments": {"dir_name": "second"}}
    )
    reflection.choices[0].message.content = json.dumps(content)
    response = run_step(handler, reflection)
    action = action_from_message(response.response.choices[0].message)
    assert len(action["tool_calls"]) == 2


def test_official_multi_turn_executor_uses_only_final_action(handler):
    handler.client = FakeClient(
        [
            tool_completion(),
            review(),
            completion("Created right."),
            tool_completion("ls", {"a": False}),
            review("KEEP"),
            completion("Listed files."),
        ]
    )
    handler.wm_client = FakeClient(
        [completion("PREDICTION_ONLY"), completion("PREDICTION_ONLY")]
    )
    entry = {
        "id": "multi_turn_base_987654321",
        "question": [
            [{"role": "user", "content": "Create right"}],
            [{"role": "user", "content": "List files"}],
        ],
        "function": [
            TOOLS[0]["function"],
            {
                "name": "ls",
                "description": "List files",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "boolean"}},
                    "required": [],
                },
            },
        ],
        "initial_config": {"GorillaFileSystem": {}},
        "involved_classes": ["GorillaFileSystem"],
    }
    result, metadata = handler.inference(
        entry, include_input_log=True, exclude_state_log=False
    )
    assert len(result) == 2
    assert "right" in str(result[0])
    assert "wrong" not in str(result)
    history = handler.client.requests[-1]["messages"]
    observations = [
        message["content"] for message in history if message["role"] == "tool"
    ]
    assert any("right" in observation for observation in observations)
    assert all("PREDICTION_ONLY" not in observation for observation in observations)
    assert metadata["input_token_count"] == [[30, 10], [30, 10]]


def test_single_turn_official_inference(handler):
    handler.client = FakeClient([tool_completion(), review()])
    handler.wm_client = FakeClient([completion("Hypothetical")])
    result, metadata = handler.inference(
        {
            "id": "simple_python_987654321",
            "function": [TOOLS[0]["function"]],
            "question": [[{"role": "user", "content": "Create right"}]],
        },
        True,
        True,
    )
    assert handler.decode_ast(result, "Python", False) == [
        {"mkdir": {"dir_name": "right"}}
    ]
    assert metadata["input_token_count"] == 30


def test_official_ast_scoring(handler):
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    response = run_step(handler, review())
    parsed = handler._parse_query_response_FC(response)
    decoded = handler.decode_ast(parsed["model_responses"], "Python", False)
    score = ast_checker(
        [TOOLS[0]["function"]],
        decoded,
        [{"mkdir": {"dir_name": ["right"]}}],
        Language.PYTHON,
        "simple_python",
        "CoMAP-FC",
    )
    assert score["valid"], score
