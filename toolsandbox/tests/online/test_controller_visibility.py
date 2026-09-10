from toolsandbox_pipeline.online.controller import Controller


def test_controller_surface_has_no_prompt_or_execution_method():
    controller = Controller()
    assert not hasattr(controller, "execute")
    assert not hasattr(controller, "to_prompt")
    assert "tool_trace" not in repr(controller)
    assert "evaluator" not in repr(controller)
