from types import SimpleNamespace
import pytest
from toolsandbox_pipeline.toolsandbox_adapter.native_external_scope import NativeExternalCallScope, native


def test_native_calls_repeat_without_dedup_and_restore_on_failure(monkeypatch):
    calls = []
    scope = NativeExternalCallScope(('one', 'two'))
    def original(console, message, role):
        calls.append(scope.current_call_id())
        if message.content == 'raise':
            raise LookupError('native error')
        return message
    monkeypatch.setattr(native, 'respond_to_single_message', original)
    with scope:
        for call_id in ('one', 'two', 'one'):
            native.respond_to_single_message(None, SimpleNamespace(openai_tool_call_id=call_id, content='ok'), None)
        with pytest.raises(LookupError):
            native.respond_to_single_message(None, SimpleNamespace(openai_tool_call_id='two', content='raise'), None)
        with pytest.raises(RuntimeError):
            scope.current_call_id()
        with pytest.raises(ValueError):
            native.respond_to_single_message(None, SimpleNamespace(openai_tool_call_id='unknown'), None)
    assert native.respond_to_single_message is original
    assert calls == ['one', 'two', 'one', 'two']


def test_cross_thread_rejected_without_native_execution(monkeypatch):
    import threading
    errors = []
    monkeypatch.setattr(native, 'respond_to_single_message', lambda *args: pytest.fail('wrong thread ran native'))
    with NativeExternalCallScope(('one',)):
        def worker():
            try:
                native.respond_to_single_message(None, SimpleNamespace(openai_tool_call_id='one'), None)
            except RuntimeError as error:
                errors.append(type(error).__name__)
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert errors == ['RuntimeError']


def test_repeated_native_call_produces_separate_boundary_attempts(tmp_path, monkeypatch):
    from tests.reproducibility.test_rapidapi_boundary import make_store, context, upstream
    from tool_sandbox.common.execution_context import ExecutionContext, new_context
    from toolsandbox_pipeline.reproducibility.rapidapi_boundary import RapidAPIBoundary
    manifest, backend_hash, store, _ = make_store(tmp_path)
    attempts = []
    scope = NativeExternalCallScope(('one',))
    monkeypatch.setattr(native, 'respond_to_single_message', lambda *args: upstream.convert_currency(2, 'usd', 'eur'))
    with scope, RapidAPIBoundary(mode='replay', profile='strict_replay', backend_manifest=manifest,
            backend_manifest_sha256=backend_hash, fixture_store=store,
            context_provider=lambda: context(backend_hash, store.manifest_sha256, call=scope.current_call_id()),
            attempt_sink=attempts.append, fixture_miss_sink=lambda _: pytest.fail('fixture miss')):
        with new_context(ExecutionContext()):
            for _ in range(2):
                assert native.respond_to_single_message(None, SimpleNamespace(openai_tool_call_id='one'), None) == 1.5
    assert len(attempts) == 2
    assert attempts[0].attempt_id != attempts[1].attempt_id
    assert all(item.context.logical_tool_call_id == 'one' for item in attempts)
