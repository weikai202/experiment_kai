"""Scoped call identity around the untouched native single-message executor."""
from contextlib import AbstractContextManager
import threading

from tool_sandbox.roles import execution_environment as native


class NativeExternalCallScope(AbstractContextManager):
    """Observe each actual invocation, including repeated native batch permutations.

    No execution is suppressed or deduplicated. One native logical call can
    produce several separately recorded HTTP attempts in the live boundary.
    """
    _lock = threading.Lock()

    def __init__(self, call_ids):
        self.call_ids = tuple(call_ids)
        if not self.call_ids or len(set(self.call_ids)) != len(self.call_ids):
            raise ValueError('unique bound native call IDs required')
        self._current = None
        self._thread = None

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('native external call scope already active')
        self._thread = threading.get_ident()
        self._original = native.respond_to_single_message
        def wrapped(interactive_console, message, role_type):
            if threading.get_ident() != self._thread or self._current is not None:
                raise RuntimeError('native external call scope thread/reentry violation')
            call_id = message.openai_tool_call_id
            if call_id not in self.call_ids:
                raise ValueError('native call is not in the committed binding')
            self._current = call_id
            try:
                return self._original(interactive_console, message, role_type)
            finally:
                self._current = None
        self._wrapped = wrapped
        native.respond_to_single_message = wrapped
        return self

    def current_call_id(self):
        if threading.get_ident() != self._thread or self._current is None:
            raise RuntimeError('external read outside a bound native call')
        return self._current

    def __exit__(self, *exc):
        try:
            native.respond_to_single_message = self._original
            self._current = None
            self._thread = None
        finally:
            self._lock.release()
        return False
