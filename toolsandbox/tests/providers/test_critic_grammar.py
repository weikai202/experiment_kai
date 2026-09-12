"""Portable grammar contract tests; deployment matcher exhaustiveness is separate."""
import hashlib
import json
from toolsandbox_pipeline.providers.critic_grammar import (
    CRITIC_GRAMMAR_VERSION, build_critic_grammar, critic_grammar_sha256,
)
from toolsandbox_pipeline.schemas.critic import CriticErrorCode


def test_version_hash_and_no_runtime_io():
    grammar = build_critic_grammar()
    assert grammar == build_critic_grammar()
    assert CRITIC_GRAMMAR_VERSION == 'ordered_unique_grammar_v1'
    assert critic_grammar_sha256() == 'sha256:' + hashlib.sha256(grammar.encode()).hexdigest()
    assert grammar.endswith('\n')


def test_error_language_is_exactly_all_nonempty_unique_ordered_subsets():
    # Read actual emitted productions rather than reproducing the generator.
    rules = dict(line.split(' ::= ', 1) for line in build_critic_grammar().splitlines())
    codes = [code.value for code in CriticErrorCode]
    decoder = json.JSONDecoder()
    edges = {}
    for name, production in rules.items():
        if name != 'error_array' and not name.startswith('tail_'):
            continue
        edges[name] = []
        for alternative in production.split(' | '):
            terminal, end = decoder.raw_decode(alternative)
            target = alternative[end:].strip()
            if terminal == ']':
                assert not target
                edges[name].append((None, None))
            else:
                value = json.loads(terminal[1:] if name == 'error_array' else terminal[2:])
                edges[name].append((value, target))
    visited = set()
    def visit(name, prefix):
        for value, target in edges[name]:
            if value is None:
                assert prefix and prefix not in visited
                assert list(prefix) == sorted(set(prefix), key=codes.index)
                visited.add(prefix)
            else:
                assert value in codes
                assert not prefix or codes.index(value) > codes.index(prefix[-1])
                visit(target, prefix + (value,))
    visit('error_array', ())
    assert len(visited) == 2 ** len(codes) - 1
    assert tuple(codes) in visited
    assert all((code,) in visited for code in codes)
    assert len(edges) == len(codes) + 1  # compact DAG, no subset enumeration
