"""Explicit Critic wire grammar with canonical, unique error-code subsets.

This changes only error-code ordering on the wire, not the expressible sets.
No model output is repaired, sorted or deduplicated after generation.
"""
import hashlib
import json

from toolsandbox_pipeline.schemas.critic import CriticErrorCode

CRITIC_GRAMMAR_VERSION = "ordered_unique_grammar_v1"


def build_critic_grammar() -> str:
    """Return deterministic XGrammar EBNF; order follows CriticErrorCode."""
    literal = lambda value: json.dumps(value, ensure_ascii=True)
    codes = tuple(code.value for code in CriticErrorCode)
    lines = ["root ::= accept | revise | uncertain"]
    prefix = lambda verdict: literal('{"verdict": ' + json.dumps(verdict) + ', "predicted_outcome": ')
    effect = literal(', "predicted_effect": ')
    errors = literal(', "error_codes": ')
    correction = literal(', "correction": ')
    end = literal('}')
    lines.extend([
        f'accept ::= {prefix("accept")} outcome {effect} nonempty_string {errors} {literal("[]")} {correction} {literal(chr(34) * 2)} {end}',
        f'revise ::= {prefix("revise")} outcome {effect} nonempty_string {errors} error_array {correction} nonempty_string {end}',
        f'uncertain ::= {prefix("uncertain")} {literal(json.dumps("uncertain"))} {effect} nonempty_string {errors} error_array {correction} nonempty_string {end}',
        'outcome ::= ' + ' | '.join(literal(json.dumps(value)) for value in ('success', 'failure', 'uncertain')),
        'nonempty_string ::= "\\\"" character+ "\\\""',
        r'character ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])',
    ])
    lines.append('error_array ::= ' + ' | '.join(
        literal('[' + json.dumps(code)) + f' tail_{index + 1}'
        for index, code in enumerate(codes)))
    for start in range(1, len(codes) + 1):
        alternatives = [literal(']')]
        alternatives.extend(literal(', ' + json.dumps(codes[index])) + f' tail_{index + 1}'
                            for index in range(start, len(codes)))
        lines.append(f'tail_{start} ::= ' + ' | '.join(alternatives))
    return '\n'.join(lines) + '\n'


def critic_grammar_sha256() -> str:
    return 'sha256:' + hashlib.sha256(build_critic_grammar().encode('utf-8')).hexdigest()
