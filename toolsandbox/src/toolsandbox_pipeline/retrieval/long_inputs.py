"""Lossless UTF-8 chunking and token-weighted embedding aggregation, version 1."""
import math
from toolsandbox_pipeline.providers.embedding_limits import (
    MAX_INPUT_TOKENS, MAX_BATCH_BYTES, embedding_encoding, validate_embedding_input,
)

STRATEGY = 'utf8-token-chunks-weighted-l2-v1'


def validate_retrieval_input(text):
    if type(text) is not str or not text:
        raise ValueError('retrieval input must be a non-empty string')
    if len(text.encode('utf-8')) > MAX_BATCH_BYTES:
        raise ValueError('retrieval byte resource limit exceeded')
    return text


def chunks(text):
    validate_retrieval_input(text)
    encoding = embedding_encoding()
    tokens = encoding.encode_ordinary(text)
    if len(tokens) <= MAX_INPUT_TOKENS:
        return (text,), (len(tokens),)
    output, weights = [], []
    # Token boundaries may bisect a UTF-8 code point. Move the end backwards
    # until strict decoding is possible; never insert replacement characters.
    start = 0
    while start < len(tokens):
        end = min(start + MAX_INPUT_TOKENS, len(tokens))
        while end > start:
            try:
                piece = b''.join(encoding.decode_single_token_bytes(t) for t in tokens[start:end]).decode('utf-8')
                validate_embedding_input(piece)
                break
            except (UnicodeDecodeError, ValueError):
                end -= 1
        if end == start:
            raise ValueError('cannot form valid embedding chunk')
        output.append(piece)
        weights.append(len(encoding.encode_ordinary(piece)))
        start = end
    if ''.join(output) != text:
        raise ValueError('embedding chunk reconstruction mismatch')
    return tuple(output), tuple(weights)


def aggregate(vectors, weights):
    if len(vectors) != len(weights) or not vectors:
        raise ValueError('embedding aggregation count mismatch')
    if len(vectors) == 1:
        return vectors[0]
    total = sum(weights)
    mean = tuple(math.fsum(v[i] * w / total for v, w in zip(vectors, weights))
                 for i in range(len(vectors[0])))
    norm = math.hypot(*mean)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('invalid aggregate embedding')
    return tuple(v / norm for v in mean)
