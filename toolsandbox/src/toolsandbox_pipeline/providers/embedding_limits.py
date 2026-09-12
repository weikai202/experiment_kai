"""Pinned, network-free token validation for text-embedding-3-small."""
from functools import lru_cache
from pathlib import Path
import json

MAX_INPUT_TOKENS = 8191  # One-token margin beneath the documented 8192 ceiling.
MAX_BATCH_BYTES = 280000


@lru_cache(maxsize=1)
def embedding_encoding():
    import tiktoken
    import base64
    from hashlib import sha256
    root = Path(__file__).with_name('tokenizer_data')
    config = json.loads((root / 'cl100k_base.json').read_text())
    raw=(root / 'cl100k_base.tiktoken').read_bytes()
    if sha256(raw).hexdigest()!=config['sha256']:
        raise ValueError('embedding tokenizer hash mismatch')
    ranks={base64.b64decode(token):int(rank) for token,rank in (line.split() for line in raw.splitlines() if line)}
    return tiktoken.Encoding(name=config['name'], pat_str=config['pat_str'],
                            mergeable_ranks=ranks, special_tokens=config['special_tokens'])


def validate_embedding_input(text):
    if type(text) is not str or not text:
        raise ValueError('embedding input must be a non-empty string')
    size = len(text.encode('utf-8'))
    if size > MAX_BATCH_BYTES:
        raise ValueError('embedding byte resource limit exceeded')
    # UTF-8 BPE cannot have more tokens than bytes. Short inputs need no tokenizer.
    if size > MAX_INPUT_TOKENS and len(embedding_encoding().encode_ordinary(text)) > MAX_INPUT_TOKENS:
        raise ValueError('embedding input exceeds 8191 tokens')
    return text
