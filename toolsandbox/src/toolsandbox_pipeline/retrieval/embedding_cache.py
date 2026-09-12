"""SQLite cache with ledger-before-cache commits and no raw input persistence."""
import json
import sqlite3
from pathlib import Path

from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.usage import TokenUsage
from toolsandbox_pipeline.providers.contracts import PhysicalAttemptStatus, ProviderRole, RequestContext
from .contracts import EmbeddingBatchConfig, EmbeddingCacheKey, EmbeddingIdentity, ResolvedEmbedding, RetrievalError, validate_vector
from .queries import input_hash, validate_input
from .long_inputs import chunks, aggregate, STRATEGY

SCHEMA_VERSION = 2
_TABLES = {
    "derivations": "CREATE TABLE derivations (input_sha256 TEXT PRIMARY KEY, strategy TEXT NOT NULL, chunk_hashes_json TEXT NOT NULL, token_weights_json TEXT NOT NULL)",
    "provenance": "CREATE TABLE provenance (attempt_id TEXT PRIMARY KEY, usage_json TEXT NOT NULL, response_hash TEXT NOT NULL)",
    "vectors": "CREATE TABLE vectors (provider TEXT NOT NULL, base_url_identity TEXT NOT NULL, model TEXT NOT NULL, encoding_format TEXT NOT NULL, dimensions_setting TEXT NOT NULL, input_sha256 TEXT NOT NULL, dimension INTEGER NOT NULL, vector_json TEXT NOT NULL, vector_hash TEXT NOT NULL, response_hash TEXT NOT NULL, attempt_id TEXT NOT NULL REFERENCES provenance(attempt_id), PRIMARY KEY(provider, base_url_identity, model, encoding_format, dimensions_setting, input_sha256))",
}


class EmbeddingCache:
    def __init__(self, path: Path, identity: EmbeddingIdentity, dimension: int, *, batches=None):
        path = Path(path)
        if not path.is_absolute() or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
            raise RetrievalError("explicit absolute non-symlink cache path required")
        if type(dimension) is not int or dimension < 1:
            raise RetrievalError("pinned positive dimension required")
        self.identity = EmbeddingIdentity.model_validate(identity)
        self.dimension = dimension
        self.batches = batches or EmbeddingBatchConfig()
        self._db = sqlite3.connect(path)
        try:
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            tables = dict(self._db.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"))
            if version == 0 and not tables:
                with self._db:
                    for sql in _TABLES.values():
                        self._db.execute(sql)
                    self._db.execute("PRAGMA user_version=2")
            elif version != SCHEMA_VERSION or tables != _TABLES:
                raise RetrievalError("unknown cache schema")
            self._db.execute("PRAGMA foreign_keys=ON")
        except Exception:
            self._db.close()
            raise

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def key(self, text):
        return EmbeddingCacheKey(**self.identity.model_dump(), input_sha256=input_hash(text))

    def _get(self, key):
        values = tuple(key.model_dump().values())
        row = self._db.execute("SELECT dimension,vector_json,vector_hash,response_hash,attempt_id FROM vectors WHERE provider=? AND base_url_identity=? AND model=? AND encoding_format=? AND dimensions_setting=? AND input_sha256=?", values).fetchone()
        if row is None:
            return None
        dimension, payload, digest, response_hash, attempt_id = row
        vector = validate_vector(json.loads(payload), self.dimension)
        if dimension != self.dimension or canonical_sha256(list(vector)) != digest:
            raise RetrievalError("corrupt cached vector")
        provenance = self._db.execute("SELECT usage_json,response_hash FROM provenance WHERE attempt_id=?", (attempt_id,)).fetchone()
        if provenance is None or provenance[1] != response_hash:
            raise RetrievalError("missing or corrupt cache provenance")
        from pydantic import TypeAdapter
        from toolsandbox_pipeline.schemas.memory import Digest
        TypeAdapter(Digest).validate_python(response_hash)
        TokenUsage.model_validate_json(provenance[0])
        derivation = self._db.execute("SELECT strategy,chunk_hashes_json,token_weights_json FROM derivations WHERE input_sha256=?", (key.input_sha256,)).fetchone()
        if derivation is None or derivation[0] != STRATEGY:
            raise RetrievalError("missing or incompatible vector derivation")
        hashes, weights = json.loads(derivation[1]), json.loads(derivation[2])
        if not hashes or len(hashes) != len(weights) or any(type(w) is not int or w <= 0 for w in weights):
            raise RetrievalError("invalid vector derivation")
        return ResolvedEmbedding(key=key, vector=vector, cache_hit=True, source_attempt_id=attempt_id)

    def resolve(self, inputs, *, gateway, context_factory, record_durable):
        if type(inputs) not in (list, tuple) or not inputs:
            raise RetrievalError("ordered non-empty input sequence required")
        texts = tuple(validate_input(text) for text in inputs)
        keys = tuple(self.key(text) for text in texts)
        config = gateway.config
        if (config.provider, config.base_url, config.model, config.encoding_format) != (
            self.identity.provider, self.identity.base_url_identity, self.identity.model, self.identity.encoding_format
        ) or config.expected_dimension not in (None, self.dimension):
            raise RetrievalError("gateway/cache identity mismatch")
        resolved, misses = {}, {}
        for text, key in zip(texts, keys):
            digest = key.input_sha256
            if digest in misses and misses[digest] != text:
                raise RetrievalError("input identity collision")
            if digest not in resolved:
                hit = self._get(key)
                if hit is not None:
                    resolved[digest] = hit
                else:
                    misses[digest] = text
        pieces = {digest: chunks(text) for digest, text in misses.items()}
        batches, batch, size, item_count = [], [], 0, 0
        for digest, text in misses.items():
            length = len(text.encode("utf-8"))
            count = len(pieces[digest][0])
            if count > self.batches.max_items or length > self.batches.max_bytes:
                raise RetrievalError("one input exceeds configured embedding batch resources")
            if batch and (item_count + count > self.batches.max_items or size + length > self.batches.max_bytes):
                batches.append(batch)
                batch, size, item_count = [], 0, 0
            batch.append((digest, text))
            size += length
            item_count += count
        if batch:
            batches.append(batch)
        for ordinal, batch in enumerate(batches):
            batch_text = [piece for digest, _ in batch for piece in pieces[digest][0]]
            context = context_factory(ordinal, tuple(batch_text))
            if type(context) is not RequestContext or context.role is not ProviderRole.EMBEDDING:
                raise RetrievalError("prepared embedding RequestContext required")
            response = gateway.embed(context, batch_text)
            attempt = response.attempt
            if attempt.context != context or attempt.status is not PhysicalAttemptStatus.COMPLETED or not attempt.response_hash:
                raise RetrievalError("invalid embedding attempt")
            # The caller's authoritative ledger must be durable even if cache validation fails.
            if record_durable(response) is not True:
                raise RetrievalError("durable ledger acknowledgement required")
            if len(response.value) != len(batch_text):
                raise RetrievalError("partial embedding response")
            raw_vectors = tuple(validate_vector(v, self.dimension) for v in response.value)
            vectors, offset = [], 0
            for digest, _ in batch:
                weights = pieces[digest][1]
                vectors.append(validate_vector(aggregate(raw_vectors[offset:offset+len(weights)], weights), self.dimension))
                offset += len(weights)
            usage = canonical_json_bytes(attempt.metrics.usage.model_dump(mode="json")).decode()
            with self._db:
                self._db.execute("INSERT INTO provenance VALUES (?,?,?)", (context.attempt_id, usage, attempt.response_hash))
                for (digest, text), vector in zip(batch, vectors):
                    key = self.key(text)
                    chunk_texts, weights = pieces[digest]
                    self._db.execute("INSERT INTO derivations VALUES (?,?,?,?)", (
                        digest, STRATEGY, json.dumps([input_hash(t) for t in chunk_texts]), json.dumps(weights)))
                    self._db.execute("INSERT INTO vectors VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                        *key.model_dump().values(), self.dimension,
                        canonical_json_bytes(list(vector)).decode(), canonical_sha256(list(vector)),
                        attempt.response_hash, context.attempt_id,
                    ))
            for (digest, text), vector in zip(batch, vectors):
                resolved[digest] = ResolvedEmbedding(key=self.key(text), vector=vector, cache_hit=False, source_attempt_id=context.attempt_id)
        return tuple(resolved[key.input_sha256] for key in keys)
