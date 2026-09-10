"""Strict, uncached embeddings. Callers persist the first observed dimension."""
import math
from toolsandbox_pipeline.schemas.runtime import EmbeddingConfig
from toolsandbox_pipeline.providers.contracts import ProviderRole, physical_request


class EmbeddingGateway:
    def __init__(self, config: EmbeddingConfig, *, transport=None, recorder=None):
        self.config = config
        self._transport = transport
        self.recorder = recorder
        self._dimension = config.expected_dimension

    @property
    def dimension(self):
        return self._dimension

    def embed(self, context, inputs):
        batch = None

        def prepare():
            nonlocal batch
            if context.role is not ProviderRole.EMBEDDING:
                raise ValueError("embedding role required")
            batch = [inputs] if type(inputs) is str else inputs
            if type(batch) is not list or not 1 <= len(batch) <= 2048:
                raise ValueError("invalid embedding batch")
            if any(type(item) is not str or not item for item in batch):
                raise ValueError("non-empty strings required")
            sizes = [len(item.encode("utf-8")) for item in batch]
            if any(size > 8000 for size in sizes) or sum(sizes) > 280000:
                raise ValueError("embedding byte limit exceeded")
            request = dict(model=self.config.model, input=inputs if type(inputs) is str else list(inputs), encoding_format="float")
            if self._transport is None:
                from toolsandbox_pipeline.providers.openai_clients import create_transport
                self._transport = create_transport(self.config, embedding=True)
            return self._transport, request

        def validate(raw, data):
            if raw.get("model") != self.config.model or raw.get("object") != "list":
                raise ValueError("embedding model or object mismatch")
            items = raw.get("data")
            if type(items) is not list or len(items) != len(batch):
                raise ValueError("embedding count mismatch")
            vectors = []
            dimension = self._dimension
            for index, item in enumerate(items):
                if type(item) is not dict or item.get("object") != "embedding" or type(item.get("index")) is not int or item["index"] != index:
                    raise ValueError("invalid embedding index/order")
                vector = item.get("embedding")
                if type(vector) is not list or not vector:
                    raise ValueError("missing vector")
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                    raise ValueError("invalid vector value")
                if dimension is None:
                    dimension = len(vector)
                if len(vector) != dimension:
                    raise ValueError("embedding dimension drift")
                vectors.append(tuple(float(v) for v in vector))
            # Commit dimension only after the entire response has validated.
            self._dimension = dimension
            return tuple(vectors)

        return physical_request(context, self.config.model, prepare=prepare,
                                validate=validate, embedding=True, recorder=self.recorder)
