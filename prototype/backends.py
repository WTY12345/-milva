from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from .config import SystemConfig
from .models import ProductRecord, SearchHit, SearchRequest, normalize_vector

try:
    from pymilvus import DataType, MilvusClient
except Exception:  # pragma: no cover - optional import path
    DataType = None
    MilvusClient = None


class BackendUnavailableError(RuntimeError):
    """Raised when the preferred backend cannot be initialized."""


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(item) for item in value]
    if hasattr(value, "items"):
        try:
            return {str(key): _to_plain_data(item) for key, item in value.items()}
        except Exception:
            pass
    if hasattr(value, "__iter__") and not isinstance(value, (bytes, bytearray)):
        try:
            return [_to_plain_data(item) for item in list(value)]
        except Exception:
            pass
    return str(value)


class VectorBackend(ABC):
    name = "base"

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, products: Sequence[ProductRecord]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get(self, item_id: int) -> ProductRecord | None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, item_ids: Sequence[int]) -> int:
        raise NotImplementedError

    @abstractmethod
    def list_products(self, limit: int = 50) -> list[ProductRecord]:
        raise NotImplementedError

    @abstractmethod
    def search(self, request: SearchRequest) -> list[SearchHit]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> dict[str, Any]:
        raise NotImplementedError


class MilvusVectorBackend(VectorBackend):
    name = "milvus"

    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)
        self.client: Any | None = None

    def initialize(self) -> None:
        if MilvusClient is None or DataType is None:
            raise BackendUnavailableError("pymilvus is not available in the current interpreter.")
        try:
            self.client = MilvusClient(self.config.milvus_uri)
        except Exception as exc:
            raise BackendUnavailableError(
                f"Unable to open Milvus backend at '{self.config.milvus_uri}'. "
                f"For local '.db' mode you typically need milvus-lite, while remote mode "
                f"requires a running Milvus service. Original error: {exc}"
            ) from exc
        if not self.client.has_collection(self.config.collection_name):
            self._create_collection()
        else:
            self._ensure_schema_compatible()
        self._load_collection()

    def _create_collection(self) -> None:
        assert self.client is not None
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="item_id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="category_id", datatype=DataType.INT64)
        schema.add_field(field_name="category_name", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="price", datatype=DataType.FLOAT)
        schema.add_field(field_name="product_name", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="brand", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="product_type", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="size", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="purpose", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="target_group", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="market", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.config.vector_dim,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self.config.index_type,
            metric_type=self.config.metric_type,
        )
        self.client.create_collection(
            collection_name=self.config.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _ensure_schema_compatible(self) -> None:
        assert self.client is not None
        description = self.client.describe_collection(self.config.collection_name)
        fields = {field["name"] for field in description.get("fields", [])}
        required = {
            "item_id", "category_id", "category_name", "price", "product_name",
            "brand", "product_type", "size", "purpose", "target_group", "market", "embedding",
        }
        if required - fields:
            self.client.drop_collection(self.config.collection_name)
            self._create_collection()

    def _load_collection(self) -> None:
        assert self.client is not None
        try:
            self.client.load_collection(self.config.collection_name)
        except Exception:
            pass

    def _flush_collection(self) -> None:
        assert self.client is not None
        try:
            self.client.flush(self.config.collection_name)
        except Exception:
            pass

    def reset(self) -> None:
        assert self.client is not None
        if self.client.has_collection(self.config.collection_name):
            self.client.drop_collection(self.config.collection_name)
        self._create_collection()
        self._load_collection()

    def upsert(self, products: Sequence[ProductRecord]) -> dict[str, Any]:
        assert self.client is not None
        payload = [product.to_payload() for product in products]
        started = perf_counter()
        result = self.client.upsert(self.config.collection_name, payload)
        self._flush_collection()
        return {
            "backend": self.name,
            "count": len(payload),
            "latency_ms": round((perf_counter() - started) * 1000.0, 3),
            "raw": _to_plain_data(result),
        }

    def get(self, item_id: int) -> ProductRecord | None:
        assert self.client is not None
        result = self.client.query(
            self.config.collection_name,
            ids=[item_id],
            output_fields=[
                "item_id", "category_id", "category_name", "price", "product_name",
                "brand", "product_type", "size", "purpose", "target_group", "market", "embedding",
            ],
        )
        if not result:
            return None
        return ProductRecord.from_mapping(result[0])

    def delete(self, item_ids: Sequence[int]) -> int:
        assert self.client is not None
        if not item_ids:
            return 0
        result = self.client.delete(self.config.collection_name, ids=[int(item_id) for item_id in item_ids])
        self._flush_collection()
        return int(result.get("delete_count", len(item_ids)))

    def list_products(self, limit: int = 50) -> list[ProductRecord]:
        assert self.client is not None
        rows = self.client.query(
            self.config.collection_name,
            filter="item_id >= 0",
            output_fields=[
                "item_id", "category_id", "category_name", "price", "product_name",
                "brand", "product_type", "size", "purpose", "target_group", "market", "embedding",
            ],
            limit=limit,
        )
        products = [ProductRecord.from_mapping(row) for row in rows]
        products.sort(key=lambda product: product.item_id, reverse=True)
        return products[:limit]

    def search(self, request: SearchRequest) -> list[SearchHit]:
        assert self.client is not None
        filters = self._build_milvus_filter(request)
        result = self.client.search(
            self.config.collection_name,
            data=[request.normalized_vector()],
            filter=filters,
            limit=request.top_k,
            output_fields=[
                "item_id", "category_id", "category_name", "price", "product_name",
                "brand", "product_type", "size", "purpose", "target_group", "market",
            ],
            search_params={"metric_type": self.config.metric_type},
        )
        hits: list[SearchHit] = []
        for row in result[0]:
            entity = row.get("entity", {})
            item_id = row.get("id", entity.get("item_id"))
            hits.append(
                SearchHit(
                    item_id=int(item_id),
                    category_id=int(entity.get("category_id", 0)),
                    category_name=str(entity.get("category_name", "")),
                    price=float(entity.get("price", 0.0)),
                    product_name=str(entity.get("product_name", "")),
                    brand=str(entity.get("brand", "")),
                    product_type=str(entity.get("product_type", "")),
                    size=str(entity.get("size", "")),
                    purpose=str(entity.get("purpose", "")),
                    target_group=str(entity.get("target_group", "")),
                    market=str(entity.get("market", "")),
                    score=float(row.get("distance", 0.0)),
                )
            )
        return hits

    def count(self) -> int:
        assert self.client is not None
        stats = self.client.get_collection_stats(self.config.collection_name)
        for key in ("row_count", "rows", "num_rows"):
            if key in stats:
                return int(stats[key])
        return 0

    def status(self) -> dict[str, Any]:
        assert self.client is not None
        info = self.client.describe_collection(self.config.collection_name)
        info["backend"] = self.name
        info["count"] = self.count()
        return info

    def _build_milvus_filter(self, request: SearchRequest) -> str:
        clauses: list[str] = []
        if request.category_ids:
            ids = ", ".join(str(int(value)) for value in request.category_ids)
            clauses.append(f"category_id in [{ids}]")
        if request.min_price is not None:
            clauses.append(f"price >= {float(request.min_price)}")
        if request.max_price is not None:
            clauses.append(f"price <= {float(request.max_price)}")
        return " and ".join(clauses)


class LocalAnnVectorBackend(VectorBackend):
    name = "local_ann"

    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)
        self._products: dict[int, ProductRecord] = {}
        self._vectors: dict[int, np.ndarray] = {}
        self._buckets: dict[int, set[int]] = defaultdict(set)
        self._id_to_code: dict[int, int] = {}
        self._hyperplanes: np.ndarray | None = None

    def initialize(self) -> None:
        rng = np.random.default_rng(self.config.local_seed)
        hyperplanes = rng.normal(size=(self.config.local_hash_bits, self.config.vector_dim)).astype(np.float32)
        norms = np.linalg.norm(hyperplanes, axis=1, keepdims=True)
        self._hyperplanes = hyperplanes / np.maximum(norms, 1e-8)

    def reset(self) -> None:
        self._products.clear()
        self._vectors.clear()
        self._buckets.clear()
        self._id_to_code.clear()

    def upsert(self, products: Sequence[ProductRecord]) -> dict[str, Any]:
        started = perf_counter()
        for product in products:
            normalized = product.normalized()
            vector = np.asarray(normalized.embedding, dtype=np.float32)
            code = self._hash(vector)
            previous_code = self._id_to_code.get(normalized.item_id)
            if previous_code is not None:
                bucket = self._buckets.get(previous_code)
                if bucket is not None:
                    bucket.discard(normalized.item_id)
            self._products[normalized.item_id] = normalized
            self._vectors[normalized.item_id] = vector
            self._id_to_code[normalized.item_id] = code
            self._buckets[code].add(normalized.item_id)
        return {
            "backend": self.name,
            "count": len(products),
            "latency_ms": round((perf_counter() - started) * 1000.0, 3),
        }

    def get(self, item_id: int) -> ProductRecord | None:
        return self._products.get(item_id)

    def delete(self, item_ids: Sequence[int]) -> int:
        deleted = 0
        for item_id in item_ids:
            item_id = int(item_id)
            code = self._id_to_code.pop(item_id, None)
            if code is not None and code in self._buckets:
                self._buckets[code].discard(item_id)
            if item_id in self._products:
                deleted += 1
                self._products.pop(item_id, None)
                self._vectors.pop(item_id, None)
        return deleted

    def list_products(self, limit: int = 50) -> list[ProductRecord]:
        products = sorted(self._products.values(), key=lambda product: product.item_id, reverse=True)
        return products[:limit]

    def search(self, request: SearchRequest) -> list[SearchHit]:
        if not self._products:
            return []
        query = np.asarray(request.normalized_vector(), dtype=np.float32)
        candidate_target = max(request.top_k * self.config.local_candidate_multiplier, 32)
        candidate_ids = self._candidate_ids(query, candidate_target)
        filtered_ids = [item_id for item_id in candidate_ids if self._matches_filter(self._products[item_id], request)]

        if len(filtered_ids) < request.top_k:
            for item_id, product in self._products.items():
                if item_id in filtered_ids:
                    continue
                if self._matches_filter(product, request):
                    filtered_ids.append(item_id)

        scores: list[tuple[float, int]] = []
        for item_id in filtered_ids:
            score = float(np.dot(query, self._vectors[item_id]))
            scores.append((score, item_id))
        scores.sort(key=lambda entry: entry[0], reverse=True)

        hits: list[SearchHit] = []
        for score, item_id in scores[: request.top_k]:
            product = self._products[item_id]
            hits.append(
                SearchHit(
                    item_id=product.item_id,
                    category_id=product.category_id,
                    category_name=product.category_name,
                    price=product.price,
                    product_name=product.product_name,
                    brand=product.brand,
                    product_type=product.product_type,
                    size=product.size,
                    purpose=product.purpose,
                    target_group=product.target_group,
                    market=product.market,
                    score=score,
                )
            )
        return hits

    def count(self) -> int:
        return len(self._products)

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "count": self.count(),
            "bucket_count": len([bucket for bucket in self._buckets.values() if bucket]),
            "hash_bits": self.config.local_hash_bits,
            "max_hamming_radius": self.config.local_max_hamming_radius,
        }

    def _hash(self, vector: np.ndarray) -> int:
        assert self._hyperplanes is not None
        projections = self._hyperplanes @ vector
        code = 0
        for index, value in enumerate(projections):
            if value >= 0:
                code |= 1 << index
        return code

    def _candidate_ids(self, query: np.ndarray, target: int) -> list[int]:
        code = self._hash(query)
        candidates: set[int] = set()
        for radius in range(self.config.local_max_hamming_radius + 1):
            for neighbor in self._neighbor_codes(code, radius):
                candidates.update(self._buckets.get(neighbor, set()))
                if len(candidates) >= target:
                    return list(candidates)
        return list(candidates)

    def _neighbor_codes(self, code: int, radius: int) -> Iterable[int]:
        if radius == 0:
            yield code
            return
        bit_range = range(self.config.local_hash_bits)
        for positions in combinations(bit_range, radius):
            neighbor = code
            for position in positions:
                neighbor ^= 1 << position
            yield neighbor

    @staticmethod
    def _matches_filter(product: ProductRecord, request: SearchRequest) -> bool:
        if request.category_ids and product.category_id not in request.category_ids:
            return False
        if request.min_price is not None and product.price < request.min_price:
            return False
        if request.max_price is not None and product.price > request.max_price:
            return False
        return True


def coerce_products(products: Sequence[ProductRecord | dict[str, Any]]) -> list[ProductRecord]:
    coerced: list[ProductRecord] = []
    for product in products:
        if isinstance(product, ProductRecord):
            coerced.append(product)
        else:
            coerced.append(ProductRecord.from_mapping(product))
    return coerced


def exact_search(products: Sequence[ProductRecord], request: SearchRequest) -> list[SearchHit]:
    query = np.asarray(normalize_vector(request.vector), dtype=np.float32)
    hits: list[SearchHit] = []
    for product in products:
        if request.category_ids and product.category_id not in request.category_ids:
            continue
        if request.min_price is not None and product.price < request.min_price:
            continue
        if request.max_price is not None and product.price > request.max_price:
            continue
        score = float(np.dot(query, np.asarray(normalize_vector(product.embedding), dtype=np.float32)))
        hits.append(
            SearchHit(
                item_id=product.item_id,
                category_id=product.category_id,
                category_name=product.category_name,
                price=product.price,
                product_name=product.product_name,
                brand=product.brand,
                product_type=product.product_type,
                size=product.size,
                purpose=product.purpose,
                target_group=product.target_group,
                market=product.market,
                score=score,
            )
        )
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[: request.top_k]
