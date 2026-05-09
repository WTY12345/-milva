from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


def normalize_vector(values: Iterable[float]) -> list[float]:
    vector = np.asarray(list(values), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Embedding vector must not be zero.")
    return (vector / norm).astype(np.float32).tolist()


@dataclass(slots=True)
class ProductRecord:
    item_id: int
    category_id: int
    category_name: str
    price: float
    product_name: str
    brand: str
    product_type: str
    size: str
    purpose: str
    target_group: str
    market: str
    embedding: list[float]

    def normalized(self) -> "ProductRecord":
        return ProductRecord(
            item_id=self.item_id,
            category_id=self.category_id,
            category_name=self.category_name,
            price=float(self.price),
            product_name=self.product_name,
            brand=self.brand,
            product_type=self.product_type,
            size=self.size,
            purpose=self.purpose,
            target_group=self.target_group,
            market=self.market,
            embedding=normalize_vector(self.embedding),
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self.normalized())

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ProductRecord":
        return cls(
            item_id=int(data["item_id"]),
            category_id=int(data["category_id"]),
            category_name=str(data.get("category_name", "")),
            price=float(data["price"]),
            product_name=str(data["product_name"]),
            brand=str(data.get("brand", "")),
            product_type=str(data.get("product_type", "")),
            size=str(data.get("size", "")),
            purpose=str(data.get("purpose", "")),
            target_group=str(data.get("target_group", "")),
            market=str(data.get("market", "")),
            embedding=[float(value) for value in data["embedding"]],
        )


@dataclass(slots=True)
class SearchRequest:
    vector: list[float]
    top_k: int = 10
    category_ids: list[int] | None = None
    min_price: float | None = None
    max_price: float | None = None

    def normalized_vector(self) -> list[float]:
        return normalize_vector(self.vector)


@dataclass(slots=True)
class SearchHit:
    item_id: int
    category_id: int
    category_name: str
    price: float
    product_name: str
    brand: str
    product_type: str
    size: str
    purpose: str
    target_group: str
    market: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
