"""Milvus-based product vector retrieval prototype."""

from .config import SystemConfig
from .models import ProductRecord, SearchHit, SearchRequest
from .service import ProductVectorSystem

__all__ = [
    "ProductRecord",
    "ProductVectorSystem",
    "SearchHit",
    "SearchRequest",
    "SystemConfig",
]
