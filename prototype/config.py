from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class SystemConfig:
    backend: str = "auto"
    milvus_uri: str = "product_demo.db"
    collection_name: str = "product_vectors"
    vector_dim: int = 128
    metric_type: str = "COSINE"
    index_type: str = "AUTOINDEX"
    local_hash_bits: int = 10
    local_max_hamming_radius: int = 3
    local_candidate_multiplier: int = 24
    local_seed: int = 20260328

    @classmethod
    def from_env(cls) -> "SystemConfig":
        return cls(
            backend=os.getenv("MILVA_BACKEND", "auto"),
            milvus_uri=os.getenv("MILVA_MILVUS_URI", "product_demo.db"),
            collection_name=os.getenv("MILVA_COLLECTION", "product_vectors"),
            vector_dim=int(os.getenv("MILVA_VECTOR_DIM", "128")),
            metric_type=os.getenv("MILVA_METRIC_TYPE", "COSINE"),
            index_type=os.getenv("MILVA_INDEX_TYPE", "AUTOINDEX"),
            local_hash_bits=int(os.getenv("MILVA_LOCAL_HASH_BITS", "10")),
            local_max_hamming_radius=int(os.getenv("MILVA_LOCAL_RADIUS", "3")),
            local_candidate_multiplier=int(os.getenv("MILVA_LOCAL_CANDIDATES", "24")),
            local_seed=int(os.getenv("MILVA_LOCAL_SEED", "20260328")),
        )
