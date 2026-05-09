from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .backends import exact_search
from .data_generator import generate_products
from .models import ProductRecord, SearchRequest


@dataclass(slots=True)
class BenchmarkRow:
    dataset_size: int
    insert_ms: float
    avg_search_ms: float
    avg_recall_at_k: float
    avg_mrr_at_k: float
    avg_ndcg_at_k: float
    update_ms: float
    delete_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mutate_products(products: list[ProductRecord], seed: int) -> list[ProductRecord]:
    rng = np.random.default_rng(seed)
    mutated: list[ProductRecord] = []
    for product in products:
        vector = np.asarray(product.embedding, dtype=np.float32)
        noise = rng.normal(loc=0.0, scale=0.03, size=vector.shape[0]).astype(np.float32)
        new_price = round(float(product.price * rng.uniform(0.95, 1.10)), 2)
        mutated.append(
            ProductRecord(
                item_id=product.item_id,
                category_id=product.category_id,
                category_name=product.category_name,
                price=new_price,
                product_name=f"{product.product_name}-更新",
                brand=product.brand,
                product_type=product.product_type,
                size=product.size,
                purpose=product.purpose,
                target_group=product.target_group,
                market=product.market,
                embedding=(vector + noise).tolist(),
            )
        )
    return mutated


def _ranking_metrics(approx_ids: list[int], exact_ids: list[int]) -> tuple[float, float, float]:
    if not exact_ids:
        return 0.0, 0.0, 0.0

    exact_set = set(exact_ids)
    recall = len(set(approx_ids) & exact_set) / max(1, len(exact_ids))

    mrr = 0.0
    for rank, item_id in enumerate(approx_ids, start=1):
        if item_id in exact_set:
            mrr = 1.0 / rank
            break

    relevance_by_id = {
        item_id: len(exact_ids) - index
        for index, item_id in enumerate(exact_ids)
    }

    def dcg(ids: list[int]) -> float:
        score = 0.0
        for index, item_id in enumerate(ids):
            relevance = relevance_by_id.get(item_id, 0)
            if relevance:
                score += relevance / float(np.log2(index + 2))
        return score

    ideal_dcg = dcg(exact_ids)
    ndcg = dcg(approx_ids[: len(exact_ids)]) / ideal_dcg if ideal_dcg else 0.0
    return recall, mrr, ndcg


def run_benchmark(
    system: Any,
    sizes: list[int],
    *,
    queries_per_size: int = 20,
    top_k: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows: list[BenchmarkRow] = []

    for dataset_size in sizes:
        system.reset()
        products = generate_products(dataset_size, system.config.vector_dim, seed=seed + dataset_size)

        started = perf_counter()
        system.upsert_products(products)
        insert_ms = round((perf_counter() - started) * 1000.0, 3)

        sampled_indices = rng.choice(
            len(products),
            size=min(queries_per_size, len(products)),
            replace=False,
        )
        search_latencies: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        for index in sampled_indices:
            probe = products[int(index)]
            request = SearchRequest(vector=probe.embedding, top_k=top_k)
            started = perf_counter()
            response = system.search(request)
            search_latencies.append((perf_counter() - started) * 1000.0)
            approx_ids = [hit["item_id"] for hit in response["hits"]]
            exact_ids = [hit.item_id for hit in exact_search(products, request)]
            recall, mrr, ndcg = _ranking_metrics(approx_ids, exact_ids)
            recalls.append(recall)
            mrrs.append(mrr)
            ndcgs.append(ndcg)

        update_sample = products[: min(100, len(products))]
        mutated = _mutate_products(update_sample, seed + dataset_size + 1)
        started = perf_counter()
        system.upsert_products(mutated)
        update_ms = round((perf_counter() - started) * 1000.0, 3)

        delete_ids = [product.item_id for product in products[-min(100, len(products)) :]]
        started = perf_counter()
        system.delete_products(delete_ids)
        delete_ms = round((perf_counter() - started) * 1000.0, 3)

        rows.append(
            BenchmarkRow(
                dataset_size=dataset_size,
                insert_ms=insert_ms,
                avg_search_ms=round(float(np.mean(search_latencies or [0.0])), 3),
                avg_recall_at_k=round(float(np.mean(recalls or [0.0])), 4),
                avg_mrr_at_k=round(float(np.mean(mrrs or [0.0])), 4),
                avg_ndcg_at_k=round(float(np.mean(ndcgs or [0.0])), 4),
                update_ms=update_ms,
                delete_ms=delete_ms,
            )
        )

    return {
        "backend": system.backend.name,
        "degraded_reason": system.degraded_reason,
        "rows": [row.to_dict() for row in rows],
    }
