from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from prototype import ProductVectorSystem, SearchRequest, SystemConfig
from prototype.http_api import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 Milvus 的海量商品向量实时存储与近似检索原型系统",
    )
    parser.add_argument("--backend", choices=["auto", "milvus", "local_ann"], default=None)
    parser.add_argument("--milvus-uri", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--dim", type=int, default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="查看系统状态")
    subparsers.add_parser("reset", help="重置集合或本地索引")

    bootstrap = subparsers.add_parser("bootstrap", help="生成并导入模拟商品数据")
    bootstrap.add_argument("--count", type=int, default=2000)
    bootstrap.add_argument("--seed", type=int, default=42)
    bootstrap.add_argument("--id-start", type=int, default=1)

    search = subparsers.add_parser("search", help="按商品 ID 执行相似商品检索")
    search.add_argument("--item-id", type=int, required=True)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--category-ids", type=int, nargs="*", default=None)
    search.add_argument("--min-price", type=float, default=None)
    search.add_argument("--max-price", type=float, default=None)

    delete = subparsers.add_parser("delete", help="删除指定商品")
    delete.add_argument("--item-id", type=int, nargs="+", required=True)

    bench = subparsers.add_parser("benchmark", help="运行实验评估")
    bench.add_argument("--sizes", type=int, nargs="+", default=[1000, 3000, 5000])
    bench.add_argument("--queries", type=int, default=20)
    bench.add_argument("--top-k", type=int, default=10)
    bench.add_argument("--seed", type=int, default=42)

    demo = subparsers.add_parser("demo", help="快速演示完整原型流程")
    demo.add_argument("--count", type=int, default=2000)
    demo.add_argument("--top-k", type=int, default=5)
    demo.add_argument("--seed", type=int, default=42)

    serve_parser = subparsers.add_parser("serve", help="启动简易 HTTP 原型接口")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)

    vector = subparsers.add_parser("search-vector", help="直接按向量进行近似检索")
    vector.add_argument("--vector", required=True, help="JSON array, e.g. [0.1, 0.2, ...]")
    vector.add_argument("--top-k", type=int, default=10)
    vector.add_argument("--category-ids", type=int, nargs="*", default=None)
    vector.add_argument("--min-price", type=float, default=None)
    vector.add_argument("--max-price", type=float, default=None)

    return parser


def create_system(args: argparse.Namespace) -> ProductVectorSystem:
    config = SystemConfig.from_env()
    if args.backend is not None:
        config.backend = args.backend
    if args.milvus_uri is not None:
        config.milvus_uri = args.milvus_uri
    if args.collection is not None:
        config.collection_name = args.collection
    if args.dim is not None:
        config.vector_dim = args.dim
    return ProductVectorSystem(config)


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    system = create_system(args)

    if args.command == "status":
        emit(system.status())
        return

    if args.command == "reset":
        emit(system.reset())
        return

    if args.command == "bootstrap":
        emit(system.bootstrap_demo_data(args.count, seed=args.seed, id_start=args.id_start))
        return

    if args.command == "search":
        emit(
            system.search_by_item_id(
                args.item_id,
                top_k=args.top_k,
                category_ids=args.category_ids,
                min_price=args.min_price,
                max_price=args.max_price,
            )
        )
        return

    if args.command == "delete":
        emit(system.delete_products(args.item_id))
        return

    if args.command == "benchmark":
        emit(
            system.benchmark(
                sizes=args.sizes,
                queries_per_size=args.queries,
                top_k=args.top_k,
                seed=args.seed,
            )
        )
        return

    if args.command == "demo":
        system.reset()
        boot = system.bootstrap_demo_data(args.count, seed=args.seed)
        probe_id = boot["sample_item_ids"][0]
        emit(
            {
                "bootstrap": boot,
                "probe_item_id": probe_id,
                "search": system.search_by_item_id(probe_id, top_k=args.top_k),
                "status": system.status(),
            }
        )
        return

    if args.command == "serve":
        serve(system, args.host, args.port)
        return

    if args.command == "search-vector":
        request = SearchRequest(
            vector=json.loads(args.vector),
            top_k=args.top_k,
            category_ids=args.category_ids,
            min_price=args.min_price,
            max_price=args.max_price,
        )
        emit(system.search(request))
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
