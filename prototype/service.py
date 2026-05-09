from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from threading import RLock, Thread
from time import perf_counter, time
from typing import Any, Iterable, Sequence

import numpy as np

from .backends import BackendUnavailableError, LocalAnnVectorBackend, MilvusVectorBackend, coerce_products, exact_search
from .benchmark import run_benchmark
from .config import SystemConfig
from .data_generator import CATEGORY_DEFINITIONS, iter_products
from .models import ProductRecord, SearchRequest, normalize_vector


CATEGORY_HINTS = {
    1: ["手机", "耳机", "平板", "键盘", "鼠标", "音箱", "数码", "电脑", "电子", "智能设备", "科技", "办公设备"],
    2: ["空调", "冰箱", "洗衣机", "净化器", "咖啡机", "电器", "家电", "厨房电器", "厨电"],
    3: ["外套", "跑鞋", "卫衣", "背包", "服饰", "衣服", "衣物", "鞋包", "穿搭", "穿的", "鞋", "包", "裤子", "上衣"],
    4: ["面霜", "精华", "口红", "护肤", "美妆", "防晒", "洗面", "化妆", "彩妆", "护肤品"],
    5: ["食物", "食品", "零食", "生鲜", "牛排", "坚果", "咖啡豆", "礼盒", "乳品", "饮料", "水果", "蔬菜", "肉类", "吃的", "喝的"],
    6: ["桌灯", "收纳", "床品", "餐具", "毛巾", "家居", "日用", "家用品", "生活用品", "用的"],
    7: ["帐篷", "冲锋衣", "运动", "登山", "徒步", "户外", "跑步", "健身", "露营", "骑行", "冒险", "运动装备"],
    8: ["图书", "文具", "手账", "钢笔", "拼图", "文创", "书", "阅读", "教材", "玩的", "玩具", "娱乐", "益智", "消遣", "解闷", "桌游"],
}

PURPOSE_HINTS = {
    "办公": ["通勤", "上班", "办公", "白领", "会议", "出差"],
    "学习": ["学习", "学生", "上课", "考试", "考研", "阅读"],
    "家用": ["家用", "家庭", "居家", "卧室", "客厅", "厨房"],
    "礼赠": ["礼物", "送礼", "礼盒", "礼赠"],
    "运动": ["运动", "健身", "跑步", "训练"],
    "户外": ["户外", "露营", "登山", "徒步", "骑行"],
    "餐饮": ["食物", "食品", "零食", "饮料", "咖啡", "生鲜", "吃的", "喝的"],
    "护理": ["护肤", "美妆", "护理", "补水", "彩妆"],
    "娱乐": ["玩的", "玩具", "娱乐", "消遣", "解闷", "拼图", "桌游", "阅读"],
    "穿搭": ["穿的", "衣服", "衣物", "穿搭", "搭配"],
}

PURPOSE_CATEGORY_MAP = {
    "办公": [1, 6],
    "学习": [8, 1],
    "家用": [6, 2],
    "礼赠": [5, 4, 3],
    "运动": [7, 3],
    "户外": [7],
    "餐饮": [5],
    "护理": [4],
    "娱乐": [8, 7],
    "穿搭": [3],
}

TARGET_GROUP_HINTS = {
    "白领": ["白领", "上班族", "通勤族", "职场"],
    "学生": ["学生", "考研", "上学", "宿舍"],
    "家庭用户": ["家庭", "全家", "父母", "孩子", "一家"],
    "母婴人群": ["母婴", "宝宝", "婴儿", "孕妇"],
    "运动人群": ["运动", "健身", "跑步", "户外"],
    "高端消费群体": ["高端", "旗舰", "奢享", "贵一点"],
}

MARKET_HINTS = {
    "大众市场": ["大众", "平价", "实惠", "便宜", "日常"],
    "中端市场": ["中端", "品质", "主流", "均衡"],
    "高端市场": ["高端", "旗舰", "专业", "高配", "premium"],
}

SIZE_HINTS = {
    "便携款": ["便携", "轻便", "mini", "小巧"],
    "大容量": ["大容量", "家庭装", "多人份", "加量"],
    "专业版": ["专业", "pro", "max", "旗舰", "高配"],
    "标准版": ["标准", "基础", "普通"],
}

CATEGORY_DEFAULT_PURPOSE = {
    1: "办公",
    2: "家用",
    3: "穿搭",
    4: "护理",
    5: "餐饮",
    6: "家用",
    7: "运动",
    8: "娱乐",
}

CATEGORY_DEFAULT_PRICE = {
    1: 1999.0,
    2: 899.0,
    3: 299.0,
    4: 269.0,
    5: 129.0,
    6: 159.0,
    7: 459.0,
    8: 89.0,
}

GENERIC_PRODUCT_TYPE = {
    1: "数码单品",
    2: "家电单品",
    3: "服饰单品",
    4: "护肤单品",
    5: "食品单品",
    6: "家居单品",
    7: "运动装备",
    8: "文创单品",
}

FALLBACK_CATEGORY_ID = 6
MIN_GENERATE_COUNT = 100
MAX_GENERATE_COUNT = 1000000
DEFAULT_GENERATE_BATCH_SIZE = 500
DEFAULT_MARKET_BENCHMARK_SAMPLE = 1000
MAX_MARKET_BENCHMARK_SAMPLE = 10000

EVALUATION_DIMENSIONS = [
    {
        "metric": "数据规模",
        "description": "当前集合中的商品向量条数，用于证明系统能够处理从百级到百万级的商品数据。",
        "method": "通过 /health 与 /metrics 实时读取集合 count；生成任务限制为 100-1000000 条。",
    },
    {
        "metric": "写入耗时 / 写入吞吐",
        "description": "衡量批量商品向量入库能力，适合展示实时存储性能。",
        "method": "记录每次 upsert 的 latency_ms，并换算为 rows/s。",
    },
    {
        "metric": "平均检索延迟",
        "description": "衡量 Top-K 近似检索响应速度，是实时检索体验的核心指标。",
        "method": "每次 search 记录 latency_ms，监控页展示平均值、最近值和最大值。",
    },
    {
        "metric": "Recall@K",
        "description": "衡量近似检索结果与精确检索结果的一致程度，用于说明精度与效率权衡。",
        "method": "benchmark 中用 exact_search 结果作为参照，计算近似结果重合比例。",
    },
    {
        "metric": "MRR@K",
        "description": "衡量第一个相关商品在 Top-K 结果中的排名位置，越接近 1 说明首个有效结果越靠前。",
        "method": "以 exact_search 的 Top-K 结果为相关集合，计算每次查询第一个命中结果的倒数排名并取平均。",
    },
    {
        "metric": "NDCG@K",
        "description": "衡量 Top-K 检索结果整体排序质量，既关注是否命中，也关注高相关商品是否排在前面。",
        "method": "以 exact_search 排名生成分级相关性，计算 DCG 与理想 DCG 的比值并取平均。",
    },
    {
        "metric": "过滤检索能力",
        "description": "商品检索通常需要结合类目、价格等标量过滤，不能只比较向量搜索速度。",
        "method": "通过 category_ids、min_price、max_price 条件验证向量检索与结构化过滤的组合能力。",
    },
    {
        "metric": "动态更新能力",
        "description": "电商商品价格、名称、上下架状态会变化，需要验证新增、删除、更新后的可检索性。",
        "method": "记录 upsert 与 delete 耗时，并在功能测试中验证更新后返回新信息。",
    },
    {
        "metric": "部署与扩展性",
        "description": "对比 Pinecone、Zilliz Cloud、Milvus、Qdrant、Weaviate、Elasticsearch/OpenSearch、pgvector、Redis Vector、Faiss 等方案的工程适配能力。",
        "method": "监控页以表格列出官方能力维度，实际运行行使用本系统实测指标，其他方案标注为待接入同一数据集实测。",
    },
]

DATABASE_COMPARISON_PROFILES = [
    {
        "name": "Milvus",
        "type": "专业向量数据库",
        "vector_support": "原生向量集合与 ANN 索引",
        "filter_support": "支持标量字段过滤",
        "dynamic_update": "支持 upsert/delete",
        "distributed": "强，适合扩展到更大规模",
        "fit": "最适合本课题的商品向量近似检索主方案",
        "data_source": "本系统可实测",
    },
    {
        "name": "Faiss",
        "type": "向量检索算法库",
        "vector_support": "ANN 算法丰富，单机性能强",
        "filter_support": "需要业务层自行组合",
        "dynamic_update": "部分索引更新不够直接",
        "distributed": "需要额外工程封装",
        "fit": "适合作为算法基线，不适合直接承担完整商品库服务",
        "data_source": "能力对比维度",
    },
    {
        "name": "Elasticsearch/OpenSearch",
        "type": "搜索引擎",
        "vector_support": "支持 kNN / dense vector",
        "filter_support": "文本检索与结构化过滤能力强",
        "dynamic_update": "文档更新成熟",
        "distributed": "强，生态完善",
        "fit": "适合文本搜索与向量混合检索场景",
        "data_source": "能力对比维度",
    },
    {
        "name": "PostgreSQL + pgvector",
        "type": "关系数据库扩展",
        "vector_support": "通过扩展支持向量索引",
        "filter_support": "SQL 过滤能力强",
        "dynamic_update": "事务与更新能力成熟",
        "distributed": "中，扩展性依赖数据库架构",
        "fit": "适合中小规模业务与事务一致性要求较高的场景",
        "data_source": "能力对比维度",
    },
    {
        "name": "Redis Vector",
        "type": "内存数据库 / 搜索模块",
        "vector_support": "支持向量相似检索",
        "filter_support": "支持部分属性过滤",
        "dynamic_update": "实时写入能力强",
        "distributed": "中到强，依赖部署方式",
        "fit": "适合低延迟在线召回与缓存型场景",
        "data_source": "能力对比维度",
    },
    {
        "name": "MySQL 顺序扫描",
        "type": "传统关系数据库基线",
        "vector_support": "非原生向量检索",
        "filter_support": "结构化过滤强",
        "dynamic_update": "事务更新成熟",
        "distributed": "中",
        "fit": "可作为低性能精确扫描基线，不适合海量高维向量实时检索",
        "data_source": "理论基线",
    },
]


MARKET_DATABASE_COMPARISON_PROFILES = [
    {
        "name": "Pinecone",
        "type": "托管向量数据库",
        "data_source": "官方文档：metadata filter、upsert、hybrid search",
        "source_url": "https://docs.pinecone.io/guides/search/filter-by-metadata",
        "vector_support": "云端托管索引，支持 dense/sparse 向量与文本集成嵌入",
        "filter_support": "支持元数据过滤，适合商品类目、价格、品牌等字段筛选",
        "hybrid_search": "支持稠密向量、稀疏向量与全文字段组合",
        "dynamic_update": "支持 upsert、update、delete；千万级大批量导入建议使用 import",
        "distributed": "托管服务，自动扩展与多租户隔离能力较强",
        "fit": "适合云上商品语义检索、推荐和多租户商品库；成本依赖托管服务",
    },
    {
        "name": "Zilliz Cloud",
        "type": "Milvus 托管向量数据库",
        "data_source": "官方文档：ANN search、filtering、upsert",
        "source_url": "https://docs.zilliz.com/docs/single-vector-search",
        "vector_support": "基于 Milvus 的托管 ANN 向量检索，支持 range/grouping 等搜索能力",
        "filter_support": "支持标量过滤表达式，可将商品属性过滤与向量搜索结合",
        "hybrid_search": "支持向量相似搜索，并提供关键词/文本匹配能力作为补充",
        "dynamic_update": "支持 upsert、delete，适合商品新增、改价和下架同步",
        "distributed": "托管集群，适合生产级扩展和运维托管",
        "fit": "与本系统技术路线最接近，可作为 Milvus 生产化方案对照",
    },
    {
        "name": "Milvus",
        "type": "开源向量数据库",
        "data_source": "官方文档：filtered search",
        "source_url": "https://milvus.io/docs/filtered-search.md",
        "vector_support": "原生集合、向量字段和 ANN 索引，适合大规模向量召回",
        "filter_support": "支持标准过滤和迭代过滤，可先按元数据缩小搜索范围",
        "hybrid_search": "以向量检索为核心，可结合标量字段与上层重排实现商品检索",
        "dynamic_update": "支持 insert/upsert/delete，分布式部署能力较强",
        "distributed": "开源可自建，适合毕业设计和生产验证之间平滑迁移",
        "fit": "适合本课题的海量商品向量实时存储与近似检索主方案",
    },
    {
        "name": "Qdrant",
        "type": "开源/托管向量数据库",
        "data_source": "官方文档：payload、filtering、upsert points",
        "source_url": "https://qdrant.tech/documentation/manage-data/payload/",
        "vector_support": "以 point 存储向量和 payload，支持向量相似搜索",
        "filter_support": "payload 支持 JSON 元数据和过滤条件，适合商品属性过滤",
        "hybrid_search": "支持 payload 过滤、全文能力和向量搜索组合，适合语义 + 条件检索",
        "dynamic_update": "支持 upsert points 与 payload 更新，已有 ID 可覆盖",
        "distributed": "支持自建和云服务，工程集成相对轻量",
        "fit": "适合需要轻量部署、过滤条件丰富的商品向量检索系统",
    },
    {
        "name": "Weaviate",
        "type": "开源/托管 AI 数据库",
        "data_source": "官方文档：hybrid search、filters、object update",
        "source_url": "https://docs.weaviate.io/weaviate/search/hybrid",
        "vector_support": "支持向量相似搜索、对象属性和多种向量化方式",
        "filter_support": "过滤器可与 nearText、hybrid、BM25 等搜索算子组合",
        "hybrid_search": "原生支持向量检索与 BM25 关键词检索融合",
        "dynamic_update": "支持对象完整或局部更新，属性变化可重新向量化和索引",
        "distributed": "支持自建与 Weaviate Cloud，生态偏 AI 应用",
        "fit": "适合商品语义搜索、关键词召回与属性过滤混合的应用型场景",
    },
    {
        "name": "Elasticsearch / OpenSearch",
        "type": "搜索引擎 + 向量检索",
        "data_source": "官方文档：kNN vector search、filtering",
        "source_url": "https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html",
        "vector_support": "支持 dense_vector/kNN，可与传统倒排索引共存",
        "filter_support": "结构化过滤、全文检索、聚合和排序能力成熟",
        "hybrid_search": "适合关键词检索、向量检索和业务字段排序的混合检索",
        "dynamic_update": "文档增删改成熟，但向量索引写入和刷新成本需评估",
        "distributed": "成熟分布式搜索引擎生态，运维和资源开销较高",
        "fit": "适合已有搜索体系升级商品语义检索，不是最轻量的纯向量库方案",
    },
    {
        "name": "PostgreSQL + pgvector",
        "type": "关系数据库扩展",
        "data_source": "官方仓库：pgvector",
        "source_url": "https://github.com/pgvector/pgvector",
        "vector_support": "在 PostgreSQL 中存储向量，支持精确检索和 HNSW/IVFFlat 索引",
        "filter_support": "SQL 过滤、事务和表关联能力强，适合业务数据同库存放",
        "hybrid_search": "可与 SQL 条件、全文索引和业务排序组合",
        "dynamic_update": "继承 PostgreSQL 事务和更新能力，数据一致性好",
        "distributed": "扩展性受 PostgreSQL 架构影响，中小规模更合适",
        "fit": "适合商品量中等、强事务和业务表关联要求高的场景",
    },
    {
        "name": "Redis Vector",
        "type": "内存数据库 / 搜索模块",
        "data_source": "官方文档：vector search concepts",
        "source_url": "https://redis.io/docs/latest/develop/ai/search-and-query/vectors/",
        "vector_support": "支持 FLAT、HNSW、SVS-VAMANA 等向量索引类型",
        "filter_support": "支持文本、数值、地理、标签等元数据过滤",
        "hybrid_search": "支持 KNN 查询与主过滤条件组合，低延迟特征明显",
        "dynamic_update": "通过 HASH/JSON 存储和更新向量及元数据",
        "distributed": "适合缓存、在线召回和低延迟链路，内存成本需关注",
        "fit": "适合作为商品推荐召回缓存层或热点商品语义检索层",
    },
    {
        "name": "Faiss",
        "type": "向量检索算法库",
        "data_source": "官方仓库：facebookresearch/faiss",
        "source_url": "https://github.com/facebookresearch/faiss",
        "vector_support": "提供多种精确和近似向量索引，单机/GPU 检索能力强",
        "filter_support": "不提供完整数据库过滤、权限、持久化和服务接口，需要业务层封装",
        "hybrid_search": "通常作为向量召回算法内核，混合检索需外部系统组合",
        "dynamic_update": "部分索引支持新增/删除，但生产级更新链路需自行设计",
        "distributed": "算法库而非数据库，分布式、持久化和服务治理需额外工程",
        "fit": "适合作为论文算法基线，不适合直接承担完整商品检索服务",
    },
]


def _dedupe(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


class ProductVectorSystem:
    def __init__(self, config: SystemConfig | None = None) -> None:
        self.config = config or SystemConfig.from_env()
        self.degraded_reason: str | None = None
        self._direction_cache: dict[tuple[str, str], np.ndarray] = {}
        self._metrics_lock = RLock()
        self._started_at = time()
        self._operations: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._job_sequence = 0
        self._last_benchmark: dict[str, Any] | None = None
        self._last_market_benchmark: dict[str, Any] | None = None
        self.backend = self._initialize_backend()

    def _validate_generate_count(self, count: int) -> int:
        count = int(count)
        if count < MIN_GENERATE_COUNT or count > MAX_GENERATE_COUNT:
            raise ValueError(f"生成数量必须在 {MIN_GENERATE_COUNT}-{MAX_GENERATE_COUNT} 之间。")
        return count

    def _validate_benchmark_sizes(self, sizes: Sequence[int]) -> list[int]:
        normalized = [self._validate_generate_count(int(size)) for size in sizes]
        if not normalized:
            raise ValueError("评测数据规模不能为空。")
        return normalized

    def _validate_market_benchmark_sample(self, sample_size: int) -> int:
        sample_size = int(sample_size)
        if sample_size < MIN_GENERATE_COUNT or sample_size > MAX_MARKET_BENCHMARK_SAMPLE:
            raise ValueError(f"同数据集实测样本量必须在 {MIN_GENERATE_COUNT}-{MAX_MARKET_BENCHMARK_SAMPLE} 之间。")
        return sample_size

    def _validate_total_after_generation(self, count: int, *, reset_first: bool = False) -> None:
        if reset_first:
            return
        current = int(self.backend.count())
        if current + count > MAX_GENERATE_COUNT:
            raise ValueError(
                f"商品总量上限为 {MAX_GENERATE_COUNT}。当前已有 {current} 条，"
                f"继续生成 {count} 条会超过上限；请先清空集合或减少生成数量。"
            )

    def _record_operation(self, name: str, latency_ms: float, **details: Any) -> None:
        with self._metrics_lock:
            stats = self._operations.setdefault(
                name,
                {
                    "count": 0,
                    "total_ms": 0.0,
                    "avg_ms": 0.0,
                    "last_ms": 0.0,
                    "max_ms": 0.0,
                    "last_at": None,
                    "last": {},
                },
            )
            latency_ms = round(float(latency_ms), 3)
            stats["count"] += 1
            stats["total_ms"] = round(float(stats["total_ms"]) + latency_ms, 3)
            stats["avg_ms"] = round(float(stats["total_ms"]) / max(1, int(stats["count"])), 3)
            stats["last_ms"] = latency_ms
            stats["max_ms"] = round(max(float(stats["max_ms"]), latency_ms), 3)
            stats["last_at"] = round(time(), 3)
            stats["last"] = details

    def _next_job_id(self, kind: str) -> str:
        with self._metrics_lock:
            self._job_sequence += 1
            return f"{kind}-{self._job_sequence}"

    def _create_job(self, kind: str, *, total: int | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
        job_id = self._next_job_id(kind)
        job = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "total": total,
            "processed": 0,
            "progress": 0.0,
            "message": "等待执行",
            "params": params or {},
            "created_at": round(time(), 3),
            "updated_at": round(time(), 3),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        with self._metrics_lock:
            self._jobs[job_id] = job
        return deepcopy(job)

    def _update_job(self, job_id: str, **updates: Any) -> None:
        with self._metrics_lock:
            job = self._jobs[job_id]
            job.update(updates)
            total = job.get("total")
            processed = job.get("processed")
            if total:
                job["progress"] = round(min(1.0, float(processed or 0) / float(total)), 4)
            job["updated_at"] = round(time(), 3)

    def job_status(self, job_id: str) -> dict[str, Any]:
        with self._metrics_lock:
            if job_id not in self._jobs:
                raise KeyError(f"Job {job_id} does not exist.")
            return deepcopy(self._jobs[job_id])

    def _initialize_backend(self):
        if self.config.backend == "milvus":
            backend = MilvusVectorBackend(self.config)
            backend.initialize()
            return backend
        if self.config.backend == "local_ann":
            backend = LocalAnnVectorBackend(self.config)
            backend.initialize()
            return backend
        try:
            backend = MilvusVectorBackend(self.config)
            backend.initialize()
            return backend
        except BackendUnavailableError as exc:
            self.degraded_reason = str(exc)
            fallback = LocalAnnVectorBackend(self.config)
            fallback.initialize()
            return fallback

    def reset(self) -> dict[str, Any]:
        started = perf_counter()
        self.backend.reset()
        latency_ms = round((perf_counter() - started) * 1000.0, 3)
        self._record_operation("reset", latency_ms)
        return {"backend": self.backend.name, "count": self.backend.count(), "latency_ms": latency_ms}

    def status(self) -> dict[str, Any]:
        payload = self.backend.status()
        payload["degraded_reason"] = self.degraded_reason
        payload["configured_backend"] = self.config.backend
        payload["milvus_uri"] = self.config.milvus_uri
        payload["vector_dim"] = self.config.vector_dim
        return payload

    def upsert_products(self, products: Sequence[ProductRecord | dict[str, Any]]) -> dict[str, Any]:
        result = self.backend.upsert(coerce_products(products))
        self._record_operation(
            "upsert",
            float(result.get("latency_ms", 0.0)),
            rows=int(result.get("count", len(products))),
            backend=result.get("backend", self.backend.name),
        )
        return result

    def bootstrap_demo_data(self, count: int, *, seed: int = 42, id_start: int = 1) -> dict[str, Any]:
        count = self._validate_generate_count(count)
        self._validate_total_after_generation(count, reset_first=False)
        started = perf_counter()
        products: list[ProductRecord] = []
        total_upsert_ms = 0.0
        sample_item_ids: list[int] = []
        for product in iter_products(count=count, dim=self.config.vector_dim, seed=seed, id_start=id_start):
            payload = product.to_payload()
            payload["embedding"] = self.build_embedding(payload)
            products.append(ProductRecord.from_mapping(payload))
            if len(sample_item_ids) < 5:
                sample_item_ids.append(product.item_id)
            if len(products) >= DEFAULT_GENERATE_BATCH_SIZE:
                result = self.upsert_products(products)
                total_upsert_ms += float(result.get("latency_ms", 0.0))
                products = []
        if products:
            result = self.upsert_products(products)
            total_upsert_ms += float(result.get("latency_ms", 0.0))
        latency_ms = round((perf_counter() - started) * 1000.0, 3)
        self._record_operation("bootstrap", latency_ms, rows=count, seed=seed, id_start=id_start)
        return {
            "backend": self.backend.name,
            "count": count,
            "latency_ms": latency_ms,
            "upsert_ms": round(total_upsert_ms, 3),
            "rows_per_second": round(count / max(latency_ms / 1000.0, 0.001), 2),
            "sample_item_ids": sample_item_ids,
        }

    def start_bootstrap_job(
        self,
        count: int,
        *,
        seed: int = 42,
        id_start: int = 1,
        reset_first: bool = True,
        batch_size: int = DEFAULT_GENERATE_BATCH_SIZE,
    ) -> dict[str, Any]:
        count = self._validate_generate_count(count)
        batch_size = max(50, min(int(batch_size), 1000))
        self._validate_total_after_generation(count, reset_first=reset_first)
        job = self._create_job(
            "generate",
            total=count,
            params={"count": count, "seed": seed, "id_start": id_start, "reset_first": reset_first, "batch_size": batch_size},
        )
        thread = Thread(
            target=self._run_bootstrap_job,
            args=(job["id"], count, seed, id_start, reset_first, batch_size),
            daemon=True,
        )
        thread.start()
        return self.job_status(job["id"])

    def _run_bootstrap_job(
        self,
        job_id: str,
        count: int,
        seed: int,
        id_start: int,
        reset_first: bool,
        batch_size: int,
    ) -> None:
        started = perf_counter()
        try:
            self._update_job(job_id, status="running", started_at=round(time(), 3), message="正在准备商品向量数据")
            if reset_first:
                self.reset()
            batch: list[ProductRecord] = []
            inserted = 0
            sample_item_ids: list[int] = []
            total_upsert_ms = 0.0

            for product in iter_products(count=count, dim=self.config.vector_dim, seed=seed, id_start=id_start):
                payload = product.to_payload()
                payload["embedding"] = self.build_embedding(payload)
                normalized = ProductRecord.from_mapping(payload)
                batch.append(normalized)
                if len(sample_item_ids) < 5:
                    sample_item_ids.append(normalized.item_id)
                if len(batch) >= batch_size:
                    result = self.upsert_products(batch)
                    inserted += len(batch)
                    total_upsert_ms += float(result.get("latency_ms", 0.0))
                    batch = []
                    self._update_job(job_id, processed=inserted, message=f"已写入 {inserted}/{count} 条商品向量")

            if batch:
                result = self.upsert_products(batch)
                inserted += len(batch)
                total_upsert_ms += float(result.get("latency_ms", 0.0))
                self._update_job(job_id, processed=inserted, message=f"已写入 {inserted}/{count} 条商品向量")

            latency_ms = round((perf_counter() - started) * 1000.0, 3)
            result = {
                "backend": self.backend.name,
                "count": count,
                "latency_ms": latency_ms,
                "upsert_ms": round(total_upsert_ms, 3),
                "rows_per_second": round(count / max(latency_ms / 1000.0, 0.001), 2),
                "sample_item_ids": sample_item_ids,
                "final_count": self.backend.count(),
            }
            self._record_operation("bootstrap", latency_ms, rows=count, seed=seed, id_start=id_start)
            self._update_job(
                job_id,
                status="completed",
                processed=count,
                finished_at=round(time(), 3),
                message=f"生成完成，共写入 {count} 条商品向量",
                result=result,
            )
        except Exception as exc:
            self._update_job(
                job_id,
                status="failed",
                finished_at=round(time(), 3),
                message="生成任务失败",
                error=str(exc),
            )

    def start_benchmark_job(
        self,
        sizes: Sequence[int],
        *,
        queries_per_size: int = 20,
        top_k: int = 10,
        seed: int = 42,
    ) -> dict[str, Any]:
        sizes = self._validate_benchmark_sizes(sizes)
        queries_per_size = max(1, min(int(queries_per_size), 100))
        top_k = max(1, min(int(top_k), 50))
        job = self._create_job(
            "benchmark",
            total=len(sizes),
            params={"sizes": sizes, "queries_per_size": queries_per_size, "top_k": top_k, "seed": seed},
        )
        thread = Thread(
            target=self._run_benchmark_job,
            args=(job["id"], sizes, queries_per_size, top_k, seed),
            daemon=True,
        )
        thread.start()
        return self.job_status(job["id"])

    def _run_benchmark_job(self, job_id: str, sizes: list[int], queries_per_size: int, top_k: int, seed: int) -> None:
        started = perf_counter()
        rows: list[dict[str, Any]] = []
        try:
            self._update_job(job_id, status="running", started_at=round(time(), 3), message="正在运行评测")
            for index, size in enumerate(sizes, 1):
                self._update_job(job_id, processed=index - 1, message=f"正在评测 {size} 条商品向量")
                response = run_benchmark(self, sizes=[size], queries_per_size=queries_per_size, top_k=top_k, seed=seed + index)
                rows.extend(response.get("rows", []))
                self._update_job(job_id, processed=index, message=f"已完成 {size} 条商品向量评测")
            latency_ms = round((perf_counter() - started) * 1000.0, 3)
            result = {"backend": self.backend.name, "degraded_reason": self.degraded_reason, "rows": rows, "latency_ms": latency_ms}
            with self._metrics_lock:
                self._last_benchmark = deepcopy(result)
            self._record_operation("benchmark", latency_ms, sizes=sizes, rows=len(rows))
            self._update_job(
                job_id,
                status="completed",
                finished_at=round(time(), 3),
                message="评测完成",
                result=result,
            )
        except Exception as exc:
            self._update_job(
                job_id,
                status="failed",
                finished_at=round(time(), 3),
                message="评测任务失败",
                error=str(exc),
            )

    def start_market_benchmark_job(
        self,
        *,
        sample_size: int = DEFAULT_MARKET_BENCHMARK_SAMPLE,
        queries_per_size: int = 20,
        top_k: int = 10,
        seed: int = 42,
    ) -> dict[str, Any]:
        sample_size = self._validate_market_benchmark_sample(sample_size)
        queries_per_size = max(1, min(int(queries_per_size), 100))
        top_k = max(1, min(int(top_k), 50))
        job = self._create_job(
            "market_benchmark",
            total=3,
            params={"sample_size": sample_size, "queries_per_size": queries_per_size, "top_k": top_k, "seed": seed},
        )
        thread = Thread(
            target=self._run_market_benchmark_job,
            args=(job["id"], sample_size, queries_per_size, top_k, seed),
            daemon=True,
        )
        thread.start()
        return self.job_status(job["id"])

    def _run_market_benchmark_job(
        self,
        job_id: str,
        sample_size: int,
        queries_per_size: int,
        top_k: int,
        seed: int,
    ) -> None:
        started = perf_counter()
        try:
            self._update_job(job_id, status="running", started_at=round(time(), 3), message="正在读取当前商品数据集")
            total_count = int(self.backend.count())
            if total_count < MIN_GENERATE_COUNT:
                raise ValueError(f"当前商品数据量为 {total_count}，请先生成至少 {MIN_GENERATE_COUNT} 条商品数据后再实测。")

            effective_sample = min(sample_size, total_count)
            products = self.backend.list_products(limit=effective_sample)
            if not products:
                raise ValueError("未能从当前集合读取商品数据，请先生成数据。")
            products = sorted(products, key=lambda product: product.item_id)

            self._update_job(
                job_id,
                processed=0,
                message=f"已读取 {len(products)} 条商品快照，正在生成统一查询集",
            )
            result = self._run_same_dataset_market_benchmark(
                products,
                total_count=total_count,
                queries_per_size=queries_per_size,
                top_k=top_k,
                seed=seed,
                job_id=job_id,
            )
            latency_ms = round((perf_counter() - started) * 1000.0, 3)
            result["latency_ms"] = latency_ms
            with self._metrics_lock:
                self._last_market_benchmark = deepcopy(result)
            self._record_operation(
                "market_benchmark",
                latency_ms,
                sample_size=len(products),
                total_count=total_count,
                rows=len(result.get("rows", [])),
            )
            self._update_job(
                job_id,
                status="completed",
                processed=3,
                finished_at=round(time(), 3),
                message="同数据集实测完成",
                result=result,
            )
        except Exception as exc:
            self._update_job(
                job_id,
                status="failed",
                finished_at=round(time(), 3),
                message="同数据集实测失败",
                error=str(exc),
            )

    def _run_same_dataset_market_benchmark(
        self,
        products: list[ProductRecord],
        *,
        total_count: int,
        queries_per_size: int,
        top_k: int,
        seed: int,
        job_id: str,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        sampled_indices = rng.choice(len(products), size=min(queries_per_size, len(products)), replace=False)
        requests = [SearchRequest(vector=products[int(index)].embedding, top_k=top_k) for index in sampled_indices]
        exact_ids_by_query = [[hit.item_id for hit in exact_search(products, request)] for request in requests]
        rows: list[dict[str, Any]] = []

        self._update_job(job_id, processed=0, message="正在测试当前系统搜索性能")
        rows.append(self._benchmark_current_backend_on_snapshot(products, total_count, requests, exact_ids_by_query))

        self._update_job(job_id, processed=1, message="正在将数据快照接入 Local ANN 基线")
        rows.append(self._benchmark_local_ann_snapshot(products, requests, exact_ids_by_query, seed=seed))

        self._update_job(job_id, processed=2, message="正在测试精确顺序扫描基线")
        rows.append(self._benchmark_exact_scan_snapshot(products, requests, exact_ids_by_query, seed=seed))

        connected = [row["name"] for row in rows]
        return {
            "backend": self.backend.name,
            "total_count": total_count,
            "sample_size": len(products),
            "queries_per_size": len(requests),
            "top_k": top_k,
            "seed": seed,
            "rows": rows,
            "external_note": "Pinecone、Zilliz Cloud、Qdrant、Weaviate、Elasticsearch/OpenSearch、pgvector、Redis Vector 等外部系统需要配置连接信息后才能导入同一数据集实测。",
            "connected": connected,
        }

    def _benchmark_current_backend_on_snapshot(
        self,
        products: list[ProductRecord],
        total_count: int,
        requests: list[SearchRequest],
        exact_ids_by_query: list[list[int]],
    ) -> dict[str, Any]:
        search_latencies: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        for request, exact_ids in zip(requests, exact_ids_by_query):
            started = perf_counter()
            response = self.search(request)
            search_latencies.append((perf_counter() - started) * 1000.0)
            if total_count == len(products):
                approx_ids = [int(hit["item_id"]) for hit in response.get("hits", [])]
                recall, mrr, ndcg = self._ranking_metrics(approx_ids, exact_ids)
                recalls.append(recall)
                mrrs.append(mrr)
                ndcgs.append(ndcg)

        recall: float | str = round(float(np.mean(recalls)), 4) if recalls else "快照外未计算"
        mrr_value: float | str = round(float(np.mean(mrrs)), 4) if mrrs else "快照外未计算"
        ndcg_value: float | str = round(float(np.mean(ndcgs)), 4) if ndcgs else "快照外未计算"
        row = {
            "name": "当前系统",
            "type": "运行实测",
            "data_source": "当前已生成数据集",
            "source_url": "",
            "dataset_count": total_count,
            "sample_size": len(products),
            "avg_search_ms": self._mean_ms(search_latencies),
            "avg_insert_ms": "-",
            "update_ms": "-",
            "delete_ms": "-",
            "recall_at_k": recall,
            "mrr_at_k": mrr_value,
            "ndcg_at_k": ndcg_value,
            "vector_support": "当前后端实际搜索",
            "filter_support": "类目 + 价格过滤",
            "hybrid_search": "向量召回 + 属性过滤 + 业务重排",
            "dynamic_update": "新增 / 删除 / upsert",
            "distributed": "Milvus 模式支持扩展，本地 ANN 用于演示",
            "fit": "本课题实测对象",
        }
        row["measured_summary"] = self._format_market_benchmark_summary(row)
        return row

    def _benchmark_local_ann_snapshot(
        self,
        products: list[ProductRecord],
        requests: list[SearchRequest],
        exact_ids_by_query: list[list[int]],
        *,
        seed: int,
    ) -> dict[str, Any]:
        backend = LocalAnnVectorBackend(self.config)
        backend.initialize()
        insert_result = backend.upsert(products)

        search_latencies: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        for request, exact_ids in zip(requests, exact_ids_by_query):
            started = perf_counter()
            hits = backend.search(request)
            search_latencies.append((perf_counter() - started) * 1000.0)
            recall, mrr, ndcg = self._ranking_metrics([hit.item_id for hit in hits], exact_ids)
            recalls.append(recall)
            mrrs.append(mrr)
            ndcgs.append(ndcg)

        update_sample = self._mutate_market_products(products[: min(100, len(products))], seed + 1)
        update_result = backend.upsert(update_sample)
        delete_ids = [product.item_id for product in products[-min(100, len(products)) :]]
        started = perf_counter()
        backend.delete(delete_ids)
        delete_ms = round((perf_counter() - started) * 1000.0, 3)

        row = {
            "name": "Local ANN 基线",
            "type": "本地近似向量索引",
            "data_source": "同一商品数据快照实测",
            "source_url": "",
            "dataset_count": len(products),
            "sample_size": len(products),
            "avg_search_ms": self._mean_ms(search_latencies),
            "avg_insert_ms": insert_result.get("latency_ms", "-"),
            "update_ms": update_result.get("latency_ms", "-"),
            "delete_ms": delete_ms,
            "recall_at_k": round(float(np.mean(recalls or [0.0])), 4),
            "mrr_at_k": round(float(np.mean(mrrs or [0.0])), 4),
            "ndcg_at_k": round(float(np.mean(ndcgs or [0.0])), 4),
            "vector_support": "LSH 候选召回 + 向量重排",
            "filter_support": "支持类目和价格过滤",
            "hybrid_search": "可作为本地近似检索对照组",
            "dynamic_update": "支持 upsert/delete",
            "distributed": "单机内存基线",
            "fit": "用于验证同一数据集下近似索引与 Milvus 的差异",
        }
        row["measured_summary"] = self._format_market_benchmark_summary(row)
        return row

    def _benchmark_exact_scan_snapshot(
        self,
        products: list[ProductRecord],
        requests: list[SearchRequest],
        exact_ids_by_query: list[list[int]],
        *,
        seed: int,
    ) -> dict[str, Any]:
        started = perf_counter()
        records = [product.normalized() for product in products]
        insert_ms = round((perf_counter() - started) * 1000.0, 3)

        search_latencies: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        for request, exact_ids in zip(requests, exact_ids_by_query):
            started = perf_counter()
            hits = exact_search(records, request)
            search_latencies.append((perf_counter() - started) * 1000.0)
            recall, mrr, ndcg = self._ranking_metrics([hit.item_id for hit in hits], exact_ids)
            recalls.append(recall)
            mrrs.append(mrr)
            ndcgs.append(ndcg)

        update_sample = self._mutate_market_products(records[: min(100, len(records))], seed + 2)
        record_map = {product.item_id: product for product in records}
        started = perf_counter()
        for product in update_sample:
            record_map[product.item_id] = product
        update_ms = round((perf_counter() - started) * 1000.0, 3)
        delete_ids = [product.item_id for product in records[-min(100, len(records)) :]]
        started = perf_counter()
        for item_id in delete_ids:
            record_map.pop(item_id, None)
        delete_ms = round((perf_counter() - started) * 1000.0, 3)

        row = {
            "name": "精确顺序扫描基线",
            "type": "精确检索基线",
            "data_source": "同一商品数据快照实测",
            "source_url": "",
            "dataset_count": len(products),
            "sample_size": len(products),
            "avg_search_ms": self._mean_ms(search_latencies),
            "avg_insert_ms": insert_ms,
            "update_ms": update_ms,
            "delete_ms": delete_ms,
            "recall_at_k": round(float(np.mean(recalls or [0.0])), 4),
            "mrr_at_k": round(float(np.mean(mrrs or [0.0])), 4),
            "ndcg_at_k": round(float(np.mean(ndcgs or [0.0])), 4),
            "vector_support": "全量向量逐条计算相似度",
            "filter_support": "可先做结构化过滤再扫描",
            "hybrid_search": "作为精确召回参考，不是生产级高性能方案",
            "dynamic_update": "数据结构更新直接",
            "distributed": "单机基线",
            "fit": "用于说明近似索引相比全量扫描的效率优势",
        }
        row["measured_summary"] = self._format_market_benchmark_summary(row)
        return row

    @staticmethod
    def _recall_at_k(approx_ids: Sequence[int], exact_ids: Sequence[int]) -> float:
        if not exact_ids:
            return 0.0
        return len(set(approx_ids) & set(exact_ids)) / max(1, len(exact_ids))

    @classmethod
    def _ranking_metrics(cls, approx_ids: Sequence[int], exact_ids: Sequence[int]) -> tuple[float, float, float]:
        if not exact_ids:
            return 0.0, 0.0, 0.0

        approx_list = [int(item_id) for item_id in approx_ids]
        exact_list = [int(item_id) for item_id in exact_ids]
        exact_set = set(exact_list)
        recall = cls._recall_at_k(approx_list, exact_list)

        mrr = 0.0
        for rank, item_id in enumerate(approx_list, start=1):
            if item_id in exact_set:
                mrr = 1.0 / rank
                break

        relevance_by_id = {
            item_id: len(exact_list) - index
            for index, item_id in enumerate(exact_list)
        }

        def dcg(ids: Sequence[int]) -> float:
            score = 0.0
            for index, item_id in enumerate(ids):
                relevance = relevance_by_id.get(int(item_id), 0)
                if relevance:
                    score += relevance / float(np.log2(index + 2))
            return score

        ideal_dcg = dcg(exact_list)
        ndcg = dcg(approx_list[: len(exact_list)]) / ideal_dcg if ideal_dcg else 0.0
        return recall, mrr, ndcg

    @staticmethod
    def _mean_ms(values: Sequence[float]) -> float:
        return round(float(np.mean(values or [0.0])), 3)

    @staticmethod
    def _mutate_market_products(products: Sequence[ProductRecord], seed: int) -> list[ProductRecord]:
        rng = np.random.default_rng(seed)
        mutated: list[ProductRecord] = []
        for product in products:
            vector = np.asarray(product.embedding, dtype=np.float32)
            noise = rng.normal(loc=0.0, scale=0.02, size=vector.shape[0]).astype(np.float32)
            mutated.append(
                ProductRecord(
                    item_id=product.item_id,
                    category_id=product.category_id,
                    category_name=product.category_name,
                    price=round(float(product.price) * 1.01, 2),
                    product_name=f"{product.product_name}-实测更新",
                    brand=product.brand,
                    product_type=product.product_type,
                    size=product.size,
                    purpose=product.purpose,
                    target_group=product.target_group,
                    market=product.market,
                    embedding=normalize_vector((vector + noise).tolist()),
                )
            )
        return mutated

    @staticmethod
    def _format_market_benchmark_summary(row: dict[str, Any]) -> str:
        return (
            f"数据量 {row.get('dataset_count', '-')}；样本 {row.get('sample_size', '-')}；"
            f"平均检索 {row.get('avg_search_ms', '-')} ms；平均写入 {row.get('avg_insert_ms', '-')} ms；"
            f"Recall@K {row.get('recall_at_k', '-')}；MRR@K {row.get('mrr_at_k', '-')}；"
            f"NDCG@K {row.get('ndcg_at_k', '-')}"
        )

    def get_product(self, item_id: int) -> dict[str, Any] | None:
        product = self.backend.get(item_id)
        return None if product is None else product.to_payload()

    def list_products(self, limit: int = 50) -> dict[str, Any]:
        products = self.backend.list_products(limit=limit)
        return {"backend": self.backend.name, "count": len(products), "items": [product.to_payload() for product in products]}

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", "", text.strip().lower())

    def _extract_best_match(self, text: str, candidates: Sequence[str]) -> str | None:
        matches = [candidate for candidate in candidates if candidate and candidate.lower() in text]
        if not matches:
            return None
        return max(matches, key=len)

    def _feature_direction(self, namespace: str, value: str) -> np.ndarray:
        key = (namespace, value)
        cached = self._direction_cache.get(key)
        if cached is None:
            seed_text = f"{namespace}|{value}|{self.config.vector_dim}"
            seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            vector = rng.normal(0.0, 1.0, size=self.config.vector_dim).astype(np.float32)
            cached = np.asarray(normalize_vector(vector.tolist()), dtype=np.float32)
            self._direction_cache[key] = cached
        return cached

    def _derive_purpose(self, text: str) -> str | None:
        normalized = self._normalize_text(text)
        for purpose, hints in PURPOSE_HINTS.items():
            if any(hint.lower() in normalized for hint in hints):
                return purpose
        return None

    def _derive_target_group(self, text: str, purpose: str | None = None) -> str | None:
        normalized = self._normalize_text(text)
        for group, hints in TARGET_GROUP_HINTS.items():
            if any(hint.lower() in normalized for hint in hints):
                return group
        if purpose == "办公":
            return "白领"
        if purpose == "学习":
            return "学生"
        if purpose in {"运动", "户外"}:
            return "运动人群"
        return None

    def _derive_market(self, text: str, price: float | None = None) -> str | None:
        normalized = self._normalize_text(text)
        for market, hints in MARKET_HINTS.items():
            if any(hint.lower() in normalized for hint in hints):
                return market
        if price is None:
            return None
        if price >= 1500:
            return "高端市场"
        if price >= 500:
            return "中端市场"
        return "大众市场"

    def _derive_size(self, text: str, price: float | None = None) -> str | None:
        normalized = self._normalize_text(text)
        for size, hints in SIZE_HINTS.items():
            if any(hint.lower() in normalized for hint in hints):
                return size
        if price is None:
            return None
        if price >= 1200:
            return "专业版"
        if price >= 500:
            return "大容量"
        return "标准版"

    def _score_categories(self, text: str, *, purpose: str | None = None) -> list[tuple[int, int]]:
        normalized = self._normalize_text(text)
        scores: dict[int, int] = {}
        for category_id, definition in CATEGORY_DEFINITIONS.items():
            score = 0
            if definition["name"].lower() in normalized:
                score += 6
            if self._extract_best_match(normalized, definition["types"]):
                score += 5
            if self._extract_best_match(normalized, definition["brands"]):
                score += 3
            for keyword in CATEGORY_HINTS.get(category_id, []):
                if keyword.lower() in normalized:
                    score += 3
            scores[category_id] = score

        if purpose:
            for category_id in PURPOSE_CATEGORY_MAP.get(purpose, []):
                scores[category_id] = scores.get(category_id, 0) + 2

        scored = list(scores.items())
        scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        return scored

    def infer_category(self, product_name: str, product_type: str = "", purpose: str = "") -> tuple[int, str]:
        text = f"{product_name} {product_type} {purpose}"
        scored = self._score_categories(text, purpose=purpose or None)
        best_category_id = FALLBACK_CATEGORY_ID
        if scored and scored[0][1] > 0:
            best_category_id = scored[0][0]
        return best_category_id, CATEGORY_DEFINITIONS[best_category_id]["name"]

    def _infer_price_hint(self, text: str, category_id: int) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            return float(match.group(1))

        normalized = self._normalize_text(text)
        base = CATEGORY_DEFAULT_PRICE.get(category_id, 299.0)
        if any(token in normalized for token in ["便宜", "平价", "实惠", "省钱"]):
            return max(29.0, base * 0.55)
        if any(token in normalized for token in ["高端", "旗舰", "专业", "高配", "贵一点"]):
            return base * 1.8
        if "中端" in normalized:
            return base * 1.15
        return base

    def infer_product_attributes(
        self,
        *,
        product_name: str,
        price: float,
        category_id: int | None = None,
        category_name: str | None = None,
        brand: str | None = None,
        product_type: str | None = None,
        size: str | None = None,
        purpose: str | None = None,
        target_group: str | None = None,
        market: str | None = None,
        fill_defaults: bool = True,
    ) -> dict[str, Any]:
        normalized = self._normalize_text(product_name)
        inferred_category_id, inferred_category_name = self.infer_category(product_name, product_type or "", purpose or "")
        final_category_id = category_id or inferred_category_id
        final_category_name = (
            category_name
            or CATEGORY_DEFINITIONS[final_category_id]["name"]
            if final_category_id in CATEGORY_DEFINITIONS
            else inferred_category_name
        )
        definition = CATEGORY_DEFINITIONS.get(final_category_id, CATEGORY_DEFINITIONS[inferred_category_id])

        chosen_type = product_type or self._extract_best_match(normalized, definition["types"])
        chosen_brand = brand or self._extract_best_match(normalized, definition["brands"])
        chosen_purpose = purpose or self._derive_purpose(product_name) or CATEGORY_DEFAULT_PURPOSE.get(final_category_id)
        chosen_size = size or self._derive_size(product_name, price)
        chosen_target_group = target_group or self._derive_target_group(product_name, chosen_purpose)
        chosen_market = market or self._derive_market(product_name, price)

        if fill_defaults:
            chosen_type = chosen_type or definition["types"][0]
            chosen_brand = chosen_brand or definition["brands"][0]
            chosen_size = chosen_size or "标准版"
            chosen_purpose = chosen_purpose or CATEGORY_DEFAULT_PURPOSE.get(final_category_id, "家用")
            chosen_target_group = chosen_target_group or "大众用户"
            chosen_market = chosen_market or self._derive_market("", price) or "大众市场"

        return {
            "category_id": final_category_id,
            "category_name": final_category_name,
            "brand": chosen_brand or "",
            "product_type": chosen_type or "",
            "size": chosen_size or "",
            "purpose": chosen_purpose or "",
            "target_group": chosen_target_group or "",
            "market": chosen_market or "",
        }

    def next_item_id(self) -> int:
        latest = self.backend.list_products(limit=1)
        return (latest[0].item_id + 1) if latest else 1

    def _collect_text_tokens(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        tokens: list[str] = []

        for hints in CATEGORY_HINTS.values():
            for hint in hints:
                if hint.lower() in normalized:
                    tokens.append(hint)

        for hints in PURPOSE_HINTS.values():
            for hint in hints:
                if hint.lower() in normalized:
                    tokens.append(hint)

        tokens.extend(re.findall(r"[a-z0-9]+", normalized))

        chinese_only = "".join(re.findall(r"[\u4e00-\u9fff]+", text))
        if 2 <= len(chinese_only) <= 6:
            tokens.append(chinese_only)

        return _dedupe(tokens)[:12]

    def _payload_tokens(self, payload: dict[str, Any]) -> list[str]:
        tokens = []
        for field in ("category_name", "brand", "product_type", "size", "purpose", "target_group", "market", "product_name"):
            value = str(payload.get(field, "") or "")
            if value:
                tokens.extend(self._collect_text_tokens(value))

        if payload.get("category_id"):
            tokens.append(f"category:{int(payload['category_id'])}")
        for candidate in payload.get("category_candidates", []) or []:
            tokens.append(f"category:{int(candidate)}")
        return _dedupe(tokens)

    def parse_query_text(self, query_text: str) -> dict[str, Any]:
        text = query_text.strip()
        if not text:
            raise ValueError("Query text must not be empty.")

        normalized = self._normalize_text(text)
        all_brands = [brand for definition in CATEGORY_DEFINITIONS.values() for brand in definition["brands"]]
        all_types = [product_type for definition in CATEGORY_DEFINITIONS.values() for product_type in definition["types"]]
        found_brand = self._extract_best_match(normalized, all_brands)
        found_type = self._extract_best_match(normalized, all_types)
        purpose = self._derive_purpose(text)

        scored_categories = self._score_categories(text, purpose=purpose)
        top_category_id = FALLBACK_CATEGORY_ID
        top_score = 0
        if scored_categories:
            top_category_id, top_score = scored_categories[0]

        category_candidates = [
            category_id
            for category_id, score in scored_categories
            if score > 0 and score >= max(2, top_score - 1)
        ][:2]

        if top_score <= 0:
            category_candidates = []
            top_category_id = FALLBACK_CATEGORY_ID

        price_hint = self._infer_price_hint(text, top_category_id)
        size = self._derive_size(text, None)
        target_group = self._derive_target_group(text, purpose)
        market = self._derive_market(text, price_hint)

        inferred = self.infer_product_attributes(
            product_name=text,
            price=price_hint,
            category_id=top_category_id,
            brand=found_brand,
            product_type=found_type,
            size=size,
            purpose=purpose,
            target_group=target_group,
            market=market,
            fill_defaults=False,
        )

        inferred["category_confident"] = bool(top_score >= 3)
        inferred["category_candidates"] = category_candidates
        inferred["semantic_tokens"] = self._payload_tokens({**inferred, "product_name": text})
        inferred["product_name"] = text
        inferred["price"] = price_hint

        if not inferred["product_type"] and inferred["category_confident"]:
            inferred["product_type"] = GENERIC_PRODUCT_TYPE.get(int(inferred["category_id"]), "")

        return inferred

    def _price_labels(self, price: float) -> list[tuple[str, float]]:
        labels: list[tuple[str, float]] = []
        if price <= 0:
            return labels
        if price < 100:
            labels.append(("budget", 0.12))
        elif price < 300:
            labels.append(("entry", 0.12))
        elif price < 800:
            labels.append(("mid", 0.12))
        elif price < 1500:
            labels.append(("upper_mid", 0.12))
        else:
            labels.append(("premium", 0.12))
        labels.append((f"bucket:{int(price // 100)}", 0.08))
        return labels

    def build_embedding(self, payload: dict[str, Any]) -> list[float]:
        vector = np.zeros(self.config.vector_dim, dtype=np.float32)

        category_candidates = [int(value) for value in payload.get("category_candidates", []) or []]
        if not category_candidates and payload.get("category_id") is not None:
            category_candidates = [int(payload["category_id"])]

        for index, category_id in enumerate(category_candidates[:3]):
            weight = 0.52 if index == 0 else (0.18 if index == 1 else 0.1)
            vector += weight * self._feature_direction("category", str(category_id))

        category_name = str(payload.get("category_name", "") or "")
        if category_name:
            vector += 0.14 * self._feature_direction("category_name", category_name)

        for field, namespace, weight in (
            ("brand", "brand", 0.15),
            ("product_type", "product_type", 0.2),
            ("size", "size", 0.08),
            ("purpose", "purpose", 0.12),
            ("target_group", "target_group", 0.08),
            ("market", "market", 0.08),
        ):
            value = str(payload.get(field, "") or "")
            if value:
                vector += weight * self._feature_direction(namespace, value)

        price = float(payload.get("price", 0.0) or 0.0)
        for label, weight in self._price_labels(price):
            vector += weight * self._feature_direction("price", label)

        semantic_tokens = [str(token) for token in payload.get("semantic_tokens", []) or self._payload_tokens(payload)]
        for token in semantic_tokens[:10]:
            vector += 0.05 * self._feature_direction("token", token)

        category_id = payload.get("category_id")
        product_type = str(payload.get("product_type", "") or "")
        purpose = str(payload.get("purpose", "") or "")
        if category_id is not None and product_type:
            vector += 0.08 * self._feature_direction("category_type", f"{int(category_id)}::{product_type}")
        if category_id is not None and purpose:
            vector += 0.06 * self._feature_direction("category_purpose", f"{int(category_id)}::{purpose}")

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector = self._feature_direction("fallback", "default").copy()
        return normalize_vector(vector.tolist())

    def add_product(
        self,
        *,
        item_id: int | None = None,
        category_id: int | None = None,
        category_name: str | None = None,
        price: float,
        product_name: str,
        brand: str | None = None,
        product_type: str | None = None,
        size: str | None = None,
        purpose: str | None = None,
        target_group: str | None = None,
        market: str | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        item_id = item_id or self.next_item_id()
        profile = self.infer_product_attributes(
            product_name=product_name,
            price=price,
            category_id=category_id,
            category_name=category_name,
            brand=brand,
            product_type=product_type,
            size=size,
            purpose=purpose,
            target_group=target_group,
            market=market,
        )
        payload = {
            "item_id": item_id,
            "price": price,
            "product_name": product_name,
            **profile,
        }
        vector_payload = dict(payload)
        vector_payload["semantic_tokens"] = self._payload_tokens(payload)
        record = ProductRecord(**payload, embedding=embedding or self.build_embedding(vector_payload))
        result = self.upsert_products([record])
        result["product"] = record.to_payload()
        result["embedding_source"] = "user" if embedding else "generated"
        result["auto_generated"] = {
            "item_id": item_id,
            "category_id": payload["category_id"],
            "category_name": payload["category_name"],
        }
        return result

    def delete_products(self, item_ids: Sequence[int]) -> dict[str, Any]:
        started = perf_counter()
        deleted = self.backend.delete(item_ids)
        latency_ms = round((perf_counter() - started) * 1000.0, 3)
        self._record_operation("delete", latency_ms, rows=deleted)
        return {"backend": self.backend.name, "delete_count": deleted, "latency_ms": latency_ms}

    def search(self, request: SearchRequest | dict[str, Any]) -> dict[str, Any]:
        if isinstance(request, dict):
            request = SearchRequest(
                vector=[float(value) for value in request["vector"]],
                top_k=int(request.get("top_k", 10)),
                category_ids=[int(value) for value in request.get("category_ids", [])] or None,
                min_price=(float(request["min_price"]) if "min_price" in request else None),
                max_price=(float(request["max_price"]) if "max_price" in request else None),
            )
        started = perf_counter()
        hits = self.backend.search(request)
        latency_ms = round((perf_counter() - started) * 1000.0, 3)
        self._record_operation("search", latency_ms, top_k=request.top_k, hits=len(hits), backend=self.backend.name)
        return {
            "backend": self.backend.name,
            "degraded_reason": self.degraded_reason,
            "latency_ms": latency_ms,
            "hits": [hit.to_dict() for hit in hits],
        }

    def search_by_item_id(
        self,
        item_id: int,
        *,
        top_k: int = 10,
        category_ids: list[int] | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        exclude_self: bool = True,
        mode: str = "similar_product",
    ) -> dict[str, Any]:
        product = self.backend.get(item_id)
        if product is None:
            raise KeyError(f"Product {item_id} does not exist.")
        request = SearchRequest(
            vector=product.embedding,
            top_k=max(top_k * 4, 20),
            category_ids=category_ids,
            min_price=min_price,
            max_price=max_price,
        )
        response = self.search(request)
        source_payload = product.to_payload()
        source_payload["semantic_tokens"] = self._payload_tokens(source_payload)
        hits = [hit for hit in response["hits"] if not exclude_self or hit["item_id"] != item_id]
        response["hits"] = self._rerank_hits(source_payload, hits, mode)[:top_k]
        response["query_product"] = source_payload
        response["mode"] = mode
        return response

    def search_by_text(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        category_ids: list[int] | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        mode: str = "similar_product",
    ) -> dict[str, Any]:
        profile = self.parse_query_text(query_text)
        effective_category_ids = category_ids
        if not effective_category_ids and profile.get("category_confident"):
            candidates = [int(value) for value in profile.get("category_candidates", []) or []]
            effective_category_ids = candidates or [int(profile["category_id"])]
        request = SearchRequest(
            vector=self.build_embedding(profile),
            top_k=max(top_k * 6, 30),
            category_ids=effective_category_ids,
            min_price=min_price,
            max_price=max_price,
        )
        response = self.search(request)
        response["hits"] = self._rerank_hits(profile, response["hits"], mode)[:top_k]
        response["query_profile"] = profile
        response["query_text"] = query_text
        response["mode"] = mode
        return response

    def _rerank_hits(self, source: dict[str, Any], hits: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
        source_tokens = set(source.get("semantic_tokens", []) or self._payload_tokens(source))
        candidate_categories = {
            int(source["category_id"])
        } if source.get("category_id") is not None else set()
        candidate_categories.update(int(value) for value in source.get("category_candidates", []) or [])

        def price_similarity(target_price: float) -> float:
            source_price = max(float(source.get("price", 0.0) or 0.0), 1.0)
            gap = abs(float(source_price) - float(target_price))
            return max(0.0, 1.0 - gap / source_price)

        scored: list[tuple[float, dict[str, Any]]] = []
        for hit in hits:
            vector_score = float(hit["score"])
            hit_tokens = set(self._payload_tokens(hit))
            overlap = (2.0 * len(source_tokens & hit_tokens) / (len(source_tokens) + len(hit_tokens))) if (source_tokens or hit_tokens) else 0.0

            hit_category = int(hit["category_id"])
            if source.get("category_id") is not None and hit_category == int(source["category_id"]):
                category_score = 1.0
            elif hit_category in candidate_categories:
                category_score = 0.75
            else:
                category_score = 0.0

            brand_score = 1.0 if source.get("brand") and hit.get("brand") == source.get("brand") else 0.0
            type_score = 1.0 if source.get("product_type") and hit.get("product_type") == source.get("product_type") else 0.0
            purpose_score = 1.0 if source.get("purpose") and hit.get("purpose") == source.get("purpose") else 0.0
            target_score = 1.0 if source.get("target_group") and hit.get("target_group") == source.get("target_group") else 0.0
            market_score = 1.0 if source.get("market") and hit.get("market") == source.get("market") else 0.0
            price_score = price_similarity(hit["price"])

            if mode == "category":
                final_score = category_score * 0.48 + overlap * 0.2 + vector_score * 0.17 + brand_score * 0.05 + type_score * 0.05 + market_score * 0.05
            elif mode == "similar_price":
                final_score = price_score * 0.42 + overlap * 0.18 + vector_score * 0.15 + category_score * 0.15 + market_score * 0.1
            else:
                final_score = vector_score * 0.34 + overlap * 0.24 + category_score * 0.18 + type_score * 0.08 + brand_score * 0.05 + purpose_score * 0.04 + target_score * 0.03 + price_score * 0.04

            enriched = dict(hit)
            enriched["match_score"] = round(final_score, 4)
            enriched["price_gap"] = round(abs(float(source.get("price", 0.0) or 0.0) - float(hit["price"])), 2)
            scored.append((final_score, enriched))

        scored.sort(key=lambda entry: entry[0], reverse=True)
        return [item for _, item in scored]

    def metrics_snapshot(self) -> dict[str, Any]:
        status = self.status()
        with self._metrics_lock:
            operations = deepcopy(self._operations)
            jobs = sorted((deepcopy(job) for job in self._jobs.values()), key=lambda item: item["created_at"], reverse=True)[:8]
            running_jobs = [job for job in jobs if job["status"] in {"queued", "running"}]
            current_job = running_jobs[0] if running_jobs else (jobs[0] if jobs else None)
            last_benchmark = deepcopy(self._last_benchmark)
            last_market_benchmark = deepcopy(self._last_market_benchmark)
        return {
            "timestamp": round(time(), 3),
            "uptime_seconds": round(time() - self._started_at, 1),
            "status": status,
            "operations": operations,
            "current_job": current_job,
            "jobs": jobs,
            "evaluation_dimensions": EVALUATION_DIMENSIONS,
            "database_comparison": self._database_comparison(status, operations, last_benchmark, last_market_benchmark),
            "last_benchmark": last_benchmark,
            "last_market_benchmark": last_market_benchmark,
            "limits": {"min_generate_count": MIN_GENERATE_COUNT, "max_generate_count": MAX_GENERATE_COUNT},
        }

    def _database_comparison(
        self,
        status: dict[str, Any],
        operations: dict[str, dict[str, Any]],
        last_benchmark: dict[str, Any] | None,
        last_market_benchmark: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        search_stats = operations.get("search", {})
        upsert_stats = operations.get("upsert", {})
        delete_stats = operations.get("delete", {})
        benchmark_rows = (last_benchmark or {}).get("rows", []) if last_benchmark else []
        latest_recall = "-"
        latest_mrr = "-"
        latest_ndcg = "-"
        latest_dataset_size = status.get("count", 0)
        if benchmark_rows:
            latest = benchmark_rows[-1]
            latest_recall = latest.get("avg_recall_at_k", "-")
            latest_mrr = latest.get("avg_mrr_at_k", "-")
            latest_ndcg = latest.get("avg_ndcg_at_k", "-")
            latest_dataset_size = latest.get("dataset_size", latest_dataset_size)

        current = {
            "name": f"当前系统（{status.get('backend', self.backend.name)}）",
            "type": "运行实测",
            "data_source": "实时监测",
            "source_url": "",
            "dataset_count": status.get("count", 0),
            "benchmark_dataset_size": latest_dataset_size,
            "avg_search_ms": search_stats.get("avg_ms", "-"),
            "last_search_ms": search_stats.get("last_ms", "-"),
            "avg_insert_ms": upsert_stats.get("avg_ms", "-"),
            "last_delete_ms": delete_stats.get("last_ms", "-"),
            "recall_at_k": latest_recall,
            "mrr_at_k": latest_mrr,
            "ndcg_at_k": latest_ndcg,
            "vector_support": "已接入",
            "filter_support": "类目 + 价格过滤",
            "hybrid_search": "向量召回 + 属性过滤 + 业务重排",
            "dynamic_update": "新增 / 删除 / upsert",
            "distributed": "Milvus 模式支持扩展，本地 ANN 用于演示",
            "fit": "本课题实测对象",
            "measured_summary": (
                f"数据量 {status.get('count', 0)}；平均检索 {search_stats.get('avg_ms', '-')} ms；"
                f"平均写入 {upsert_stats.get('avg_ms', '-')} ms；Recall@K {latest_recall}；"
                f"MRR@K {latest_mrr}；NDCG@K {latest_ndcg}"
            ),
        }
        measured_rows = (last_market_benchmark or {}).get("rows", []) if last_market_benchmark else []
        measured_by_name = {str(row.get("name")): row for row in measured_rows}
        current_market_row = measured_by_name.get("当前系统")
        if current_market_row:
            current.update(
                {
                    "data_source": current_market_row.get("data_source", current["data_source"]),
                    "sample_size": current_market_row.get("sample_size"),
                    "avg_search_ms": current_market_row.get("avg_search_ms", current["avg_search_ms"]),
                    "recall_at_k": current_market_row.get("recall_at_k", current["recall_at_k"]),
                    "mrr_at_k": current_market_row.get("mrr_at_k", current["mrr_at_k"]),
                    "ndcg_at_k": current_market_row.get("ndcg_at_k", current["ndcg_at_k"]),
                    "measured_summary": current_market_row.get("measured_summary", current["measured_summary"]),
                }
            )

        local_measured_profiles = []
        for row in measured_rows:
            if row.get("name") == "当前系统":
                continue
            local_measured_profiles.append(
                {
                    **row,
                    "benchmark_dataset_size": row.get("sample_size", "-"),
                    "last_search_ms": "-",
                    "last_delete_ms": row.get("delete_ms", "-"),
                }
            )

        profiles = []
        for profile in MARKET_DATABASE_COMPARISON_PROFILES:
            if profile["name"] == "Milvus" and status.get("backend") == "milvus":
                continue
            profiles.append(
                {
                    **profile,
                    "dataset_count": "待接入同一数据集实测",
                    "benchmark_dataset_size": "-",
                    "avg_search_ms": "待实测",
                    "last_search_ms": "-",
                    "avg_insert_ms": "待实测",
                    "last_delete_ms": "待实测",
                    "recall_at_k": "待实测",
                    "mrr_at_k": "待实测",
                    "ndcg_at_k": "待实测",
                    "measured_summary": "未配置外部连接；需安装对应客户端并提供连接参数后导入同一商品数据集实测",
                }
            )
        return [current, *local_measured_profiles, *profiles]

    def benchmark(
        self,
        sizes: list[int],
        *,
        queries_per_size: int = 20,
        top_k: int = 10,
        seed: int = 42,
    ) -> dict[str, Any]:
        sizes = self._validate_benchmark_sizes(sizes)
        started = perf_counter()
        result = run_benchmark(self, sizes=sizes, queries_per_size=queries_per_size, top_k=top_k, seed=seed)
        latency_ms = round((perf_counter() - started) * 1000.0, 3)
        result["latency_ms"] = latency_ms
        with self._metrics_lock:
            self._last_benchmark = deepcopy(result)
        self._record_operation("benchmark", latency_ms, sizes=sizes, rows=len(result.get("rows", [])))
        return result
