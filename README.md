# 基于 Milvus 的海量商品向量实时存储与近似检索原型系统

这个仓库把论文草稿里的第五章和第六章直接落成了一个可运行的原型，覆盖了商品向量的实时写入、近似检索、条件过滤、动态更新、删除和实验评估。

## 设计目标

- 对应论文需求分析中的四个核心能力：`向量存储`、`索引构建`、`相似检索`、`动态更新`
- 默认优先使用 `Milvus` 作为向量数据库后端
- 当当前环境无法直接启用本地 `Milvus Lite` 时，自动降级为本地 `Local ANN` 原型，以保证系统现在就可以跑通
- 保留统一的服务层和接口层，便于后续扩展成 Web 服务或毕业设计答辩演示系统

## 系统架构

```text
数据生成/业务输入
        |
        v
ProductVectorSystem
        |
        +-- MilvusVectorBackend
        |      |- 集合创建
        |      |- AUTOINDEX 索引
        |      |- 向量 upsert / delete / search
        |
        +-- LocalAnnVectorBackend
               |- 随机超平面哈希
               |- 桶化候选召回
               |- 余弦相似度重排
```

## 数据模型

集合设计：

- `item_id`: 商品主键
- `category_id`: 商品类别
- `price`: 商品价格
- `product_name`: 商品名称
- `embedding`: 商品向量

## 目录说明

- `main.py`: CLI 入口
- `prototype/service.py`: 统一业务服务层
- `prototype/backends.py`: Milvus 后端与本地 ANN 降级后端
- `prototype/data_generator.py`: 模拟商品向量数据生成器
- `prototype/benchmark.py`: 实验评估脚本
- `prototype/http_api.py`: 简易 HTTP 原型接口
- `tests/test_system.py`: 基础功能测试

## 环境说明

当前目录里的虚拟环境已经安装了 `pymilvus` 和 `numpy`。但在本机 Python `3.14.2` 下，本地 `Milvus Lite` 依赖 `milvus-lite` 无法直接安装，所以：

- 如果你有运行中的 Milvus 服务，可使用 `--backend milvus --milvus-uri http://127.0.0.1:19530`
- 如果没有，系统会自动切到 `local_ann`，依然可以完成毕业设计原型演示、接口联调和实验评估

这意味着代码层面仍然是“基于 Milvus 的系统设计”，同时兼顾当前开发环境可运行性。

## 解决 Milvus Lite 问题

`milvus-lite` 目前不支持在 Windows 上直接安装，所以本项目改用官方支持的 `Docker Milvus Standalone` 作为本机 Milvus 运行方式。

直接启动 Milvus：

```powershell
docker compose up -d
```

或者双击运行：

- `start_milvus.bat`
- `stop_milvus.bat`

启动后可先检查：

```powershell
python main.py --backend milvus --milvus-uri http://127.0.0.1:19530 status
```

再启动网页：

```powershell
python main.py --backend milvus --milvus-uri http://127.0.0.1:19530 serve --host 127.0.0.1 --port 8765
```

这样就不再走 `local_ann` 降级路径，而是直接使用真正的 Milvus 后端。

## 快速开始

直接启动：

```powershell
python main.py status
python main.py demo --count 2000 --top-k 5
python main.py benchmark --sizes 1000 3000 5000
```

连接远程或本地已运行的 Milvus：

```powershell
python main.py --backend milvus --milvus-uri http://127.0.0.1:19530 status
python main.py --backend milvus --milvus-uri http://127.0.0.1:19530 bootstrap --count 5000
python main.py --backend milvus --milvus-uri http://127.0.0.1:19530 search --item-id 1 --top-k 10
```

## 可视化网页

启动服务后，直接打开浏览器访问 `http://127.0.0.1:8765/`，就能看到可视化页面。页面支持：

- 添加商品
- 自动生成商品向量
- 一次生成指定数量商品数据，范围为 `100-1000000`
- 查看最近写入商品
- 按商品 ID 检索相似商品
- 查看当前后端状态

实时评测监控页：

```text
http://127.0.0.1:8765/monitor.html
```

监控页支持：

- 后台生成 100-1000000 条商品向量，并实时显示生成进度
- 实时显示商品总量、平均检索延迟、写入耗时、删除耗时等运行指标
- 启动 benchmark 评测，记录插入耗时、平均检索耗时、Recall@K、MRR@K、NDCG@K、更新耗时和删除耗时
- 展示 Pinecone、Zilliz Cloud、Milvus、Qdrant、Weaviate、Elasticsearch/OpenSearch、PostgreSQL pgvector、Redis Vector、Faiss 等市场已有系统的能力对比
- 支持将当前已生成的商品数据快照接入可本地复现的 Local ANN 与精确顺序扫描基线，生成同数据集实测结果；外部云服务需配置连接后再导入实测

## HTTP 原型接口

启动服务：

```powershell
python main.py serve --host 127.0.0.1 --port 8765
```

可用接口：

- `GET /`
- `GET /health`
- `GET /metrics`
- `GET /monitor.html`
- `GET /products?limit=12`
- `POST /reset`
- `POST /bootstrap`
- `POST /jobs/generate`
- `POST /jobs/benchmark`
- `POST /products/add`
- `POST /products/upsert`
- `POST /products/search`
- `POST /products/search/by-id`
- `POST /products/search/by-text`
- `GET /products/{item_id}`
- `DELETE /products/{item_id}`
- `POST /benchmark`

## 论文映射建议

映射到论文的章节结构：

- 第 4 章总体设计：使用 `README` 中的架构图、模块职责和数据模型
- 第 5 章详细设计与实现：结合 `prototype/backends.py`、`prototype/service.py`、`main.py`
- 第 6 章实验评估：使用 `main.py benchmark` 的结果填充查询时延、Recall@K、MRR@K、NDCG@K、更新时延等指标

## 测试

```powershell
python -m unittest discover -s tests -v
```
