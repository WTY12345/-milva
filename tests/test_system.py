from __future__ import annotations

import unittest

from prototype import ProductRecord, ProductVectorSystem, SearchRequest, SystemConfig
from prototype.benchmark import run_benchmark


def make_product(
    item_id: int,
    category_id: int,
    category_name: str,
    price: float,
    product_name: str,
    embedding: list[float],
) -> ProductRecord:
    return ProductRecord(
        item_id=item_id,
        category_id=category_id,
        category_name=category_name,
        price=price,
        product_name=product_name,
        brand="测试品牌",
        product_type="测试类型",
        size="标准版",
        purpose="家用",
        target_group="大众用户",
        market="大众市场",
        embedding=embedding,
    )


class ProductVectorSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        config = SystemConfig(backend="local_ann", vector_dim=4, local_hash_bits=6)
        self.system = ProductVectorSystem(config)
        self.system.reset()
        self.products = [
            make_product(1, 1, "手机数码", 100.0, "A", [1, 0, 0, 0]),
            make_product(2, 1, "手机数码", 120.0, "B", [0.9, 0.1, 0, 0]),
            make_product(3, 2, "家用电器", 260.0, "C", [0, 1, 0, 0]),
        ]
        self.system.upsert_products(self.products)

    def test_search_returns_self_as_top_hit(self) -> None:
        response = self.system.search(SearchRequest(vector=[1, 0, 0, 0], top_k=2))
        self.assertEqual(response["hits"][0]["item_id"], 1)

    def test_search_by_item_id_excludes_self(self) -> None:
        response = self.system.search_by_item_id(1, top_k=1)
        self.assertEqual(response["hits"][0]["item_id"], 2)

    def test_filter_by_category(self) -> None:
        response = self.system.search(SearchRequest(vector=[1, 0, 0, 0], top_k=5, category_ids=[2]))
        self.assertEqual([hit["item_id"] for hit in response["hits"]], [3])

    def test_delete_product(self) -> None:
        self.system.delete_products([3])
        self.assertIsNone(self.system.get_product(3))

    def test_add_product_auto_generates_id_and_category(self) -> None:
        result = self.system.add_product(product_name="索尼降噪耳机", price=899.0)
        self.assertEqual(result["embedding_source"], "generated")
        self.assertTrue(result["product"]["item_id"] >= 4)
        self.assertEqual(result["product"]["category_name"], "手机数码")

    def test_search_mode_category(self) -> None:
        response = self.system.search_by_item_id(1, top_k=2, mode="category")
        self.assertEqual(response["mode"], "category")
        self.assertTrue(all("match_score" in hit for hit in response["hits"]))

    def test_search_by_text(self) -> None:
        self.system.add_product(product_name="适合白领通勤的降噪耳机", price=799.0)
        response = self.system.search_by_text("适合我这种白领的通勤耳机", top_k=3, mode="similar_product")
        self.assertEqual(response["mode"], "similar_product")
        self.assertIn("query_profile", response)
        self.assertIsInstance(response["hits"], list)

    def test_generic_wear_query_prefers_clothing(self) -> None:
        self.system.reset()
        self.system.add_product(product_name="安踏运动卫衣", price=299.0)
        self.system.add_product(product_name="盒马坚果礼盒", price=129.0)
        response = self.system.search_by_text("穿的", top_k=3, mode="similar_product")
        self.assertEqual(response["query_profile"]["category_name"], "服饰鞋包")
        self.assertTrue(response["hits"])
        self.assertEqual(response["hits"][0]["category_name"], "服饰鞋包")

    def test_generic_play_query_avoids_food_category(self) -> None:
        self.system.reset()
        self.system.add_product(product_name="乐高益智拼图", price=89.0)
        self.system.add_product(product_name="探路者露营帐篷", price=499.0)
        self.system.add_product(product_name="盒马牛排礼盒", price=129.0)
        response = self.system.search_by_text("玩的", top_k=3, mode="similar_product")
        self.assertEqual(response["query_profile"]["category_name"], "图书文创")
        self.assertTrue(response["hits"])
        self.assertNotEqual(response["hits"][0]["category_name"], "食品生鲜")

    def test_bootstrap_count_range(self) -> None:
        with self.assertRaises(ValueError):
            self.system.bootstrap_demo_data(99)

    def test_metrics_snapshot_contains_comparison_rows(self) -> None:
        self.system.search(SearchRequest(vector=[1, 0, 0, 0], top_k=2))
        metrics = self.system.metrics_snapshot()
        self.assertIn("database_comparison", metrics)
        self.assertTrue(metrics["database_comparison"])
        self.assertEqual(metrics["database_comparison"][0]["data_source"], "实时监测")

    def test_ranking_metrics_include_mrr_and_ndcg(self) -> None:
        recall, mrr, ndcg = self.system._ranking_metrics([2, 3, 1], [1, 2, 3])
        self.assertEqual(recall, 1.0)
        self.assertEqual(mrr, 1.0)
        self.assertGreater(ndcg, 0.0)
        self.assertLess(ndcg, 1.0)

    def test_benchmark_reports_mrr_and_ndcg(self) -> None:
        result = run_benchmark(self.system, sizes=[100], queries_per_size=3, top_k=5, seed=7)
        row = result["rows"][0]
        self.assertIn("avg_mrr_at_k", row)
        self.assertIn("avg_ndcg_at_k", row)
        self.assertGreaterEqual(row["avg_mrr_at_k"], 0.0)
        self.assertLessEqual(row["avg_mrr_at_k"], 1.0)
        self.assertGreaterEqual(row["avg_ndcg_at_k"], 0.0)
        self.assertLessEqual(row["avg_ndcg_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
