from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


WEB_ROOT = Path(__file__).with_name("web")
PROJECT_ROOT = WEB_ROOT.parent.parent


class PrototypeRequestHandler(BaseHTTPRequestHandler):
    server_version = "MilvaPrototype/1.0"

    @property
    def system(self):
        return self.server.system  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/monitor.html":
            self._send_file(WEB_ROOT / "monitor.html", "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/assets/"):
            asset_path = self._resolve_web_path(parsed.path)
            if asset_path is None:
                self._send_json(404, {"error": "File not found."})
                return
            content_type = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
            self._send_file(asset_path, content_type)
            return
        if parsed.path == "/health":
            self._send_json(200, self.system.status())
            return
        if parsed.path == "/metrics":
            self._send_json(200, self.system.metrics_snapshot())
            return
        if parsed.path.startswith("/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            try:
                self._send_json(200, self.system.job_status(job_id))
            except KeyError as exc:
                self._send_json(404, {"error": str(exc)})
            return
        if parsed.path == "/products":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["12"])[0])
            self._send_json(200, self.system.list_products(limit=limit))
            return
        if parsed.path.startswith("/products/"):
            item_id = self._extract_item_id(parsed.path)
            if item_id is None:
                self._send_json(404, {"error": "Invalid product path."})
                return
            product = self.system.get_product(item_id)
            if product is None:
                self._send_json(404, {"error": f"Product {item_id} not found."})
                return
            self._send_json(200, product)
            return
        self._send_json(404, {"error": "Route not found."})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        payload = self._read_json()
        if parsed.path == "/reset":
            self._send_json(200, self.system.reset())
            return
        if parsed.path == "/bootstrap":
            count = int(payload.get("count", 1000))
            seed = int(payload.get("seed", 42))
            id_start = int(payload.get("id_start", 1))
            try:
                self._send_json(200, self.system.bootstrap_demo_data(count, seed=seed, id_start=id_start))
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            return
        if parsed.path == "/jobs/generate":
            try:
                self._send_json(
                    202,
                    self.system.start_bootstrap_job(
                        int(payload.get("count", 1000)),
                        seed=int(payload.get("seed", 42)),
                        id_start=int(payload.get("id_start", 1)),
                        reset_first=bool(payload.get("reset_first", True)),
                        batch_size=int(payload.get("batch_size", 500)),
                    ),
                )
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            return
        if parsed.path == "/jobs/benchmark":
            try:
                self._send_json(
                    202,
                    self.system.start_benchmark_job(
                        [int(value) for value in payload.get("sizes", [1000, 3000, 5000])],
                        queries_per_size=int(payload.get("queries_per_size", 20)),
                        top_k=int(payload.get("top_k", 10)),
                        seed=int(payload.get("seed", 42)),
                    ),
                )
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            return
        if parsed.path == "/jobs/market-benchmark":
            try:
                self._send_json(
                    202,
                    self.system.start_market_benchmark_job(
                        sample_size=int(payload.get("sample_size", 1000)),
                        queries_per_size=int(payload.get("queries_per_size", 20)),
                        top_k=int(payload.get("top_k", 10)),
                        seed=int(payload.get("seed", 42)),
                    ),
                )
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            return
        if parsed.path == "/products/upsert":
            self._send_json(200, self.system.upsert_products(payload.get("products", [])))
            return
        if parsed.path == "/products/add":
            self._send_json(
                200,
                self.system.add_product(
                    item_id=(int(payload["item_id"]) if "item_id" in payload and str(payload["item_id"]).strip() else None),
                    category_id=(int(payload["category_id"]) if "category_id" in payload and str(payload["category_id"]).strip() else None),
                    category_name=payload.get("category_name"),
                    price=float(payload["price"]),
                    product_name=str(payload["product_name"]),
                    brand=payload.get("brand"),
                    product_type=payload.get("product_type"),
                    size=payload.get("size"),
                    purpose=payload.get("purpose"),
                    target_group=payload.get("target_group"),
                    market=payload.get("market"),
                    embedding=payload.get("embedding"),
                ),
            )
            return
        if parsed.path == "/products/search":
            self._send_json(200, self.system.search(payload))
            return
        if parsed.path == "/products/search/by-id":
            item_id = int(payload["item_id"])
            response = self.system.search_by_item_id(
                item_id,
                top_k=int(payload.get("top_k", 10)),
                category_ids=[int(value) for value in payload.get("category_ids", [])] or None,
                min_price=(float(payload["min_price"]) if "min_price" in payload else None),
                max_price=(float(payload["max_price"]) if "max_price" in payload else None),
                exclude_self=bool(payload.get("exclude_self", True)),
                mode=str(payload.get("mode", "similar_product")),
            )
            self._send_json(200, response)
            return
        if parsed.path == "/products/search/by-text":
            response = self.system.search_by_text(
                str(payload["query_text"]),
                top_k=int(payload.get("top_k", 10)),
                category_ids=[int(value) for value in payload.get("category_ids", [])] or None,
                min_price=(float(payload["min_price"]) if "min_price" in payload else None),
                max_price=(float(payload["max_price"]) if "max_price" in payload else None),
                mode=str(payload.get("mode", "similar_product")),
            )
            self._send_json(200, response)
            return
        if parsed.path == "/benchmark":
            sizes = [int(value) for value in payload.get("sizes", [1000, 3000, 5000])]
            try:
                response = self.system.benchmark(
                    sizes=sizes,
                    queries_per_size=int(payload.get("queries_per_size", 20)),
                    top_k=int(payload.get("top_k", 10)),
                    seed=int(payload.get("seed", 42)),
                )
                self._send_json(200, response)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            return
        self._send_json(404, {"error": "Route not found."})

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/products/"):
            item_id = self._extract_item_id(parsed.path)
            if item_id is None:
                self._send_json(404, {"error": "Invalid product path."})
                return
            self._send_json(200, self.system.delete_products([item_id]))
            return
        self._send_json(404, {"error": "Route not found."})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json(404, {"error": "File not found."})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _resolve_web_path(request_path: str) -> Path | None:
        if request_path == "/assets/1.png":
            root_image = (PROJECT_ROOT / "1.png").resolve()
            if root_image.exists() and root_image.is_file():
                return root_image
            return None

        relative = request_path.lstrip("/")
        candidate = (WEB_ROOT / relative).resolve()
        web_root = WEB_ROOT.resolve()
        try:
            candidate.relative_to(web_root)
        except ValueError:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    @staticmethod
    def _extract_item_id(path: str) -> int | None:
        try:
            return int(path.rsplit("/", 1)[-1])
        except ValueError:
            return None


def serve(system: Any, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), PrototypeRequestHandler)
    server.system = system  # type: ignore[attr-defined]
    print(f"Prototype API listening on http://{host}:{port}")
    print("UI: GET /")
    print("Monitor: GET /monitor.html")
    print("Endpoints: GET /health, GET /metrics, GET /products, POST /jobs/generate, POST /jobs/benchmark, POST /jobs/market-benchmark, POST /products/add, POST /products/search/by-id, POST /products/search/by-text")
    server.serve_forever()
