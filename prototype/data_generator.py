from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from .models import ProductRecord, normalize_vector


CATEGORY_DEFINITIONS = {
    1: {"name": "手机数码", "types": ["手机", "耳机", "平板", "键盘", "音箱"], "brands": ["华为", "小米", "Apple", "索尼", "漫步者"]},
    2: {"name": "家用电器", "types": ["咖啡机", "净化器", "空调", "冰箱", "洗衣机"], "brands": ["美的", "海尔", "格力", "松下", "飞利浦"]},
    3: {"name": "服饰鞋包", "types": ["外套", "卫衣", "跑鞋", "背包", "衬衫"], "brands": ["耐克", "阿迪达斯", "优衣库", "安踏", "李宁"]},
    4: {"name": "美妆护肤", "types": ["面霜", "精华", "口红", "防晒", "洁面"], "brands": ["欧莱雅", "兰蔻", "珀莱雅", "雅诗兰黛", "完美日记"]},
    5: {"name": "食品生鲜", "types": ["礼盒", "牛排", "坚果", "咖啡豆", "乳品"], "brands": ["三只松鼠", "良品铺子", "盒马", "伊利", "雀巢"]},
    6: {"name": "家居日用", "types": ["桌灯", "收纳箱", "床品", "毛巾", "餐具"], "brands": ["网易严选", "MUJI", "九阳", "苏泊尔", "罗莱"]},
    7: {"name": "运动户外", "types": ["帐篷", "冲锋衣", "运动手表", "瑜伽垫", "登山包"], "brands": ["迪卡侬", "佳明", "北面", "凯乐石", "探路者"]},
    8: {"name": "图书文创", "types": ["图书", "手账", "钢笔", "拼图", "文具套装"], "brands": ["晨光", "得力", "中信出版社", "人民文学", "乐高"]},
}

SIZE_OPTIONS = ["XS", "S", "M", "L", "XL", "便携款", "标准版", "大容量", "家庭装", "专业版"]
PURPOSE_OPTIONS = ["通勤", "家用", "学习", "办公", "礼赠", "运动", "户外", "美妆护理", "餐饮", "内容消费"]
TARGET_GROUPS = ["大众用户", "学生", "白领", "家庭用户", "母婴人群", "运动人群", "高端消费群体"]
MARKETS = ["大众市场", "中端市场", "高端市场", "新锐市场", "家庭消费市场", "专业细分市场"]
ADJECTIVES = ["旗舰", "智能", "轻量", "静音", "专业", "经典", "便携", "舒适", "潮流", "高性能"]


def generate_products(
    count: int,
    dim: int,
    *,
    seed: int = 42,
    id_start: int = 1,
    category_count: int = 8,
) -> list[ProductRecord]:
    return list(iter_products(count=count, dim=dim, seed=seed, id_start=id_start, category_count=category_count))


def iter_products(
    count: int,
    dim: int,
    *,
    seed: int = 42,
    id_start: int = 1,
    category_count: int = 8,
) -> Iterator[ProductRecord]:
    if category_count > len(CATEGORY_DEFINITIONS):
        raise ValueError(f"category_count must be <= {len(CATEGORY_DEFINITIONS)}")
    rng = np.random.default_rng(seed)
    category_ids = list(range(1, category_count + 1))
    centers = _build_centers(dim, category_ids, seed + 100)
    price_bases = {category_id: 59.0 + category_id * 85.0 for category_id in category_ids}

    for offset in range(count):
        category_id = int(rng.choice(category_ids))
        definition = CATEGORY_DEFINITIONS[category_id]
        center = centers[category_id]
        product_type = str(rng.choice(definition["types"]))
        brand = str(rng.choice(definition["brands"]))
        size = str(rng.choice(SIZE_OPTIONS))
        purpose = str(rng.choice(PURPOSE_OPTIONS))
        target_group = str(rng.choice(TARGET_GROUPS))
        market = str(rng.choice(MARKETS))
        noise = rng.normal(loc=0.0, scale=0.11, size=dim).astype(np.float32)
        embedding = normalize_vector(center + noise)
        price = max(9.9, float(rng.normal(price_bases[category_id], 35.0)))
        product_name = f"{definition['name']}-{brand}-{rng.choice(ADJECTIVES)}{product_type}-{id_start + offset}"
        yield ProductRecord(
            item_id=id_start + offset,
            category_id=category_id,
            category_name=definition["name"],
            price=round(price, 2),
            product_name=product_name,
            brand=brand,
            product_type=product_type,
            size=size,
            purpose=purpose,
            target_group=target_group,
            market=market,
            embedding=embedding,
        )


def _build_centers(dim: int, category_ids: Sequence[int], seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers: dict[int, np.ndarray] = {}
    for category_id in category_ids:
        center = rng.normal(size=dim).astype(np.float32)
        centers[category_id] = np.asarray(normalize_vector(center), dtype=np.float32)
    return centers
