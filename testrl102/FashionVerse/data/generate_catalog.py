"""
FashionVerse — Fashion Catalog Generator
Generates a realistic local fashion catalog (no external APIs required).
Run once to create fashion_items.csv used by all downstream components.
"""

import json
import random
import csv
import os

random.seed(42)

# ── Palette and attribute pools ──────────────────────────────────────────────

COLORS = [
    "black", "white", "navy", "grey", "beige", "olive", "maroon",
    "royal_blue", "forest_green", "burnt_orange", "dusty_rose",
    "lavender", "teal", "mustard", "cream", "charcoal", "rust",
    "sage_green", "wine", "sky_blue", "coral", "tan", "blush",
    "slate", "indigo"
]

PATTERNS = ["solid", "stripes", "checks", "floral", "geometric",
            "abstract", "animal_print", "embroidered", "block_print", "tie_dye"]

MATERIALS = ["cotton", "polyester", "linen", "denim", "silk", "rayon",
             "wool", "chiffon", "georgette", "velvet", "jersey", "knit",
             "canvas", "suede", "leather", "synthetic_leather"]

OCCASIONS = ["casual", "college", "office", "semi_formal", "formal",
             "party", "festive", "date", "travel", "gym", "beach"]

SEASONS = ["summer", "winter", "monsoon", "all"]

STYLES = ["casual", "streetwear", "formal", "semi_formal", "minimalist",
          "bohemian", "athleisure", "indo_western", "festive", "preppy"]

GENDERS = ["male", "female", "unisex"]


def rand_occasions(n=3):
    return random.sample(OCCASIONS, min(n, len(OCCASIONS)))


def rand_seasons(n=2):
    pool = ["summer", "winter", "monsoon", "all"]
    return random.sample(pool, random.randint(1, n))


def rand_color_pair():
    c1 = random.choice(COLORS)
    c2 = random.choice([c for c in COLORS if c != c1] + [None, None])
    return c1, c2


def comfort(price_hint, material):
    base = {"cotton": 0.90, "linen": 0.85, "jersey": 0.88, "knit": 0.87,
            "rayon": 0.82, "polyester": 0.70, "silk": 0.80, "chiffon": 0.75,
            "denim": 0.72, "georgette": 0.73, "wool": 0.78, "velvet": 0.76,
            "canvas": 0.68, "suede": 0.65, "leather": 0.60,
            "synthetic_leather": 0.58, "chiffon": 0.74}
    return round(base.get(material, 0.75) + random.uniform(-0.05, 0.05), 2)


def versatility(occasions, style):
    base = min(len(occasions) / len(OCCASIONS) * 1.5, 1.0)
    style_bonus = {"minimalist": 0.10, "casual": 0.08, "formal": 0.05}.get(style, 0.0)
    return round(min(base + style_bonus + random.uniform(-0.05, 0.05), 1.0), 2)


# ── Tops (100) ───────────────────────────────────────────────────────────────

TOP_NAMES = [
    "Oversized Cotton T-Shirt", "Slim Fit Polo", "Linen Button-Down Shirt",
    "Striped Crew Neck Tee", "Formal Oxford Shirt", "Casual Henley",
    "Graphic Print Tee", "Mandarin Collar Kurta", "Denim Shirt",
    "Chiffon Blouse", "Crop Top", "Tank Top", "Peplum Top",
    "Off-Shoulder Top", "Tube Top", "Wrap Blouse", "Embroidered Kurti",
    "Cold Shoulder Top", "Halter Neck Top", "Sleeveless Linen Top",
    "Ribbed Knit Top", "Ruffle Neck Blouse", "Peter Pan Collar Top",
    "V-Neck Tunic", "Sheer Overlay Top"
]

BOTTOM_NAMES = [
    "Slim Fit Chinos", "High-Waist Jeans", "Relaxed Fit Trousers",
    "Pleated Formal Trousers", "Cargo Pants", "Jogger Pants",
    "Flared Jeans", "Mom Jeans", "Straight-Leg Trousers",
    "Palazzo Pants", "Mini Skirt", "Midi Skirt", "Maxi Skirt",
    "Pencil Skirt", "A-Line Skirt", "Wrap Skirt", "Denim Shorts",
    "Bermuda Shorts", "Dhoti Pants", "Culottes",
    "Wide Leg Trousers", "Tapered Trousers", "Paper Bag Waist Pants",
    "Pleated Midi Skirt", "Athletic Shorts"
]

DRESS_NAMES = [
    "Maxi Wrap Dress", "A-Line Midi Dress", "Bodycon Dress",
    "Shift Dress", "Shirt Dress", "Sundress", "Slip Dress",
    "Blazer Dress", "Tiered Ruffle Dress", "Fit and Flare Dress",
    "Linen Shirt Dress", "Floral Midi Dress", "Ethnic Kurta Dress",
    "Asymmetric Hem Dress", "Pleated Dress"
]

SHOE_NAMES = [
    "White Canvas Sneakers", "Chunky Sole Sneakers", "Running Shoes",
    "Slip-On Loafers", "Oxford Brogues", "Derby Shoes",
    "Block Heel Pumps", "Stiletto Heels", "Wedge Sandals",
    "Flat Sandals", "Platform Sneakers", "Chelsea Boots",
    "Ankle Boots", "Kolhapuri Sandals", "Espadrilles",
    "Ballet Flats", "Mary Janes", "Mules", "Kitten Heels",
    "Sports Sliders"
]

ACCESSORY_NAMES = [
    "Minimal Silver Necklace", "Gold Hoop Earrings", "Beaded Bracelet",
    "Leather Belt", "Canvas Tote Bag", "Crossbody Bag", "Backpack",
    "Wide Brim Hat", "Baseball Cap", "Silk Scarf", "Statement Ring",
    "Layered Chain Necklace", "Stud Earrings", "Oxidised Jhumkas",
    "Fabric Headband", "Sunglasses", "Watch", "Anklet",
    "Denim Belt Bag", "Jute Bag"
]


def make_item(idx, category, subcategory, name_pool, gender_pool,
              price_range, formality_range):
    item_id = f"{category.upper()[:3]}_{idx:03d}"
    name_prefix = random.choice(COLORS).replace("_", " ").title()
    base_name = random.choice(name_pool)
    full_name = f"{name_prefix} {base_name}"
    mat = random.choice(MATERIALS)
    occ = rand_occasions(random.randint(2, 5))
    c1, c2 = rand_color_pair()
    style = random.choice(STYLES)
    price = round(random.randint(*price_range) / 50) * 50  # round to ₹50
    form = random.randint(*formality_range)  # 1=very casual ... 5=very formal

    return {
        "item_id": item_id,
        "name": full_name,
        "category": category,
        "subcategory": subcategory,
        "color": c1,
        "secondary_color": c2 if c2 else "",
        "style": style,
        "occasion": occ,
        "season": rand_seasons(),
        "formality": form,
        "pattern": random.choice(PATTERNS),
        "material": mat,
        "price": price,
        "gender": random.choice(gender_pool),
        "comfort_score": comfort(price, mat),
        "versatility_score": versatility(occ, style),
        "popularity_score": round(random.uniform(0.3, 1.0), 2),
        "trend_score": round(random.uniform(0.2, 1.0), 2),
        "image_placeholder": f"assets/{category}/{item_id}.svg"
    }


def generate_catalog():
    items = []
    idx = 1

    # Tops (100)
    for i in range(100):
        sub = random.choice(["t_shirt", "shirt", "blouse", "kurti", "top"])
        gender = random.choice(GENDERS)
        items.append(make_item(idx, "top", sub, TOP_NAMES, [gender],
                               (299, 2499), (1, 4)))
        idx += 1

    # Bottoms (100)
    idx = 1
    for i in range(100):
        sub = random.choice(["jeans", "trousers", "skirt", "shorts", "pants"])
        gender = random.choice(GENDERS)
        items.append(make_item(idx, "bottom", sub, BOTTOM_NAMES, [gender],
                               (399, 2999), (1, 5)))
        idx += 1

    # Dresses (60)
    idx = 1
    for i in range(60):
        sub = random.choice(["midi_dress", "maxi_dress", "mini_dress", "kurta"])
        items.append(make_item(idx, "dress", sub, DRESS_NAMES, ["female", "unisex"],
                               (699, 3999), (2, 5)))
        idx += 1

    # Shoes (60)
    idx = 1
    for i in range(60):
        sub = random.choice(["sneakers", "formal", "sandals", "boots", "heels", "flats"])
        gender = random.choice(GENDERS)
        items.append(make_item(idx, "shoes", sub, SHOE_NAMES, [gender],
                               (499, 3999), (1, 5)))
        idx += 1

    # Accessories (40)
    idx = 1
    for i in range(40):
        sub = random.choice(["jewellery", "bag", "hat", "belt", "scarf"])
        gender = random.choice(GENDERS)
        items.append(make_item(idx, "accessory", sub, ACCESSORY_NAMES, [gender],
                               (99, 1499), (1, 5)))
        idx += 1

    return items


def save_catalog(items, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, "fashion_items.csv")
    fieldnames = [
        "item_id", "name", "category", "subcategory", "color", "secondary_color",
        "style", "occasion", "season", "formality", "pattern", "material",
        "price", "gender", "comfort_score", "versatility_score",
        "popularity_score", "trend_score", "image_placeholder"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = dict(item)
            row["occasion"] = "|".join(item["occasion"])
            row["season"] = "|".join(item["season"])
            writer.writerow(row)

    # JSON (for easy loading)
    json_path = os.path.join(output_dir, "fashion_items.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"[OK] Catalog saved: {len(items)} items")
    print(f"    CSV  -> {csv_path}")
    print(f"    JSON -> {json_path}")

    # Summary by category
    from collections import Counter
    cats = Counter(i["category"] for i in items)
    for cat, cnt in sorted(cats.items()):
        print(f"    {cat:<12} {cnt} items")

    return csv_path, json_path


if __name__ == "__main__":
    items = generate_catalog()
    save_catalog(items, os.path.join(os.path.dirname(__file__)))
