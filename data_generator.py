"""Step 1 - Generate a synthetic e-commerce transaction dataset.

Replace this with your own CSV (e.g. the UCI Online Retail dataset) if you have one.
Required columns: InvoiceNo, CustomerID, InvoiceDate, StockCode, Description,
Category, Quantity, UnitPrice
"""
import os
import numpy as np
import pandas as pd

CATEGORIES = {
    "Electronics": (40, 400),
    "Fashion": (15, 120),
    "Home & Kitchen": (10, 150),
    "Beauty": (8, 60),
    "Sports": (12, 200),
    "Books": (5, 35),
}


def build_products(n_per_cat=25, rng=None):
    rng = rng or np.random.default_rng(42)
    rows = []
    pid = 1000
    for cat, (lo, hi) in CATEGORIES.items():
        for i in range(n_per_cat):
            rows.append({
                "StockCode": f"P{pid}",
                "Description": f"{cat} Item {i + 1}",
                "Category": cat,
                "UnitPrice": round(rng.uniform(lo, hi), 2),
            })
            pid += 1
    return pd.DataFrame(rows)


def generate_transactions(n_customers=1500, seed=42):
    rng = np.random.default_rng(seed)
    products = build_products(rng=rng)
    end_date = pd.Timestamp("2026-10-01")
    start_date = end_date - pd.Timedelta(days=365)

    # Latent personas: (share, orders mean, recency bias, basket mean, favourite categories)
    personas = [
        ("champion",  0.15, 18, 0.9, 4, ["Electronics", "Fashion", "Home & Kitchen"]),
        ("regular",   0.30, 8,  0.6, 3, ["Home & Kitchen", "Beauty", "Books"]),
        ("occasional", 0.30, 3, 0.4, 2, ["Fashion", "Sports", "Books"]),
        ("lapsed",    0.25, 2,  0.05, 2, ["Beauty", "Sports", "Electronics"]),
    ]

    records, inv = [], 100000
    cust_id = 10000
    for name, share, n_orders, recent, basket, fav in personas:
        for _ in range(int(n_customers * share)):
            cust_id += 1
            k = max(1, rng.poisson(n_orders))
            for _ in range(k):
                # recency bias: higher value -> dates skewed toward the end of the year
                frac = rng.beta(1 + 6 * recent, 1 + 6 * (1 - recent))
                date = start_date + pd.Timedelta(days=int(frac * 364),
                                                 hours=int(rng.integers(8, 22)))
                inv += 1
                n_items = max(1, rng.poisson(basket))
                pool = products[products.Category.isin(fav)] if rng.random() < 0.75 else products
                for _, p in pool.sample(n_items, replace=True,
                                        random_state=int(rng.integers(1e9))).iterrows():
                    records.append({
                        "InvoiceNo": inv, "CustomerID": cust_id, "InvoiceDate": date,
                        "StockCode": p.StockCode, "Description": p.Description,
                        "Category": p.Category, "Quantity": int(rng.integers(1, 4)),
                        "UnitPrice": p.UnitPrice,
                    })
    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_transactions()
    df.to_csv("data/transactions.csv", index=False)
    print(f"Saved data/transactions.csv  rows={len(df):,}  customers={df.CustomerID.nunique():,}")