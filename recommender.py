"""Step 5 - Recommendation system.

Two strategies:
  1. Item-based collaborative filtering (cosine similarity on the customer x product matrix)
  2. Segment-based popularity (cold-start / fallback): best-sellers inside the customer's segment
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:
    def __init__(self, transactions, customer_segments):
        """transactions: cleaned df; customer_segments: Series indexed by CustomerID."""
        self.tx = transactions
        self.segments = customer_segments
        self.products = (transactions.drop_duplicates("StockCode")
                         .set_index("StockCode")[["Description", "Category", "UnitPrice"]])

        # implicit-feedback matrix: number of purchases
        self.matrix = (transactions.pivot_table(index="CustomerID", columns="StockCode",
                                                values="Quantity", aggfunc="sum", fill_value=0))
        self.item_sim = pd.DataFrame(cosine_similarity(self.matrix.T),
                                     index=self.matrix.columns, columns=self.matrix.columns)

    # ---- collaborative filtering ----
    def recommend_cf(self, customer_id, n=5):
        if customer_id not in self.matrix.index:
            return self.recommend_popular(None, n)
        bought = self.matrix.loc[customer_id]
        owned = bought[bought > 0].index
        scores = self.item_sim[owned].mul(bought[owned], axis=1).sum(axis=1)
        scores = scores.drop(owned).sort_values(ascending=False).head(n)
        return self._format(scores, "CF score")

    # ---- segment popularity ----
    def recommend_popular(self, segment=None, n=5):
        tx = self.tx
        if segment is not None:
            ids = self.segments[self.segments == segment].index
            tx = tx[tx.CustomerID.isin(ids)]
        pop = tx.groupby("StockCode").Quantity.sum().sort_values(ascending=False).head(n)
        return self._format(pop, "Units sold")

    # ---- hybrid ----
    def recommend(self, customer_id, n=5):
        """CF first; fill remaining slots with segment best-sellers."""
        recs = self.recommend_cf(customer_id, n)
        if len(recs) < n:
            seg = self.segments.get(customer_id)
            extra = self.recommend_popular(seg, n * 2)
            extra = extra[~extra.index.isin(recs.index)].head(n - len(recs))
            recs = pd.concat([recs, extra])
        return recs

    def _format(self, scores, col):
        out = self.products.loc[scores.index].copy()
        out[col] = scores.round(3).values
        return out

    # ---- evaluation: hold-out hit rate ----
    def evaluate_hit_rate(self, n=10, sample=300, seed=42):
        """Hide one purchased product per customer and check if it appears in top-n."""
        rng = np.random.default_rng(seed)
        ids = [c for c in self.matrix.index if (self.matrix.loc[c] > 0).sum() >= 3]
        ids = rng.choice(ids, size=min(sample, len(ids)), replace=False)
        hits = 0
        for c in ids:
            row = self.matrix.loc[c].copy()
            owned = row[row > 0].index
            hidden = rng.choice(owned)
            row[hidden] = 0
            owned_left = row[row > 0].index
            scores = self.item_sim[owned_left].mul(row[owned_left], axis=1).sum(axis=1)
            top = scores.drop(owned_left).sort_values(ascending=False).head(n).index
            hits += hidden in top
        return hits / len(ids)
