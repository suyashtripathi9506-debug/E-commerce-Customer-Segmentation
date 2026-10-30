"""Step 2 - RFM (Recency, Frequency, Monetary) analysis."""
import pandas as pd


def clean_transactions(df):
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.dropna(subset=["CustomerID"])
    df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]
    df["TotalPrice"] = df.Quantity * df.UnitPrice
    return df


def compute_rfm(df):
    """Return one row per customer with Recency (days), Frequency (orders), Monetary (spend)."""
    snapshot = df.InvoiceDate.max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    return rfm


def score_rfm(rfm, q=5):
    """Score each metric 1-5 (5 = best) using quantiles."""
    rfm = rfm.copy()
    rfm["R_Score"] = pd.qcut(rfm.Recency, q, labels=range(q, 0, -1), duplicates="drop").astype(int)
    rfm["F_Score"] = pd.qcut(rfm.Frequency.rank(method="first"), q, labels=range(1, q + 1)).astype(int)
    rfm["M_Score"] = pd.qcut(rfm.Monetary.rank(method="first"), q, labels=range(1, q + 1)).astype(int)
    rfm["RFM_Score"] = rfm.R_Score + rfm.F_Score + rfm.M_Score
    return rfm


def label_segment(row):
    r, f = row.R_Score, row.F_Score
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal Customers"
    if r >= 4 and f <= 2:
        return "New / Promising"
    if r <= 2 and f >= 4:
        return "At Risk"
    if r <= 2 and f >= 2:
        return "About to Sleep"
    if r <= 2:
        return "Lost"
    return "Needs Attention"


def rfm_pipeline(df):
    rfm = score_rfm(compute_rfm(df))
    rfm["RFM_Segment"] = rfm.apply(label_segment, axis=1)
    return rfm
