# features/build_features.py
import argparse, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_DEFAULT = ROOT / "db" / "retailrocket_mini.db"

def _latest_item_attributes(conn: sqlite3.Connection) -> pd.DataFrame:
    ip = pd.read_sql("SELECT timestamp, itemid, property, value FROM item_properties", conn)
    if ip.empty:
        return pd.DataFrame({"itemid": []})
    ip["timestamp"] = pd.to_datetime(ip["timestamp"], unit="ms", errors="coerce")
    ip = ip.dropna(subset=["timestamp"])
    last_vals = (ip.sort_values("timestamp")
                   .groupby(["itemid","property"], as_index=False)
                   .last()[["itemid","property","value"]])
    wide = last_vals.pivot(index="itemid", columns="property", values="value").reset_index()
    wide.columns.name = None
    for c in ["price", "available"]:
        if c in wide.columns:
            wide[c] = pd.to_numeric(wide[c], errors="coerce")
    return wide

def build_features(db_path: Path = DB_DEFAULT, inactivity_days: int = 60) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    events = pd.read_sql("SELECT timestamp, visitorid, event, itemid, transactionid FROM events", conn)
    conn.close()
    if events.empty:
        return pd.DataFrame()

    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms", errors="coerce")
    events = events.dropna(subset=["timestamp"]).sort_values("timestamp")
    events["visitorid"] = events["visitorid"].astype(str)

    # join latest item attributes for spend/category proxies
    conn = sqlite3.connect(db_path)
    item_attrs = _latest_item_attributes(conn)
    conn.close()
    if not item_attrs.empty:
        keep = ["itemid"] + [c for c in ["categoryid","brand","price","available"] if c in item_attrs.columns]
        events = events.merge(item_attrs[keep], on="itemid", how="left")

    ref = events["timestamp"].max()
    g = events.groupby("visitorid", observed=True)

    views      = g.apply(lambda x: (x["event"]=="view").sum())
    carts      = g.apply(lambda x: (x["event"]=="addtocart").sum())
    purchases  = g.apply(lambda x: (x["event"]=="transaction").sum())
    total_evts = views + carts + purchases

    first_ts   = g["timestamp"].min()
    last_ts    = g["timestamp"].max()
    active_days= g["timestamp"].agg(lambda s: s.dt.normalize().nunique())
    orders     = g["transactionid"].nunique() if "transactionid" in events.columns else purchases

    if "price" in events.columns:
        spend = g.apply(lambda x: float(x.loc[x["event"]=="transaction","price"].fillna(0).sum()))
    else:
        spend = pd.Series(0.0, index=first_ts.index)

    if "categoryid" in events.columns:
        distinct_cats = g["categoryid"].nunique()
    else:
        distinct_cats = pd.Series(0, index=first_ts.index)

    def _avg_gap_days(gr: pd.DataFrame) -> float:
        t = gr["timestamp"].sort_values().values
        if len(t) < 2: return np.nan
        d = np.diff(t).astype("timedelta64[D]").astype(int)
        return float(np.mean(d)) if len(d) else np.nan

    avg_gap = g.apply(_avg_gap_days)

    feat = pd.DataFrame({
        "visitorid": first_ts.index.astype(str),
        "first_event_ts": first_ts.values,
        "last_event_ts":  last_ts.values,
        "active_days":    active_days.values,
        "days_since_last_event": (ref - last_ts).dt.days.values,
        "views": views.values,
        "carts": carts.values,
        "purchases": purchases.values,
        "total_events": total_evts.values,
        "orders": orders.values,
        "total_spend": spend.values,
        "distinct_categories_viewed": distinct_cats.values,
        "avg_gap_days": avg_gap.values
    }).fillna(0)

    feat["cart_rate"]       = feat["carts"]     / feat["views"].replace(0,1)
    feat["purchase_rate"]   = feat["purchases"] / feat["views"].replace(0,1)
    feat["avg_order_value"] = feat["total_spend"]/ feat["orders"].replace(0,1)
    feat["churn_flag"]      = (feat["days_since_last_event"] >= inactivity_days).astype(int)
    feat["cohort_month"]    = pd.to_datetime(feat["first_event_ts"]).dt.to_period("M").astype(str)

    # -------------------------------
    # Extra engineered churn features
    # -------------------------------
    cut_30d = ref - pd.Timedelta(days=30)
    cut_7d  = ref - pd.Timedelta(days=7)
    cut_prev30d = ref - pd.Timedelta(days=60)

    # Last 30 days
    last30 = events[events["timestamp"] >= cut_30d]
    views30 = last30[last30["event"]=="view"].groupby("visitorid").size().rename("views_last30d")
    purch30 = last30[last30["event"]=="transaction"].groupby("visitorid").size().rename("purchases_last30d")

    # Last 7 days
    last7 = events[events["timestamp"] >= cut_7d]
    views7 = last7[last7["event"]=="view"].groupby("visitorid").size().rename("views_last7d")

    # Trend feature (last30 vs prev30)
    prev30 = events[(events["timestamp"] >= cut_prev30d) & (events["timestamp"] < cut_30d)]
    act_prev30 = prev30.groupby("visitorid").size().rename("act_prev30d")
    act_last30 = last30.groupby("visitorid").size().rename("act_last30d")
    activity_change = (act_last30 / (act_prev30 + 1)).rename("activity_change")

    # Join engineered features (index-based join to avoid error)
    feat = feat.set_index("visitorid")
    feat = feat.join([views30, purch30, views7, activity_change])
    feat = feat.reset_index()

    # Fill NA
    for col in ["views_last30d","purchases_last30d","views_last7d","activity_change"]:
        if col not in feat.columns:
            feat[col] = 0
        else:
            feat[col] = feat[col].fillna(0)

    return feat

def materialize_features(db_path: Path = DB_DEFAULT, inactivity_days: int = 60) -> Path:
    feat = build_features(db_path, inactivity_days)
    if feat.empty:
        raise RuntimeError("No events to build features from.")
    conn = sqlite3.connect(db_path)
    feat.to_sql("customer_features", conn, if_exists="replace", index=False)
    conn.commit(); conn.close()
    pos = int(feat["churn_flag"].sum()); n = len(feat)
    print(f"✅ Wrote customer_features into {db_path} | positives={pos}/{n} ({pos/n:.1%})")
    return db_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=str(DB_DEFAULT))
    parser.add_argument("--inactivity", type=int, default=60)
    args = parser.parse_args()
    materialize_features(Path(args.db), args.inactivity)

# ----------------------------------------------------
# Extra utilities for churn segmentation & risk scoring
# ----------------------------------------------------
import hashlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def kmeans_churn_labels(feats: pd.DataFrame) -> pd.DataFrame:
    feats = feats.copy()
    if feats.empty:
        feats["cluster"] = pd.Series(dtype="int64")
        feats["churn_label"] = pd.Series(dtype="string")
        feats["is_churn_cluster"] = pd.Series(dtype="bool")
        return feats
    X = feats[["days_since_last_event"]].to_numpy(dtype=float)
    uniq, n = len(np.unique(X)), len(feats)
    k = max(1, min(3, uniq, n))
    if k == 1:
        feats["cluster"] = 0
        feats["churn_label"] = "🔥 Slipping Away"
        feats["is_churn_cluster"] = feats["days_since_last_event"].eq(
            feats["days_since_last_event"].max()
        )
        feats.loc[feats["is_churn_cluster"], "churn_label"] = "💀 Ghosted (Churned)"
        return feats
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    feats["cluster"] = km.fit_predict(Xs)
    means = feats.groupby("cluster")["days_since_last_event"].mean().sort_values(ascending=False)
    ranked = means.index.tolist()
    labels = {}
    if len(ranked) >= 1: labels[ranked[0]] = "💀 Ghosted (Churned)"
    if len(ranked) >= 2: labels[ranked[-1]] = "🌈 Happy Camper"
    for c in set(feats["cluster"].unique()) - set(labels.keys()):
        labels[c] = "🔥 Slipping Away"
    feats["churn_label"] = feats["cluster"].map(labels)
    feats["is_churn_cluster"] = feats["cluster"].eq(ranked[0])
    return feats

def retention_flag(last_event_ts: pd.Series, ref_end: pd.Timestamp) -> pd.Series:
    same_day = last_event_ts.dt.normalize().eq(ref_end.normalize())
    return np.where(same_day, "✅ Retained", "❌ Not Retained")

def behavior_segment(row, ref_end: pd.Timestamp,
                     T_NEW=7, T_RECENT_JOIN=14, T_INACTIVE=30,
                     T_RECENT_PURCHASE=30, T_BROWSER_VIEWS=3,
                     T_REPEAT_BUYER=3) -> str:
    dsl = int(row.get("days_since_last_event", 10**6))
    views = int(row.get("views", 0))
    carts = int(row.get("carts", 0))
    purch = int(row.get("purchases", 0))
    first_ts = row.get("first_event_ts")

    age_days = None
    if pd.notna(first_ts):
        age_days = (ref_end.normalize() - pd.to_datetime(first_ts).normalize()).days

    if age_days is not None and age_days <= T_NEW and views <= 2 and carts == 0 and purch == 0:
        return "🆕 New"
    if (age_days is not None and age_days <= T_RECENT_JOIN) and dsl <= 14:
        return "🟢 Recent Visitor"
    if purch >= T_REPEAT_BUYER:
        return "🛍️ Repeat Buyer"
    if purch >= 1 and dsl <= T_RECENT_PURCHASE:
        return "🛍️ Active Buyer"
    if carts >= 1 and purch == 0 and dsl <= 30:
        return "🛒 Cart Abandoner"
    if views >= T_BROWSER_VIEWS and carts == 0 and purch == 0 and dsl <= 30:
        return "👀 Window Shopper"
    if dsl >= T_INACTIVE and purch == 0:
        return "😴 Inactive"
    if purch == 1 and dsl >= T_INACTIVE:
        return "🧊 One-and-Done"
    return "🙂 Other"

def cap_new_to_twenty_percent(feats: pd.DataFrame) -> pd.DataFrame:
    if feats.empty or "segment" not in feats.columns: 
        return feats
    m = feats["segment"] == "🆕 New"
    total = len(feats); new_n = int(m.sum())
    if total == 0 or new_n / total <= 0.20: 
        return feats
    target = int(np.floor(0.20 * total)); excess = new_n - target
    cand = feats.loc[m, ["first_event_ts"]].sort_values("first_event_ts")
    feats.loc[cand.index[:excess], "segment"] = "🟢 Recent Visitor"
    return feats

def cap_risk_by_segment(seg: str, risk_cat: str) -> str:
    never_high = {"🆕 New","🟢 Recent Visitor","🛍️ Active Buyer","🛍️ Repeat Buyer"}
    if seg in never_high and risk_cat == "🔴 High":
        return "🟠 Medium"
    return risk_cat

def _variant(seed_str: str, k: int) -> int:
    h = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % k

def personalized_reco(row) -> str:
    risk = row.get("risk_category", "🟢 Low")
    seg  = row.get("segment", "🙂 Other")
    inact = int(row.get("days_since_last_event", 0))
    views = int(row.get("views", 0)); carts = int(row.get("carts", 0)); purch = int(row.get("purchases", 0))
    vid = str(row.get("visitorid", ""))

    disc = 0
    if risk == "🔴 High": disc = 20 if inact >= 45 else 15
    elif risk == "🟠 Medium": disc = 12 if inact >= 30 else 10
    else: disc = 5 if inact >= 14 else 0

    v = _variant(vid, 3)
    if risk == "🔴 High":
        bases = [f"Win-back: {disc}% + free shipping.", f"Last chance: {disc}% off favorites.", f"{disc}% comeback coupon waiting."]
    elif risk == "🟠 Medium":
        bases = [f"Try what you viewed: {disc}% off.", f"Nudge: {disc}% on picks you liked.", f"Trending now—take {disc}% off."]
    else:
        bases = ["New arrivals we think you’ll love.", "Member perk: early access & bundles.", "Weekly picks based on your browsing."]
        if disc > 0: bases = [f"Welcome back: extra {disc}%.", f"Treat yourself: {disc}% on bestsellers.", f"{disc}% off—limited time."]
    cta = ""
    if carts > 0 and purch == 0: cta = " Recover your cart in one click."
    elif views >= 3 and carts == 0 and purch == 0: cta = " Add to cart now to lock the price."
    elif purch > 0: cta = " Because you liked your last purchase, here’s what pairs well."

    sfx_opts = {
        "🆕 New": [" Welcome gift inside.", " Kickstart with a first-time perk.", " Start with our top-rated picks."],
        "🟢 Recent Visitor": [" Still curious? See what’s trending.", " Your recent views are back in stock.", " Items you checked are on promo."],
        "😴 Inactive": [" We miss you—exclusive comeback deal.", " A surprise is waiting in your account.", " Let’s pick up where you left off."],
        "🛒 Cart Abandoner": [" Complete checkout in 2 steps.", " Your cart is waiting.", " Items in your cart are limited."],
        "👀 Window Shopper": [" See price drops on watched items.", " Explore bundles that save more.", " Join members grabbing these deals."],
        "🙂 Other": [" Explore editor’s picks.", " This week’s bestsellers for you.", " Community favorites you’ll love."]
    }
    sfx = sfx_opts.get(seg, sfx_opts["🙂 Other"])[_variant(vid+"sfx", 3)]
    return bases[v] + cta + " " + sfx

def assign_risk_buckets_with_targets(df, score_col="risk_score",
                                     target_high=0.25, target_low=0.25):
    """
    1) Rank by risk_score descending
    2) Assign target shares (e.g., High 25%, Low 25%, rest Medium)
    3) Return a 'risk_category' column: 🔴 High / 🟠 Medium / 🟢 Low
    """
    if df.empty or score_col not in df.columns:
        df["risk_category"] = pd.Series(dtype="string")
        return df

    n = len(df)
    n_high = int(round(target_high * n))
    n_low  = int(round(target_low  * n))
    n_high = max(0, min(n_high, n))
    n_low  = max(0, min(n_low,  n - n_high))

    tmp = df[[score_col]].copy()
    tmp["_rank"] = tmp[score_col].rank(method="first", ascending=False)

    cats = pd.Series("🟠 Medium", index=tmp.index, dtype="string")
    cats.loc[tmp["_rank"] <= n_high] = "🔴 High"
    cats.loc[tmp["_rank"] >  n - n_low] = "🟢 Low"

    out = df.copy()
    out["risk_category"] = cats
    return out


def rebalance_after_capping(df, score_col="risk_score", target_high=0.25):
    """After capping segments, ensure we still hit the High % target."""
    n = len(df); want_high = int(round(target_high * n))
    mask_high = df["risk_category"] == "🔴 High"
    have_high = int(mask_high.sum())
    if have_high >= want_high:
        return df
    # promote highest-scoring Mediums
    need = want_high - have_high
    idx = (df[df["risk_category"] == "🟠 Medium"]
             .sort_values(by=score_col, ascending=False)
             .head(need).index)
    out = df.copy()
    out.loc[idx, "risk_category"] = "🔴 High"
    return out


def compute_risk(feats: pd.DataFrame, target_high: float = 0.25, target_low: float = 0.25) -> pd.DataFrame:
    """
    Compute a heuristic risk_score and enforce final risk buckets to fixed shares:
    High ~= target_high, Low ~= target_low, Medium = remainder.
    """
    if feats.empty:
        feats["risk_score"] = pd.Series(dtype="float")
        feats["risk_category"] = pd.Series(dtype="string")
        return feats

    df = feats.copy()

    # --- Base risk score (heuristic) ---
    max_inact = max(1, float(df["days_since_last_event"].max()))
    norm_inact = (df["days_since_last_event"] / max_inact).clip(0, 1)

    purch = df["purchases"].astype(float)
    purchase_factor = 1 - (purch / 3.0).clip(0, 1)

    cart_abandon = ((df["carts"] > 0) & (df["purchases"] == 0)).astype(float)
    window_shopper = ((df["views"] >= 3) & (df["carts"] == 0) & (df["purchases"] == 0)).astype(float)

    df["risk_score"] = (
        0.60 * norm_inact +
        0.25 * purchase_factor +
        0.15 * cart_abandon +
        0.10 * window_shopper
    ).clip(0, 1)

    # --- Initial assignment with target shares (optional pre-step) ---
    df = assign_risk_buckets_with_targets(
        df, score_col="risk_score",
        target_high=target_high, target_low=target_low
    )

    # --- Apply segment-based capping (keeps your policy) ---
    df["risk_category"] = df.apply(
        lambda r: cap_risk_by_segment(r.get("segment", "🙂 Other"), r["risk_category"]),
        axis=1
    )

    # --- FINAL ENFORCEMENT: force exact target shares regardless of caps ---
    n = len(df)
    n_high = int(round(target_high * n))
    n_low  = int(round(target_low  * n))

    # sort by score desc
    sorted_idx = df.sort_values("risk_score", ascending=False).index

    # reset to Medium, then set top/bottom slices
    df["risk_category"] = "🟠 Medium"
    if n_high > 0:
        df.loc[sorted_idx[:n_high], "risk_category"] = "🔴 High"
    if n_low > 0:
        df.loc[sorted_idx[-n_low:], "risk_category"] = "🟢 Low"

    return df
