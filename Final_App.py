# RetailRocket Churn Dashboard — v3 layout (refined segmentation + risk capping + improved overview)
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from features.build_features import (
    kmeans_churn_labels,
    retention_flag,
    behavior_segment,
    cap_new_to_twenty_percent,
    cap_risk_by_segment,
    personalized_reco,
    compute_risk,
)

import io, re, hashlib, sqlite3
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- App setup ----------
ROOT = Path(__file__).resolve().parents[1]
DB_MINI = ROOT / "db" / "retailrocket_mini.db"
ROW_CAP = 200_000

st.set_page_config(page_title="RetailRocket Churn Dashboard", page_icon="🛒", layout="wide")
st.title("🛒 RetailRocket E-Commerce Customer Churn & Retention Dashboard")

# ---------- Data helpers ----------
@st.cache_data(show_spinner=False)
def load_events_from_db(db_path: str, limit: int = ROW_CAP) -> pd.DataFrame:
    p = Path(db_path)
    if not p.exists():
        return pd.DataFrame(columns=["timestamp","event","itemid","visitorid","transactionid"])
    with sqlite3.connect(p) as conn:
        ev = pd.read_sql(
            "SELECT timestamp, event, itemid, visitorid, transactionid FROM events "
            f"LIMIT {int(limit)}", conn
        )
    return _standardize_events(ev)

def _standardize_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp","event","itemid","visitorid","transactionid"])
    df = df.copy()
    ts_col = df["timestamp"]
    if is_numeric_dtype(ts_col):
        ts = pd.to_datetime(ts_col, unit="ms", errors="coerce", utc=True)
    else:
        ts = pd.to_datetime(ts_col, errors="coerce", utc=True)
    valid = ts.notna()
    df = df.loc[valid].copy()
    df["timestamp"] = ts.loc[valid].dt.tz_localize(None)
    if "event" in df.columns:
        df["event"] = df["event"].astype(str).str.strip().str.lower()
    if "visitorid" in df.columns:
        df["visitorid"] = df["visitorid"].astype(str).str.strip().str.lower()
    for c in ["itemid","transactionid"]:
        if c in df.columns: df[c] = df[c].astype(str)
    keep = [c for c in ["timestamp","event","itemid","visitorid","transactionid"] if c in df.columns]
    return df[keep].sort_values("timestamp").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_events_from_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".parquet") or name.endswith(".pq"):
        ev = pd.read_parquet(uploaded_file)
        if len(ev) > ROW_CAP: ev = ev.head(ROW_CAP)
    else:
        content = uploaded_file.read()
        ev = pd.read_csv(io.BytesIO(content), nrows=ROW_CAP)
    return _standardize_events(ev)

def build_counts(ev: pd.DataFrame) -> pd.DataFrame:
    if ev.empty:
        return pd.DataFrame(columns=["views","carts","purchases"])
    counts = ev.groupby(["visitorid","event"]).size().unstack(fill_value=0)
    for col in ["view","addtocart","transaction"]:
        if col not in counts.columns: counts[col] = 0
    return counts.rename(columns={"view":"views","addtocart":"carts","transaction":"purchases"})

def build_customer_features(ev_window: pd.DataFrame, ref_end: pd.Timestamp, user_selected: bool=False) -> pd.DataFrame:
    """
    Build customer-level features from events in the current window.
    - ev_window: events filtered between start_date and dataset_end
    - ref_end: dataset end date (fixed)
    - user_selected: False = first dashboard load (default start_date)
                     True = user has changed the start_date
    """
    if ev_window.empty:
        return pd.DataFrame(columns=[
            "visitorid","first_event_ts","last_event_ts",
            "days_since_last_event","views","carts","purchases"
        ])

    counts = build_counts(ev_window)
    first_ts = ev_window.groupby("visitorid")["timestamp"].min().rename("first_event_ts")
    last_ts  = ev_window.groupby("visitorid")["timestamp"].max().rename("last_event_ts")
    df = counts.join(first_ts).join(last_ts).reset_index()

    # 🔑 Inactivity calculation
    if not user_selected:
        # Default load → dataset_end − last_event
        df["days_since_last_event"] = (
            ref_end.normalize() - df["last_event_ts"].dt.normalize()
        ).dt.days.clip(lower=0)
    else:
        # User changed start_date → recompute inactivity relative to dataset end
        # still dataset_end − last_event (ensures silence is counted)
        df["days_since_last_event"] = (
            ref_end.normalize() - df["last_event_ts"].dt.normalize()
        ).dt.days.clip(lower=0)

    return df
# --- ML risk helper (silently uses trained model if available) ---
def _try_apply_trained_model(feats: pd.DataFrame) -> pd.DataFrame:
    """If models/risk_model.pkl exists, use it to compute risk_score & risk_category.
       Otherwise return feats unchanged."""
    try:
        from joblib import load
        from pathlib import Path
        MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "risk_model.pkl"
        if not MODEL_PATH.exists():
            return feats
        payload = load(MODEL_PATH)
        est = payload["pipeline"]; Xcols = payload["Xcols"]
        for c in Xcols:
            if c not in feats.columns: feats[c] = 0.0
        X = feats[Xcols].replace([np.inf, -np.inf], np.nan).fillna(0)
        proba = est.predict_proba(X)[:, 1]
        feats = feats.copy()
        feats["risk_score"] = proba
        # bucket using same quantiles as heuristic for consistency
        qs_high = feats["risk_score"].quantile(0.85)
        qs_med  = feats["risk_score"].quantile(0.60)
        feats["risk_category"] = feats["risk_score"].apply(
            lambda x: "🔴 High" if x >= qs_high else ("🟠 Medium" if x >= qs_med else "🟢 Low")
        )
        return feats
    except Exception:
        return feats



# ---------- Cached feature pipeline ----------
@st.cache_data(show_spinner=False)
def build_feats(ev_window: pd.DataFrame, ref_end: pd.Timestamp, user_selected: bool=False) -> pd.DataFrame:
    # 1) Build customer-level features
    feats = build_customer_features(ev_window, ref_end, user_selected=user_selected)
    feats = kmeans_churn_labels(feats)

    # 2) Retention + behavior segments
    if feats.empty:
        feats["retention"] = pd.Series(dtype="string")
        feats["segment"]   = pd.Series(dtype="string")
    else:
        feats["retention"] = retention_flag(feats["last_event_ts"], ref_end=ref_end)
        feats["segment"]   = feats.apply(lambda r: behavior_segment(r, ref_end), axis=1)
        feats = cap_new_to_twenty_percent(feats)

    # 3) Risk scoring — try ML first, fallback to heuristic
    if not feats.empty:
        feats_ml = _try_apply_trained_model(feats.copy())
        if "risk_score" in feats_ml.columns and "risk_category" in feats_ml.columns:
            feats = feats_ml
        else:
            feats = compute_risk(feats)
    
    feats["recommendation"] = feats.apply(personalized_reco, axis=1)

    return feats

# ---------- Visual helpers ----------
def re_split_multi(s: str):
    return [p for p in re.split(r"[,\s;]+", s) if p]

# ---------- Sidebar ----------
st.sidebar.markdown("## ⚙️ Data & Time Window")
uploaded = st.sidebar.file_uploader("Upload events (.csv or .parquet)", type=["csv","parquet","pq"])
events_db = load_events_from_db(str(DB_MINI), limit=ROW_CAP)
events_all = load_events_from_upload(uploaded) if uploaded is not None else events_db
if events_all.empty:
    st.error("No events found (DB or upload)."); st.stop()

# Dataset start and end
ev_min_ts = events_all["timestamp"].min()
ev_max_ts = events_all["timestamp"].max()
ref_end = ev_max_ts.normalize()          # ✅ always dataset end
fixed_end_date = ref_end.date()
end_ts_exclusive = pd.to_datetime(ref_end) + pd.Timedelta(days=1)

min_start_date = ev_min_ts.normalize().date()
max_start_date = ev_max_ts.normalize().date()
st.sidebar.write("Dataset start:", ev_min_ts)
st.sidebar.write("Dataset end:", ev_max_ts)


# Detect first load vs user change
if "start_date" not in st.session_state:
    # First load → default = full dataset start
    st.session_state.start_date = min_start_date
    user_selected = False
else:
    user_selected = True

st.sidebar.markdown("### 📅 Select Start Date")
start_date = st.sidebar.date_input(
    "Start (end fixed to dataset end)",
    value=st.session_state.start_date,
    min_value=min_start_date,
    max_value=max_start_date
)
# Update session state
if start_date != st.session_state.start_date:
    user_selected = True
st.session_state.start_date = start_date

# Filter events into current window
start_ts = pd.to_datetime(start_date)
mask = (events_all["timestamp"] >= start_ts) & (events_all["timestamp"] < end_ts_exclusive)
ev_window = events_all.loc[mask].copy()

# ✅ Always build features (no 30-day restriction)
feats = build_feats(ev_window, ref_end, user_selected=user_selected)

# Sidebar status
max_inactivity_days = int(feats["days_since_last_event"].max()) if not feats.empty else 0
st.sidebar.caption(f"**🕒 Max inactivity (vs dataset end): {max_inactivity_days} days**")


# Sidebar utilities
# Sidebar buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Reset", use_container_width=True):
        st.experimental_rerun()
with col2:
    if st.button("🧹Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")


with st.sidebar.expander("📘 Glossary", expanded=False):
    st.markdown("""
    - **Churn:** When a customer stops interacting (buying/viewing) for a long time.  
    - **Retention:** A customer who continues to interact within the dataset window.  
    - **PAC:** Prediction Accuracy of Churn (churn detection precision).  
    - **RAC:** Retention Accuracy of Churn (retention classification precision).  
    - **AOV:** Average Order Value.  
    - **Uplift:** The improvement in retention due to an intervention (campaign/offer).  
    - **Inactivity:** Days since last event (view/cart/purchase).  
    - **Segments:** Groups of customers by behavior (e.g., 🛒 Cart Abandoner, 👀 Window Shopper).  
    - **Risk Category:** Churn likelihood grouping → 🔴 High, 🟠 Medium, 🟢 Low.  
    - **Immediate Action Customers:** High-risk ❌ not retained customers who need urgent campaigns.  
    """
    )

# ---------- KPIs (define BEFORE tabs) ----------
users_n = int(feats["visitorid"].nunique()) if not feats.empty else 0

avg_views = float(feats["views"].mean()) if ("views" in feats.columns and not feats.empty) else 0.0
avg_purch = float(feats["purchases"].mean()) if ("purchases" in feats.columns and not feats.empty) else 0.0

avg_inact = float(feats["days_since_last_event"].mean()) if ("days_since_last_event" in feats.columns and not feats.empty) else 0.0
max_inact = int(feats["days_since_last_event"].max()) if ("days_since_last_event" in feats.columns and not feats.empty) else 0

# churn_rate uses the KMeans “is_churn_cluster” flag if present
if not feats.empty and "is_churn_cluster" in feats.columns:
    churn_rate = float(feats["is_churn_cluster"].mean())
else:
    churn_rate = 0.0

# ---------- Tabs ----------
tab_overview, tab_churn, tab_segments, tab_opt, tab_about = st.tabs(
    ["Overview", "Churn Analysis", "Clusters & Retention", "Optimizer", "About"]
)

# ===== 1) OVERVIEW: KPIs + Risk/Events charts + High-risk customers =====
with tab_overview:
    st.subheader("📊 Overview")

    # KPI styling with distinct colors
    st.markdown("""
        <style>
        .kpi {padding:14px;border-radius:12px;text-align:center;color:white;font-weight:600}
        .kpi .v {font-size:22px;font-weight:700}
        .blue {background:#3498db}
        .red {background:#e74c3c}
        .green {background:#2ecc71}
        .orange {background:#f39c12}
        .purple {background:#9b59b6}
        .gray {background:#7f8c8d}
        </style>
    """, unsafe_allow_html=True)

    # KPI row
    a,b,c,d,e,f = st.columns(6)
    a.markdown(f"<div class='kpi blue'>👥 Users<div class='v'>{users_n:,}</div></div>", unsafe_allow_html=True)
    b.markdown(f"<div class='kpi red'>📉 Churn rate<div class='v'>{churn_rate:.1%}</div></div>", unsafe_allow_html=True)
    c.markdown(f"<div class='kpi green'>👀 Avg views<div class='v'>{avg_views:.2f}</div></div>", unsafe_allow_html=True)
    d.markdown(f"<div class='kpi orange'>🛒 Avg purchases<div class='v'>{avg_purch:.2f}</div></div>", unsafe_allow_html=True)
    e.markdown(f"<div class='kpi purple'>🕒 Avg inactivity<div class='v'>{avg_inact:.1f} d</div></div>", unsafe_allow_html=True)
    f.markdown(f"<div class='kpi gray'>⏳ Max inactivity<div class='v'>{max_inact} d</div></div>", unsafe_allow_html=True)

    st.caption(f"Window: **{start_ts.date()} → {ref_end.date()}**")

    # --- Charts row: Risk pie + Events overview ---
    c1, c2 = st.columns(2)

    # Risk composition pie
    if not feats.empty and "risk_category" in feats.columns:
        order = ["🔴 High","🟠 Medium","🟢 Low"]
        risk_counts = (
            feats["risk_category"]
            .value_counts()
            .reindex(order)
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        risk_counts.columns = ["risk_category","count"]

        fig_pie = px.pie(
            risk_counts,
            names="risk_category",
            values="count",
            hole=0.35,
            title="Risk Composition",
            color="risk_category",
            color_discrete_map={"🔴 High":"#e74c3c","🟠 Medium":"#f39c12","🟢 Low":"#2ecc71"}
        )
        fig_pie.update_traces(
            textinfo="percent",
            hovertemplate="%{label}<br>%{value} customers<br>%{percent}"
        )
        fig_pie.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0))
        c1.plotly_chart(fig_pie, use_container_width=True)

    # Events overview bar
    if not ev_window.empty:
        ev_counts = ev_window["event"].value_counts().reset_index()
        ev_counts.columns = ["event","count"]

        fig_ev = px.bar(
            ev_counts,
            x="event",
            y="count",
            text="count",
            title="Events Overview (window)",
            color="event",
            color_discrete_map={"view":"#3498db","addtocart":"#f39c12","transaction":"#2ecc71"}
        )
        fig_ev.update_traces(textposition="outside")
        fig_ev.update_layout(
            height=320,  # increased so labels aren't cut
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
            yaxis=dict(title="count", automargin=True)
        )
        c2.plotly_chart(fig_ev, use_container_width=True)

    # --- High-risk immediate action table ---
    st.markdown("### ⚠️ Top-risk Customers Likely to Churn")
    if not feats.empty:
        df = feats[feats["risk_category"]=="🔴 High"].copy()

        if not df.empty:
            df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
            df.insert(0, "Rank", range(1, len(df)+1))

            # Derive last event
            last_events = ev_window.groupby("visitorid")["event"].last().rename("last_event")
            df = df.merge(last_events, on="visitorid", how="left")

            # Reason for churn risk
            def reason_for_risk(row):
                if row["days_since_last_event"] >= 30: return "Inactive too long ⏳"
                if row["carts"] > 0 and row["purchases"] == 0: return "Cart abandoned 🛒"
                if row["views"] >= 3 and row["purchases"] == 0: return "Browsing but no purchases 👀"
                if row["purchases"] == 1 and row["days_since_last_event"] >= 30: return "One-and-Done 🧊"
                if row["purchases"] == 0: return "Never purchased ❌"
                return "Low engagement ⚠️"

            df["reason_for_risk"] = df.apply(reason_for_risk, axis=1)

            # Filters side-by-side
            SEGMENT_CATEGORIES = ["🆕 New","🟢 Recent Visitor","🛍️ Active Buyer","🛍️ Repeat Buyer",
                                  "🛒 Cart Abandoner","👀 Window Shopper","😴 Inactive","🧊 One-and-Done","🙂 Other"]
            EVENT_CATEGORIES = ["view","addtocart","transaction"]

            f1, f2 = st.columns(2)
            seg_opts = f1.multiselect("Segment", SEGMENT_CATEGORIES, default=None)
            event_opts = f2.multiselect("Last Event", EVENT_CATEGORIES, default=None)

            if seg_opts: df = df[df["segment"].isin(seg_opts)]
            if event_opts: df = df[df["last_event"].isin(event_opts)]

            max_n = len(df)
            if max_n > 0:
                top_n = st.slider("Top-N customers", 5, max_n, min(50, max_n), step=5)

                # Add witty emojis if missing
                df["recommendation"] = df["recommendation"].apply(
                    lambda x: x if any(em in x for em in ["🔥","💡","🎯","💌","⚡","😴","🛒"]) else "💡 " + str(x)
                )

                df_show = df[["Rank","visitorid","days_since_last_event","last_event",
                              "segment","risk_score","risk_category","reason_for_risk","recommendation"]].head(top_n)

                # --- Pagination ---
                page_size = 200
                total_pages = (len(df_show) - 1) // page_size + 1 if len(df_show) > 0 else 1
                page = st.number_input("Page", 1, total_pages, 1, key="hr_page")
                start = (page-1) * page_size
                end = start + page_size
                st.dataframe(df_show.iloc[start:end], use_container_width=True, height=420)

                csv = df_show.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", data=csv,
                                   file_name="high_risk_customers.csv", mime="text/csv")
            else:
                st.warning("No results found for this filter selection.")
        else:
            st.info("No high-risk customers in this window.")
    else:
        st.info("No customers found in the selected window.")

# ===== 2) CHURN ANALYSIS: events + patterns + risk insights =====
with tab_churn:
    st.subheader("📉 Churn Analysis")

    if not feats.empty:
        st.markdown("#### 🔎 Risk & Inactivity Patterns")
        col1, col2 = st.columns(2)

        # --- Graph 1: Scatter of inactivity vs risk ---
        fig_scatter = px.strip(feats, x="risk_category", y="days_since_last_event",
                               color="risk_category",
                               title="Inactivity Days by Risk Category",
                               color_discrete_map={"🔴 High":"#e74c3c",
                                                   "🟠 Medium":"#f39c12",
                                                   "🟢 Low":"#2ecc71"})
        fig_scatter.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        col1.plotly_chart(fig_scatter, use_container_width=True)

        # --- Graph 2: Heatmap risk vs activity (views/carts/purchases) ---
        activity_summary = feats.groupby("risk_category")[["views","carts","purchases"]].mean().reset_index()
        fig_heat = px.imshow(activity_summary.set_index("risk_category"),
                             text_auto=True, aspect="auto",
                             color_continuous_scale="Reds",
                             title="Avg Activity by Risk Category")
        fig_heat.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        col2.plotly_chart(fig_heat, use_container_width=True)

        # --- Graph 3: Bubble chart inactivity vs purchases ---
        st.markdown("#### 🫧 Customer Distribution by Activity")
        fig_bubble = px.scatter(feats, x="days_since_last_event", y="purchases",
                                size="views", color="risk_category",
                                hover_data=["visitorid"],
                                title="Customers: Inactivity vs Purchases (bubble size=views)",
                                color_discrete_map={"🔴 High":"#e74c3c",
                                                    "🟠 Medium":"#f39c12",
                                                    "🟢 Low":"#2ecc71"})
        fig_bubble.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_bubble, use_container_width=True)

    # --- Graph 4: Events over time (moved to bottom) ---
    if not ev_window.empty:
        st.markdown("#### 🕒 Event Trends Over Time")
        daily = (
            ev_window.assign(date=ev_window["timestamp"].dt.date)
            .groupby(["date", "event"]).size().reset_index(name="count")
        )
        fig_ev = px.line(daily, x="date", y="count", color="event",
                         title="Events Over Time")
        fig_ev.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0),
                             legend_title="")
        st.plotly_chart(fig_ev, use_container_width=True)

    # ===== Customer Table Section (now inside tab_churn) =====
    st.markdown("<h3 style='color:#333;'>📊Customer Behavior & Risk Insights Table</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:15px; color:#666;'>Filter, explore, and export churn-related customer data</p>", unsafe_allow_html=True)

    if st.checkbox("📑 Show Customer Table"):
        if not feats.empty:
            # ✅ Derive churn_status column with emojis
            df = feats.copy()
            df["churn_status"] = "🟢 Healthy"
            df.loc[df["risk_category"]=="🟠 Medium", "churn_status"] = "🟠 At risk"
            df.loc[df["risk_category"]=="🔴 High", "churn_status"] = "⚠️ Likely to churn"
            if "is_churn_cluster" in df.columns:
                df.loc[df["is_churn_cluster"], "churn_status"] = "💀 Already churned"

            # --- Filters row 1 ---
            c1, c2, c3 = st.columns([2,1,1])
            id_query = c1.text_area("🔎 Search Visitor IDs (comma / space / newline separated)",
                                    height=60, placeholder="e.g. 1a2b3c, 9f8e7d …")
            churn_opts = c2.multiselect("🧭 Churn Status",
                                        ["💀 Already churned","⚠️ Likely to churn","🟠 At risk","🟢 Healthy"],
                                        default=["💀 Already churned","⚠️ Likely to churn","🟠 At risk","🟢 Healthy"])
            risk_opts = c3.multiselect("🎯 Risk Category",
                                       ["🔴 High","🟠 Medium","🟢 Low"],
                                       default=["🔴 High","🟠 Medium","🟢 Low"])

            # --- Filters row 2 ---
            c4, c5 = st.columns([1,2])
            days_slider = c4.slider("📅 Max inactivity (days)", 
                                    min_value=0, 
                                    max_value=int(df["days_since_last_event"].max()), 
                                    value=(0, int(df["days_since_last_event"].max())))
            seg_opts = c5.multiselect("👥 Segment Filter", sorted(df["segment"].unique()), default=None)

            # Apply filters
            if id_query.strip():
                parts = [p.strip().lower() for p in re_split_multi(id_query)]
                if parts:
                    df = df[df["visitorid"].astype(str).str.lower().isin(parts)]
            if churn_opts:
                df = df[df["churn_status"].isin(churn_opts)]
            if risk_opts:
                df = df[df["risk_category"].isin(risk_opts)]
            if seg_opts:
                df = df[df["segment"].isin(seg_opts)]
            if days_slider:
                min_days, max_days = days_slider
                df = df[(df["days_since_last_event"] >= min_days) & (df["days_since_last_event"] <= max_days)]

            # --- Build final table ---
            desired_cols = ["visitorid","first_event_ts","last_event_ts",
                            "days_since_last_event","views","carts","purchases",
                            "risk_score","risk_category","segment",
                            "churn_status","recommendation"]
            all_cols = [c for c in desired_cols if c in df.columns]
            shown_cols = st.multiselect("📑 Columns to display", all_cols, default=all_cols)

            risk_order = {"💀 Already churned":0,"⚠️ Likely to churn":1,"🟠 At risk":2,"🟢 Healthy":3}
            if not df.empty:
                df = df.assign(_rorder=df["churn_status"].map(risk_order))
                df = df.sort_values(["_rorder","risk_score","days_since_last_event"],
                                    ascending=[True, False, False]).drop(columns=["_rorder"])
            df_show = df.loc[:, shown_cols].reset_index(drop=True)

            show_full = st.checkbox("Show full table (may be slow)", value=False)
            if not show_full:
                df_show = df_show.head(500)

            # --- Pagination ---
            page_size = 200
            total_pages = (len(df_show) - 1) // page_size + 1 if len(df_show) > 0 else 1
            page = st.number_input("Page", 1, total_pages, 1)
            start = (page-1) * page_size
            end = start + page_size
            st.dataframe(df_show.iloc[start:end], use_container_width=True, height=420)

            # --- Export ---
            csv = df_show.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv,
                               file_name="churn_analysis_customers.csv", mime="text/csv")
        else:
            st.info("No customers found in the selected window.")

# ===== 3) RETENTION & EMAIL SUGGESTIONS =====
# ---------- Tab 3 safety: dataset boundaries ----------
if "events_all" in locals() and not events_all.empty:
    _ev_max_ts = events_all["timestamp"].max()
    _ev_min_ts = events_all["timestamp"].min()
else:
    st.error("No events loaded — Tab 3 requires events_all."); st.stop()

if "fixed_end_date" not in locals():
    fixed_end_date = _ev_max_ts.normalize().date()
if "min_start_date" not in locals():
    min_start_date = _ev_min_ts.normalize().date()
if "max_start_date" not in locals():
    max_start_date = _ev_max_ts.normalize().date()
if "end_ts_inclusive" not in locals():
    end_ts_inclusive = pd.to_datetime(fixed_end_date)
if "end_ts_exclusive" not in locals():
    end_ts_exclusive = end_ts_inclusive + pd.Timedelta(days=1)

with tab_segments:
    st.subheader("📧 Retention & Email Suggestions")

    # --- KPIs ---
    st.markdown("""<style>
    .kpi{padding:14px;border-radius:12px;text-align:center;
         font-weight:600;color:#222}
    .kpi .v{font-size:22px;font-weight:800;color:#000}
    .blue{border:2px solid #2980b9;background:#d6eaf8}
    .red{border:2px solid #c0392b;background:#f5b7b1}
    .green{border:2px solid #27ae60;background:#abebc6}
    </style>""", unsafe_allow_html=True)

    total_customers = feats["visitorid"].nunique() if not feats.empty else 0
    high_risk_n = (feats["risk_category"]=="🔴 High").sum() if not feats.empty else 0
    retained_n = (feats["retention"]=="✅ Retained").sum() if not feats.empty else 0
    churned_n = (feats["retention"]=="❌ Not Retained").sum() if not feats.empty else 0
    churn_rate = (churned_n / total_customers) if total_customers > 0 else 0

    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='kpi blue'>👥 Users<div class='v'>{total_customers:,}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi red'>🔴 High Risk<div class='v'>{high_risk_n:,}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi green'>✅ Retained<div class='v'>{retained_n:,}</div></div>", unsafe_allow_html=True)

        # --- Date filters for churn status by segment ---
    st.markdown("### 📅 Select Analysis Window")
    csa, csb = st.columns(2)
    seg_start = csa.date_input("**Segment Start Date** (affects graphs & Retention Table)",
                               value=max(min_start_date, (end_ts_inclusive - timedelta(days=60)).date()),
                               min_value=min_start_date, max_value=max_start_date)
    seg_end   = csb.date_input("**Segment End Date** (affects graphs & Retention Table)", 
                               value=fixed_end_date,
                               min_value=min_start_date, max_value=max_start_date)
    st.caption("🔎 Adjust the dates above to change what’s shown in the graphs and the Retention Table.")

    seg_start_ts = pd.to_datetime(seg_start)
    seg_end_ts_incl = pd.to_datetime(seg_end)
    seg_end_ts_excl = seg_end_ts_incl + pd.Timedelta(days=1)

    if seg_start_ts >= seg_end_ts_incl:
        st.error("Start must be before end.")
    else:
        mask_seg = (events_all["timestamp"] >= seg_start_ts) & (events_all["timestamp"] < seg_end_ts_excl)
        ev_seg = events_all.loc[mask_seg].copy()
        feats_seg = build_customer_features(ev_seg, ref_end=seg_end_ts_incl)

        if feats_seg.empty:
            st.info("No events in the selected range.")
        else:
            # ✅ retention always against dataset end
            feats_seg["retention"] = retention_flag(feats_seg["last_event_ts"], ref_end=ref_end)
            feats_seg["segment"]   = feats_seg.apply(lambda r: behavior_segment(r, ref_end), axis=1)
            feats_seg = cap_new_to_twenty_percent(feats_seg)
            feats_seg = compute_risk(feats_seg)

            # --- Behavior column (distinct witty names) ---
            behavior_map = {
                "🆕 New": "🌱 Freshie",
                "🟢 Recent Visitor": "👣 Curious Visitor",
                "🛍️ Repeat Buyer": "🏆 Shopping Champ",
                "🛍️ Active Buyer": "⚡ Frequent Shopper",
                "🛒 Cart Abandoner": "👻 Cart Ghost",
                "👀 Window Shopper": "🕶️ Browse-a-holic",
                "😴 Inactive": "💤 Snoozer",
                "🧊 One-and-Done": "🥶 One-Hit Wonder",
                "🌈 Happy Camper": "🌟 Loyalist",
                "🙂 Other": "🤔 Mixed Bag"
            }
            feats_seg["behavior_name"] = feats_seg["segment"].map(behavior_map).fillna("🤔 Mixed Bag")

            # --- Recommendations (segment + risk + behavior) ---
            def adjusted_reco(row):
                if row["retention"] == "✅ Retained":
                    return "🎁 Loyalty perk: keep engaged with VIP coupons"
                if row["risk_category"] == "🔴 High":
                    return "🚨 Urgent: win-back campaign with 20% off + free shipping"
                if row["risk_category"] == "🟠 Medium":
                    return "⚠️ Nudge: personalized picks with 10% incentive"
                if row["segment"] == "🛒 Cart Abandoner":
                    return "👻 Cart recovery: quick checkout link + discount"
                if row["segment"] == "👀 Window Shopper":
                    return "🕶️ Push trending bundle offers to convert browsers"
                if row["segment"] == "😴 Inactive":
                    return "💤 Wake-up mail: comeback offer + surprise gift"
                return "💡 General engagement: new arrivals & recommendations"

            feats_seg["recommendation"] = feats_seg.apply(adjusted_reco, axis=1)

            # --- Graphs ---
            st.markdown("### 📊 Segment Distribution & Churn Status")
            g1, g2 = st.columns(2)

            with g1:
                seg_counts = feats_seg["segment"].value_counts().reset_index()
                seg_counts.columns = ["segment","count"]

                fig_sun = px.pie(
                    seg_counts,
                    names="segment",
                    values="count",
                    hole=0.35,
                    title="Segment Distribution"
                )
                fig_sun.update_traces(
                    textinfo="label+percent",
                    textposition="outside",
                    pull=[0.05]*len(seg_counts),
                    textfont=dict(size=14)
                )
                fig_sun.update_layout(
                    height=450, width=450, margin=dict(l=40,r=40,t=60,b=40),
                    showlegend=False
                )
                st.plotly_chart(fig_sun, use_container_width=True)

            with g2:
                seg_opts = st.multiselect("Filter by Segment", sorted(feats_seg["segment"].unique()), default=None)
                risk_opts = st.multiselect("Filter by Risk", ["🔴 High","🟠 Medium","🟢 Low"],
                                           default=["🔴 High","🟠 Medium","🟢 Low"])
                ret_opts = st.multiselect("Filter by Retention", ["✅ Retained","❌ Not Retained"],
                                          default=["✅ Retained","❌ Not Retained"])

                df_bar = feats_seg.copy()
                if seg_opts: df_bar = df_bar[df_bar["segment"].isin(seg_opts)]
                if risk_opts: df_bar = df_bar[df_bar["risk_category"].isin(risk_opts)]
                if ret_opts: df_bar = df_bar[df_bar["retention"].isin(ret_opts)]

                ch_tbl = df_bar.groupby(["segment","retention"]).size().reset_index(name="count")
                fig_churn = px.bar(ch_tbl, x="segment", y="count", color="retention", barmode="stack",
                                   title="Churn Status by Segment")
                st.plotly_chart(fig_churn, use_container_width=True)

                # --- Retention Table ---
    st.markdown("### 📑 Retention Table")

    rt = feats_seg.copy()

    # ✅ Adjust retention: blank for Medium/Low not retained
    rt["retention_adj"] = rt.apply(
        lambda r: ("" if r["retention"] == "❌ Not Retained" and r["risk_category"] in ["🟠 Medium","🟢 Low"]
                   else r["retention"]),
        axis=1
    )

    # Witty behavior names (different from segment)
    behavior_map = {
        "🆕 New": "🌱 Freshie",
        "🟢 Recent Visitor": "👣 Curious Visitor",
        "🛍️ Repeat Buyer": "🏆 Shopping Champ",
        "🛍️ Active Buyer": "⚡ Frequent Shopper",
        "🛒 Cart Abandoner": "👻 Cart Ghost",
        "👀 Window Shopper": "🕶️ Browse-a-holic",
        "😴 Inactive": "💤 Snoozer",
        "🧊 One-and-Done": "🥶 One-Hit Wonder",
        "🌈 Happy Camper": "🌟 Loyalist",
        "🙂 Other": "🤔 Mixed Bag"
    }
    rt["behavior_of_customer"] = rt["segment"].map(behavior_map).fillna("🤔 Mixed Bag")

    # Smarter recommendations
    def adjusted_reco(row):
        if row["retention_adj"] == "✅ Retained":
            return "🎁 Loyalty perk: keep engaged with VIP coupons"
        if row["risk_category"] == "🔴 High":
            return "🚨 Urgent: win-back campaign with 20% off + free shipping"
        if row["risk_category"] == "🟠 Medium":
            return "⚠️ Nudge: personalized picks with 10% incentive"
        if row["segment"] == "🛒 Cart Abandoner":
            return "👻 Cart recovery: quick checkout link + discount"
        if row["segment"] == "👀 Window Shopper":
            return "🕶️ Bundle offers to convert browsers"
        if row["segment"] == "😴 Inactive":
            return "💤 Wake-up mail: comeback offer + surprise gift"
        return "💡 General engagement: new arrivals & recommendations"

    rt["recommendation"] = rt.apply(adjusted_reco, axis=1)

    # --- Filters (keep them!) ---
    f1, f2, f3 = st.columns([1,1,1])
    seg_opts = f1.multiselect("Segment", sorted(rt["segment"].unique()), default=None)
    ret_opts = f2.multiselect("Retention Status", ["✅ Retained","❌ Not Retained"],
                              default=["✅ Retained","❌ Not Retained"])
    max_rows = f3.number_input("Rows to load", min_value=100, max_value=5000, step=100, value=500)

    if seg_opts: rt = rt[rt["segment"].isin(seg_opts)]
    if ret_opts: rt = rt[rt["retention"].isin(ret_opts)]

    rt_cols = ["visitorid","last_event_ts","days_since_last_event",
               "views","carts","purchases","segment",
               "behavior_of_customer","risk_category","retention_adj","recommendation"]

    df_show = rt[rt_cols].reset_index(drop=True).head(int(max_rows))
    st.dataframe(df_show, use_container_width=True, height=320)

    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Retention Table CSV", data=csv,
                       file_name="retention_table.csv", mime="text/csv")

    # --- Email Tool ---
    st.markdown("#### 🚨 Compose Email for Immediate Action Customers")
    st.caption("High-risk ❌ Not Retained customers only")

    high_risk_ids = rt.loc[
        (rt["risk_category"]=="🔴 High") & (rt["retention"]=="❌ Not Retained"),
        "visitorid"
    ].astype(str).tolist()

    if high_risk_ids:
        select_all = st.checkbox("Select All High-risk ❌ Not Retained Customers")
        sel_ids = high_risk_ids if select_all else st.multiselect(
            "Select customers (searchable)",  # ✅ search bar stays
            options=high_risk_ids,
            default=[]
        )

        colS, colP = st.columns(2)
        subject = colS.text_input("Subject", value="We miss you — exclusive comeback offer inside")
        prehdr  = colP.text_input("Preheader", value="Open for your limited-time discount and curated picks")
        body    = st.text_area("Body", height=200,
                               value=("Hi there,\n\nWe noticed you haven't been active lately. "
                                      "Here’s a special offer to win you back: {OFFER}.\n\n"
                                      "Complete checkout in one click.\n\nCheers,\nThe Team"))

        if sel_ids:
            out = pd.DataFrame({
                "visitorid": sel_ids,
                "email": [f"{vid}@example.com" for vid in sel_ids],
                "subject": subject,
                "preheader": prehdr,
                "body": body
            })
            st.download_button(
                "⬇️ Download emails.csv",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="emails_immediate_action.csv",
                mime="text/csv"
            )
        else:
            st.info("Select at least one customer to generate the CSV.")
    else:
        st.warning("No high-risk not retained customers available for email generation.")

# ===== 4) OPTIMIZER: Risk-based & Proxy Uplift Simulator =====
with tab_opt:
    st.subheader("🧪 Optimizer / Simulator")

    if feats.empty:
        st.info("Load a window first to run a simulation.")
    else:
        # --- USAGE GUIDE ---
        with st.expander("ℹ️ Usage Guide"):
            st.markdown("""
            This tab helps you **decide where to focus your retention budget**.

            **Key Terms:**
            - **AOV (Average Order Value):** Average revenue per order (£).
            - **Uplift (pp):** The % of customers who will stay only because of the campaign.
            - **Discount (%):** Incentive given to customers (e.g., 20% off coupon).
            - **Cost per Email (£):** Small cost of sending each email.
            - **Baseline Retention:** Customers already retained without any campaign.
            - **Net Value:** (Expected revenue gained) − (discount cost + email cost).

            **Modes:**
            - **🚨 Risk-Based Optimizer:** Targets the riskiest customers first, simulates retention gain vs cost.
            - **📈 Proxy Uplift-Based Optimizer:** Uses risk × recent activity to highlight customers 
              most likely to respond positively to interventions.
            """)

        # --- MODE TOGGLE ---
        mode = st.radio(
            "Choose Optimizer Mode",
            ["🚨 Risk-Based Optimizer", "📈 Proxy Uplift-Based Optimizer (Demo)"]
        )

        # ------------------------
        # 🚨 RISK-BASED OPTIMIZER
        # ------------------------
        if "Risk-Based" in mode:
            st.markdown("### 🚨 Risk-Based Optimizer")

            c1, c2, c3 = st.columns(3)
            offer_high = c1.slider("High risk discount (%)", 10, 40, 20, step=1)
            offer_med  = c2.slider("Medium risk discount (%)", 5, 25, 12, step=1)
            offer_low  = c3.slider("Low risk discount (%)", 0, 15, 5, step=1)

            u1, u2, u3 = st.columns(3)
            uplift_high = u1.slider("High risk uplift (pp)", 1, 20, 8)
            uplift_med  = u2.slider("Medium risk uplift (pp)", 1, 15, 5)
            uplift_low  = u3.slider("Low risk uplift (pp)", 0, 8, 2)

            avg_order = st.number_input("Avg order value (AOV £)", min_value=1.0, value=60.0, step=1.0)
            cost_per_email = st.number_input("Cost per email (£)", min_value=0.0, value=0.01, step=0.01)

            # Baseline retention proxy
            base_ret = (feats["retention"] == "✅ Retained").mean() if not feats.empty else 0.0

            # Counts per risk
            counts = feats["risk_category"].value_counts().reindex(["🔴 High","🟠 Medium","🟢 Low"]).fillna(0)
            ups = {"🔴 High": uplift_high/100.0, "🟠 Medium": uplift_med/100.0, "🟢 Low": uplift_low/100.0}
            disc = {"🔴 High": offer_high/100.0, "🟠 Medium": offer_med/100.0, "🟢 Low": offer_low/100.0}

            expected_additional_retained = sum(counts[r]*ups[r] for r in counts.index)
            expected_cost_discounts = sum(counts[r]*ups[r]*(avg_order*disc[r]) for r in counts.index)
            email_cost = len(feats) * cost_per_email
            net_value = expected_additional_retained*avg_order - (expected_cost_discounts + email_cost)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Baseline Retention", f"{base_ret:.1%}")
            s2.metric("Expected add'l Retained", f"{int(expected_additional_retained):,}")
            s3.metric("Campaign Cost (est.)", f"£{(expected_cost_discounts + email_cost):,.0f}")
            s4.metric("Net Value (est.)", f"£{net_value:,.0f}")

        # --------------------------------------------
        # 📈 PROXY UPLIFT-BASED OPTIMIZER (DEMO)
        # --------------------------------------------
        if "Uplift" in mode:
            st.markdown("### 📈 Proxy Uplift-Based Optimizer (Demo)")

            df_uplift = feats.copy()
            if not df_uplift.empty:
                # Risk & activity weights
                risk_map = {"🔴 High": 3, "🟠 Medium": 2, "🟢 Low": 1}
                df_uplift["risk_w"] = df_uplift["risk_category"].map(risk_map).fillna(1)

                # Scale recency to your ~60-day dataset
                df_uplift["activity_w"] = 1 / (1 + df_uplift["days_since_last_event"] / 60)

                # Weighted-sum uplift score (risk-dominant as agreed)
                df_uplift["uplift_score"] = (
                    0.85 * df_uplift["risk_w"] + 0.15 * (df_uplift["activity_w"] * 10)
                ).round(2)

                # --- Recommendations ---
                def adjusted_reco(row):
                    if row.get("retention") == "✅ Retained":
                        return "🎁 Loyalty perk: keep engaged with VIP coupons"
                    if row["risk_category"] == "🔴 High":
                        return "🚨 Urgent: win-back campaign with 20% off + free shipping"
                    if row["risk_category"] == "🟠 Medium":
                        return "⚠️ Nudge: personalized picks with 10% incentive"
                    if row["segment"] == "🛒 Cart Abandoner":
                        return "👻 Cart recovery: quick checkout link + discount"
                    if row["segment"] == "👀 Window Shopper":
                        return "🕶️ Bundle offers to convert browsers"
                    if row["segment"] == "😴 Inactive":
                        return "💤 Wake-up mail: comeback offer + surprise gift"
                    return "💡 General engagement: new arrivals & recommendations"

                df_uplift["recommendation"] = df_uplift.apply(adjusted_reco, axis=1)

                # --- Controls / Filters ---
                f1, f2, f3 = st.columns(3)
                risk_opts = f1.multiselect(
                    "Filter by Risk Category",
                    ["🔴 High", "🟠 Medium", "🟢 Low"],
                    default=["🔴 High", "🟠 Medium", "🟢 Low"]
                )
                max_days = f2.slider("Max days since last activity", 0, 60, 60)
                topN = f3.slider("Top-N to peg (by uplift score)", 20, 1000, 100, step=20)

                # Apply filters
                df_considered = df_uplift.copy()
                if risk_opts:
                    df_considered = df_considered[df_considered["risk_category"].isin(risk_opts)]
                df_considered = df_considered[df_considered["days_since_last_event"] <= max_days]

                # Pegged set = Top-N by uplift_score
                df_pegged = (
                    df_considered.sort_values("uplift_score", ascending=False)
                    .head(int(topN))
                    .reset_index(drop=True)
                )

                # --- KPIs ---
                total_customers = len(df_uplift)
                considered_customers = len(df_considered)
                pegged_customers = len(df_pegged)
                high_count = (df_pegged["risk_category"] == "🔴 High").sum()
                med_count  = (df_pegged["risk_category"] == "🟠 Medium").sum()

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("👥 Total Considered (all)", f"{total_customers:,}")
                k2.metric("🔎 Considered After Filters", f"{considered_customers:,}")
                k3.metric("🏆 Customers Pegged (Top-N)", f"{pegged_customers:,}")
                k4.metric("Breakdown (Pegged)", f"🔴 {high_count:,} | 🟠 {med_count:,}")

                # --- Table (pegged only) ---
                st.dataframe(
                    df_pegged[[
                        "visitorid","segment","risk_category",
                        "days_since_last_event","uplift_score","recommendation"
                    ]],
                    use_container_width=True, height=360
                )

                # --- Download ---
                st.download_button(
                    "⬇️ Download Pegged Uplift Candidates (CSV)",
                    data=df_pegged.to_csv(index=False).encode("utf-8"),
                    file_name="uplift_candidates_pegged.csv",
                    mime="text/csv"
                )
            else:
                st.info("No data available for uplift scoring.")

# ===== 5) ABOUT =====
with tab_about:
    st.header("ℹ️ About the Dashboard")
    st.write("""
    This dashboard was developed as part of a Master's dissertation project. 
    It provides an interactive interface for customer churn analysis, 
    combining heuristic and machine learning methods to segment customers, 
    calculate churn risk, and explore retention strategies.

    🛠️ **Features**
    - **Overview Tab:** High-level KPIs and churn rate.
    - **Churn Analysis Tab:** Explore churn metrics, inactivity, and trends.
    - **Clusters & Retention Tab:** Customer segmentation using K-means clustering.
    - **Optimizer Tab:** Simulate risk-based and uplift-based retention strategies.
    - **About Tab:** Project background and model evaluation.
    """)

    st.header("🤖 Model Evaluation")

    try:
        from joblib import load
        MODEL_PATH = ROOT / "models" / "risk_model.pkl"
        if MODEL_PATH.exists():
            payload = load(MODEL_PATH)

            def _get_any(m, keys, default=0.0):
                for k in keys:
                    if k in m and m[k] is not None:
                        return m[k]
                return default

            if "metrics_by_model" in payload:
                metrics_by_model = payload["metrics_by_model"]
                rows = []
                for model_name, m in metrics_by_model.items():
                    rows.append([
                        model_name,
                        round(_get_any(m, ["Accuracy","accuracy"]), 3),
                        round(_get_any(m, ["ROC-AUC","AUC","roc_auc"]), 3),
                        round(_get_any(m, ["PR-AUC","PR_AUC","pr_auc"]), 3),
                        round(_get_any(m, ["Brier","brier"]), 5),
                    ])
                df_metrics = pd.DataFrame(
                    rows, columns=["Model","Accuracy","ROC-AUC","PR-AUC","Brier"]
                )
                st.markdown("🤖 **Model Comparison**")
                st.dataframe(df_metrics, use_container_width=True)

                best = payload.get("best_name","(unknown)")
                best_auc = _get_any(payload.get("chosen_metrics", {}), ["ROC-AUC","AUC"], default=None)
                st.markdown(f"🏆 **Best model selected: {best}** (ROC-AUC={best_auc})")
            else:
                st.info("No model metrics available yet. Please run `train_risk.py` first.")
        else:
            st.warning("No saved model found. Run training script to generate models.")
    except Exception as e:
        st.error(f"Error loading model evaluation: {e}")
