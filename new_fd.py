import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import os
try:
    import pyarrow.parquet as pq
    import pyarrow
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="E-Commerce Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-high { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); }
    .risk-medium { background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%); }
    .risk-low { background: linear-gradient(135deg, #48cab2 0%, #2dd4bf 100%); }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e2e6;
    }
    .stTabs [data-baseweb="tab-selected"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_retail_rocket_data(uploaded_file=None, sample_size="All data"):
    """Load Retail Rocket dataset from uploaded CSV/Parquet file, local Parquet file, or sample data"""
    if not PYARROW_AVAILABLE:
        st.error("PyArrow is not installed. Please install it using 'pip install pyarrow' to use Parquet files. Using sample data.")
        return generate_sample_data(), False

    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                required_columns = ['timestamp', 'visitorid', 'event']
                if not all(col in df.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in df.columns]
                    st.error(f"Uploaded CSV missing required columns: {', '.join(missing)}. Using sample data.")
                    return generate_sample_data(), False
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
                    if df['timestamp'].isna().all():
                        st.error("All timestamps in CSV are invalid. Expected milliseconds since epoch (e.g., 1433221332117). Using sample data.")
                        return generate_sample_data(), False
                    if df['timestamp'].isna().any():
                        st.warning(f"{df['timestamp'].isna().sum():,} invalid timestamps found in CSV. These rows will be ignored.")
                        df = df.dropna(subset=['timestamp'])
                except Exception as e:
                    st.error(f"Invalid timestamp format in CSV: {str(e)}. Expected milliseconds since epoch. Using sample data.")
                    return generate_sample_data(), False
                parquet_path = f"uploaded_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                try:
                    df.to_parquet(parquet_path, engine='pyarrow', index=False, compression='snappy')
                    st.info(f"Converted uploaded CSV to Parquet and saved as {parquet_path}")
                    df = pd.read_parquet(parquet_path, engine='pyarrow')
                except PermissionError:
                    st.error(f"Permission denied when saving {parquet_path}. Ensure the app has write access to the directory. Using sample data.")
                    return generate_sample_data(), False
                except Exception as e:
                    st.error(f"Failed to convert CSV to Parquet: {str(e)}. Using sample data.")
                    return generate_sample_data(), False
            elif file_name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file, engine='pyarrow')
                required_columns = ['timestamp', 'visitorid', 'event']
                if not all(col in df.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in df.columns]
                    st.error(f"Uploaded Parquet missing required columns: {', '.join(missing)}. Using sample data.")
                    return generate_sample_data(), False
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    if df['timestamp'].isna().all():
                        st.error("All timestamps in Parquet are invalid. Using sample data.")
                        return generate_sample_data(), False
                    if df['timestamp'].isna().any():
                        st.warning(f"{df['timestamp'].isna().sum():,} invalid timestamps found in Parquet. These rows will be ignored.")
                        df = df.dropna(subset=['timestamp'])
                except Exception as e:
                    st.error(f"Invalid timestamp format in Parquet: {str(e)}. Using sample data.")
                    return generate_sample_data(), False
            else:
                st.error("Unsupported file format. Please upload a .csv or .parquet file. Using sample data.")
                return generate_sample_data(), False
            if sample_size != "All data":
                try:
                    sample_users = df['visitorid'].drop_duplicates().sample(n=min(int(sample_size), df['visitorid'].nunique()), random_state=42)
                    df = df[df['visitorid'].isin(sample_users)]
                except ValueError as e:
                    st.error(f"Sampling error: {str(e)}. Ensure sample size is valid. Using sample data.")
                    return generate_sample_data(), False
            st.success(f"✅ Loaded {len(df):,} events from uploaded file")
            return df, True
        except Exception as e:
            st.error(f"Error loading uploaded file: {str(e)}. Using sample data.")
            return generate_sample_data(), False
    
    try:
        if not os.path.exists('events.parquet'):
            raise FileNotFoundError("Local events.parquet not found")
        df = pd.read_parquet('events.parquet', engine='pyarrow')
        required_columns = ['timestamp', 'visitorid', 'event']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Local events.parquet missing required columns: {', '.join(missing)}. Using sample data.")
            return generate_sample_data(), False
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].isna().all():
            st.error("All timestamps in local events.parquet are invalid. Using sample data.")
            return generate_sample_data(), False
        if df['timestamp'].isna().any():
            st.warning(f"{df['timestamp'].isna().sum():,} invalid timestamps found in local events.parquet. These rows will be ignored.")
            df = df.dropna(subset=['timestamp'])
        if sample_size != "All data":
            try:
                sample_users = df['visitorid'].drop_duplicates().sample(n=min(int(sample_size), df['visitorid'].nunique()), random_state=42)
                df = df[df['visitorid'].isin(sample_users)]
            except ValueError as e:
                st.error(f"Sampling error: {str(e)}. Ensure sample size is valid. Using sample data.")
                return generate_sample_data(), False
        st.success(f"✅ Loaded {len(df):,} events from local events.parquet")
        return df, True
    except FileNotFoundError:
        st.error("Local events.parquet not found. Please create it using the conversion script or upload a file. Using sample data.")
        return generate_sample_data(), False
    except PermissionError:
        st.error("Permission denied when accessing local events.parquet. Check file permissions. Using sample data.")
        return generate_sample_data(), False
    except Exception as e:
        st.error(f"Error loading local events.parquet: {str(e)}. Using sample data.")
        return generate_sample_data(), False

@st.cache_data
def generate_sample_data():
    """Fallback sample data if no file is uploaded or loaded"""
    np.random.seed(42)
    n_users = 10000
    n_events = 50000
    user_ids = np.random.randint(100000, 999999, n_users)
    events_data = []
    
    start_date = pd.to_datetime("2015-05-01")
    for _ in range(n_events):
        user_id = np.random.choice(user_ids)
        days_offset = np.random.randint(0, 150)
        timestamp = start_date + timedelta(days=days_offset, hours=np.random.randint(0, 24))
        event_type = np.random.choice(['view', 'addtocart', 'transaction'], p=[0.6, 0.2, 0.2])
        item_id = np.random.randint(10000, 99999)
        transaction_id = np.random.randint(1000, 9999) if event_type == 'transaction' else None
        
        events_data.append({
            'timestamp': timestamp,
            'visitorid': user_id,
            'event': event_type,
            'itemid': item_id,
            'transactionid': transaction_id
        })
    
    df = pd.DataFrame(events_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_data
def extract_features(df, min_events=2, avg_transaction_value=100, decay_factor=120, base_event_value=20):
    """Extract features for churn prediction with fallback conversion rate"""
    if df.empty:
        st.error("Input DataFrame is empty. Check data loading. Using sample data features.")
        df = generate_sample_data()
    
    current_date = df['timestamp'].max()
    
    user_event_counts = df.groupby('visitorid').size()
    active_users = user_event_counts[user_event_counts >= min_events].index
    df = df[df['visitorid'].isin(active_users)]
    
    user_features = df.groupby('visitorid').agg({
        'timestamp': ['count', 'min', 'max'],
        'event': lambda x: x.nunique(),
        'itemid': 'nunique',
        'transactionid': lambda x: x.notna().sum()
    }).round(2)
    
    user_features.columns = [
        'total_events', 'first_activity', 'last_activity',
        'event_types', 'unique_items', 'total_transactions'
    ]
    
    user_features['activity_span'] = (user_features['last_activity'] - user_features['first_activity']).dt.days
    user_features['recency'] = (current_date - user_features['last_activity']).dt.days
    user_features['frequency'] = user_features['total_events'] / (user_features['activity_span'] + 1)
    
    event_breakdown = df.groupby(['visitorid', 'event']).size().unstack(fill_value=0)
    for col in ['view', 'addtocart', 'transaction']:
        if col not in event_breakdown.columns:
            event_breakdown[col] = 0
    user_features = user_features.join(event_breakdown, how='left').fillna(0)
    
    user_features['conversion_rate'] = np.where(
        user_features['total_events'] > 0,
        user_features['transaction'] / user_features['total_events'],
        0.01
    )
    user_features['cart_abandonment_rate'] = np.where(
        user_features['addtocart'] > 0,
        (user_features['addtocart'] - user_features['transaction']) / user_features['addtocart'],
        0
    )
    user_features['view_to_cart_rate'] = np.where(
        user_features['view'] > 0,
        user_features['addtocart'] / user_features['view'],
        0
    )
    
    session_stats = df.groupby(['visitorid', df['timestamp'].dt.date]).size().reset_index()
    session_stats = session_stats.groupby('visitorid')[0].agg(['mean', 'max']).fillna(0)
    session_stats.columns = ['avg_events_per_day', 'max_events_per_day']
    user_features = user_features.join(session_stats, how='left').fillna(0)
    
    user_features['clv_estimate'] = (
        user_features['transaction'] * avg_transaction_value * 
        np.exp(-user_features['recency'] / decay_factor)
    )
    user_features['clv_estimate'] = np.where(
        user_features['transaction'] == 0,
        user_features['total_events'] * base_event_value,
        user_features['clv_estimate']
    )
    user_features['clv_estimate'] = user_features['clv_estimate'].clip(lower=10)
    
    user_features = user_features.fillna(0)
    
    for col in ['total_events', 'frequency']:
        if col in user_features.columns:
            q99 = user_features[col].quantile(0.99)
            user_features[col] = user_features[col].clip(upper=q99)
    
    return user_features

@st.cache_data
def predict_churn(features):
    """Revised churn prediction model with capped recency and adjusted weights"""
    features = features.fillna({
        'recency': features['recency'].max(),
        'frequency': 0,
        'conversion_rate': 0,
        'cart_abandonment_rate': 0,
        'total_transactions': 0
    })
    
    churn_score = (
        np.minimum(features['recency'] / 90, 1) * 0.20 +
        (1 - (features['frequency'] / (features['frequency'].max() + 1))) * 0.40 +
        (1 - features['conversion_rate']) * 0.25 +
        (features['cart_abandonment_rate']) * 0.15
    )
    
    condition = (features['recency'] > 90) & (features['total_transactions'] == 0)
    churn_score = np.where(condition, churn_score * 1.3, churn_score)
    
    churn_score_min = churn_score.min()
    churn_score_max = churn_score.max()
    churn_probability = np.where(
        churn_score_max > churn_score_min,
        (churn_score - churn_score_min) / (churn_score_max - churn_score_min),
        0
    )
    
    return churn_probability

@st.cache_data
def assign_churn_cause_and_recommendation(features, churn_threshold=0.7):
    """Assign churn cause, recommendation, and risk segment based on key features"""
    causes = []
    recommendations = []
    
    # Thresholds for identifying churn causes
    recency_threshold = 90  # High recency (inactive for over 90 days)
    frequency_threshold = features['frequency'].median()  # Below median frequency
    cart_abandonment_threshold = 0.5  # High cart abandonment rate
    
    for _, row in features.iterrows():
        # Identify primary churn cause
        if row['recency'] > recency_threshold:
            cause = "High Recency (Long Inactivity)"
            recommendation = "Send re-engagement email with personalized offer"
        elif row['frequency'] < frequency_threshold:
            cause = "Low Activity Frequency"
            recommendation = "Increase engagement with regular promotions"
        elif row['cart_abandonment_rate'] > cart_abandonment_threshold:
            cause = "High Cart Abandonment"
            recommendation = "Offer cart recovery incentives (e.g., discount, free shipping)"
        else:
            cause = "Multiple Factors"
            recommendation = "Review user behavior for targeted interventions"
            
        causes.append(cause)
        recommendations.append(recommendation)
    
    features['churn_cause'] = causes
    features['recommendation'] = recommendations
    
    # Assign risk segment based on churn probability
    churn_probs = features['churn_probability']
    q60 = churn_probs.quantile(0.60)
    q85 = churn_probs.quantile(0.85)
    features['risk_segment'] = pd.cut(
        churn_probs, 
        bins=[0, q60, q85, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk'],
        include_lowest=True
    )
    
    return features

@st.cache_data
def simulate_revenue_at_risk(features, churn_threshold, avg_transaction_value, decay_factor, base_event_value, recency_reduction=0, transaction_increase=0):
    """Simulate Revenue at Risk with adjusted parameters"""
    sim_features = features.copy()
    
    sim_features['recency'] = sim_features['recency'] * (1 - recency_reduction / 100)
    sim_features['transaction'] = sim_features['transaction'] * (1 + transaction_increase / 100)
    
    sim_features['clv_estimate'] = (
        sim_features['transaction'] * avg_transaction_value * 
        np.exp(-sim_features['recency'] / decay_factor)
    )
    sim_features['clv_estimate'] = np.where(
        sim_features['transaction'] == 0,
        sim_features['total_events'] * base_event_value,
        sim_features['clv_estimate']
    )
    sim_features['clv_estimate'] = sim_features['clv_estimate'].clip(lower=10)
    
    sim_churn_score = (
        np.minimum(sim_features['recency'] / 90, 1) * 0.20 +
        (1 - (sim_features['frequency'] / (sim_features['frequency'].max() + 1))) * 0.40 +
        (1 - sim_features['conversion_rate']) * 0.25 +
        (sim_features['cart_abandonment_rate']) * 0.15
    )
    condition = (sim_features['recency'] > 90) & (sim_features['total_transactions'] == 0)
    sim_churn_score = np.where(condition, sim_churn_score * 1.3, sim_churn_score)
    sim_features['churn_probability'] = np.where(
        sim_churn_score.max() > sim_churn_score.min(),
        (sim_churn_score - sim_churn_score.min()) / (sim_churn_score.max() - sim_churn_score.min()),
        0
    )
    
    churned_users = len(sim_features[sim_features['churn_probability'] > churn_threshold])
    churn_rate = churned_users / len(sim_features) if len(sim_features) > 0 else 0
    revenue_at_risk = (sim_features[sim_features['churn_probability'] > churn_threshold]['clv_estimate'] * 
                       sim_features[sim_features['churn_probability'] > churn_threshold]['churn_probability']).sum()
    avg_clv = sim_features['clv_estimate'].mean()
    
    return churned_users, churn_rate, revenue_at_risk, avg_clv

def main():
    st.title("🛍️ E-Commerce Churn Prediction Dashboard")
    st.markdown("*Retail Rocket Dataset Analysis*")
    
    st.sidebar.header("Dashboard Controls")
    uploaded_file = st.sidebar.file_uploader("Upload events file", type=["csv", "parquet"], help="Upload a CSV or Parquet file with columns: timestamp, visitorid, event, itemid, transactionid")
    sample_size = st.sidebar.selectbox(
        "Data Sample Size (for faster processing)",
        options=[100, 1000, 5000, 10000, 50000, "All data"],
        index=2,
        help="Select number of unique users to process"
    )
    min_events = st.sidebar.slider(
        "Minimum Events per User",
        min_value=1,
        max_value=10,
        value=2,
        help="Filter users with at least this many events"
    )
    churn_threshold = st.sidebar.slider(
        "Churn Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Users above this threshold are considered at risk"
    )
    
    with st.spinner("Loading dataset..."):
        df, data_loaded_successfully = load_retail_rocket_data(uploaded_file, sample_size)
        
        if data_loaded_successfully:
            if uploaded_file is not None:
                st.sidebar.success(f"✅ Loaded {len(df):,} events from uploaded file")
            elif sample_size == "All data":
                st.sidebar.success(f"✅ Loaded {len(df):,} events from {df['visitorid'].nunique():,} users")
            else:
                st.sidebar.info(f"📊 Using sample: {len(df):,} events from {df['visitorid'].nunique():,} users")
        else:
            st.sidebar.error("⚠️ Using sample data due to loading issues. Check error messages above for details.")
        
        avg_transaction_value = 100
        decay_factor = 120
        base_event_value = 20
        features = extract_features(df, min_events, avg_transaction_value, decay_factor, base_event_value)
        churn_probs = predict_churn(features)
        features['churn_probability'] = churn_probs
        features = assign_churn_cause_and_recommendation(features, churn_threshold)
    
    st.sidebar.markdown(f"**Dataset Date Range:** {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    date_range = st.sidebar.date_input(
        "Analysis Date Range",
        value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
        min_value=df['timestamp'].min().date(),
        max_value=df['timestamp'].max().date(),
        help="Select the date range for analysis"
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_filtered = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    else:
        df_filtered = df
        st.sidebar.warning("Incomplete date range selected. Using full dataset.")
    
    total_users = len(features)
    churned_users = len(features[features['churn_probability'] > churn_threshold])
    churn_rate = churned_users / total_users if total_users > 0 else 0
    revenue_at_risk = (features[features['churn_probability'] > churn_threshold]['clv_estimate'] * 
                       features[features['churn_probability'] > churn_threshold]['churn_probability']).sum()
    avg_clv = features['clv_estimate'].mean()
    avg_recency = features['recency'].mean()
    
    st.header("📊 Executive Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Total Users",
            value=f"{total_users:,}",
            delta=f"{len(df_filtered['visitorid'].unique()):,} active"
        )
    
    with col2:
        st.metric(
            label="📉 Churn Rate",
            value=f"{churn_rate:.1%}",
            delta=f"{churned_users:,} at risk",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="💷 Approx. Revenue at Risk",
            value=f"£{revenue_at_risk:,.0f}",
            delta=f"Est. £{revenue_at_risk/churned_users if churned_users > 0 else 0:.0f} per user",
            help="Estimated based on average user value"
        )
    
    with col4:
        st.metric(
            label="📅 Avg Days Since Last Visit",
            value=f"{avg_recency:.1f}",
            delta=f"£{avg_clv:.0f} avg CLV"
        )
    
    # Tabbed interface
    tabs = st.tabs(["Churn Analysis", "Feature Importance & Data Explorer", "Simulation Tools"])
    
    with tabs[0]:
        st.subheader("Churn Analysis")
        st.write(f"**Churn Probability Threshold: {churn_threshold:.2f}** | **Churn Rate: {churn_rate:.1%}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                features, 
                x='churn_probability',
                nbins=20,
                title="Distribution of Churn Probabilities",
                labels={'churn_probability': 'Churn Probability', 'count': 'Number of Users'}
            )
            fig_hist.add_vline(x=churn_threshold, line_dash="dash", line_color="red", 
                              annotation_text=f"Threshold: {churn_threshold}")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            segment_counts = features['risk_segment'].value_counts()
            fig_pie = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Distribution in Risk Zones",
                color_discrete_map={
                    'Low Risk': '#48cab2',
                    'Medium Risk': '#feca57',
                    'High Risk': '#ff6b6b'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("Top High-Risk Customers with Recent Activity")
        high_risk_recent = features[
            (features['churn_probability'] > churn_threshold)
        ].sort_values('recency', ascending=True).head(100)
        if not high_risk_recent.empty:
            st.dataframe(
                high_risk_recent[['clv_estimate', 'churn_probability', 'recency', 'total_events', 'churn_cause', 'recommendation']],
                column_config={
                    "clv_estimate": st.column_config.NumberColumn("CLV Estimate (£)", format="£%.0f")
                },
                use_container_width=True
            )
        else:
            st.info("No high-risk customers with current threshold.")
    
    with tabs[1]:
        st.subheader("Feature Importance Analysis")
        
        numeric_features = ['recency', 'frequency', 'total_events', 'conversion_rate', 
                           'view', 'addtocart', 'transaction', 'unique_items',
                           'view_to_cart_rate', 'cart_abandonment_rate']
        
        correlations = features[numeric_features + ['churn_probability']].corr()['churn_probability'].abs().sort_values(ascending=False)[1:]
        
        fig_importance = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title="Feature Importance (Correlation with Churn Probability)",
            labels={'x': 'Absolute Correlation', 'y': 'Features'}
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.subheader("Filtered User Data Explorer")
        st.markdown("Select risk segments to view and download user data")
        
        risk_filter = st.multiselect(
            "Filter by Risk Segment",
            options=['High Risk', 'Medium Risk', 'Low Risk'],
            default=['High Risk', 'Medium Risk', 'Low Risk'],
            key="risk_filter_explorer"
        )
        
        if risk_filter:
            filtered_features = features[features['risk_segment'].isin(risk_filter)]
            filter_label = ", ".join(risk_filter)
        else:
            filtered_features = features
            filter_label = "All Data"
        
        st.markdown(f"**Showing {len(filtered_features):,} users in {filter_label}**")
        st.dataframe(
            filtered_features[['clv_estimate', 'churn_probability', 'recency', 'total_events', 'churn_cause', 'recommendation', 'risk_segment']],
            column_config={
                "clv_estimate": st.column_config.NumberColumn("CLV Estimate (£)", format="£%.0f")
            },
            use_container_width=True
        )
        
        csv_filtered = filtered_features[['clv_estimate', 'churn_probability', 'recency', 'total_events', 'churn_cause', 'recommendation', 'risk_segment']].to_csv(index=True)
        st.download_button(
            label="📥 Download Filtered User Data",
            data=csv_filtered,
            file_name=f"filtered_users_{filter_label.replace(', ', '_').replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tabs[2]:
        st.subheader("Simulation & Intervention Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Revenue at Risk Simulation")
            
            st.markdown("**Adjust CLV Parameters**")
            sim_avg_transaction_value = st.number_input(
                "Average Transaction Value (£)",
                min_value=10.0,
                max_value=1000.0,
                value=100.0,
                step=10.0
            )
            sim_decay_factor = st.number_input(
                "Recency Decay Factor (days)",
                min_value=30.0,
                max_value=360.0,
                value=120.0,
                step=10.0
            )
            sim_base_event_value = st.number_input(
                "Base Event Value for Non-Transaction Users (£)",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0
            )
            
            st.markdown("**Simulate Engagement Improvements**")
            recency_reduction = st.slider(
                "Reduce Recency by (%)",
                min_value=0,
                max_value=50,
                value=0,
                help="Simulate users returning sooner (e.g., 10% reduces recency by 10%)"
            )
            transaction_increase = st.slider(
                "Increase Transactions by (%)",
                min_value=0,
                max_value=100,
                value=0,
                help="Simulate increase in transaction frequency"
            )
            
            sim_churned_users, sim_churn_rate, sim_revenue_at_risk, sim_avg_clv = simulate_revenue_at_risk(
                features, churn_threshold, sim_avg_transaction_value, sim_decay_factor, 
                sim_base_event_value, recency_reduction, transaction_increase
            )
            
            st.metric(
                "Simulated Churn Rate",
                f"{sim_churn_rate:.1%}",
                delta=f"{sim_churned_users:,} at risk"
            )
            st.metric(
                "Approx. Simulated Revenue at Risk",
                f"£{sim_revenue_at_risk:,.0f}",
                delta=f"Est. £{sim_revenue_at_risk/sim_churned_users if sim_churned_users > 0 else 0:.0f} per user",
                help="Estimated based on average user value"
            )
            st.metric(
                "Simulated Avg CLV",
                f"£{sim_avg_clv:.0f}",
                delta=f"Base: £{avg_clv:.0f}"
            )
        
        with col2:
            st.markdown("#### Threshold Impact Analysis")
            
            thresholds = np.arange(0.1, 1.0, 0.1)
            threshold_analysis = []
            
            for thresh in thresholds:
                sim_churned, sim_rate, sim_revenue, _ = simulate_revenue_at_risk(
                    features, thresh, sim_avg_transaction_value, sim_decay_factor, 
                    sim_base_event_value, recency_reduction, transaction_increase
                )
                threshold_analysis.append({
                    'threshold': thresh,
                    'users_at_risk': sim_churned,
                    'revenue_at_risk': sim_revenue
                })
            
            threshold_df = pd.DataFrame(threshold_analysis)
            
            fig_threshold = px.line(
                threshold_df,
                x='threshold',
                y='users_at_risk',
                title="Users and Revenue at Risk vs Threshold",
                labels={'threshold': 'Churn Probability Threshold', 'users_at_risk': 'Users at Risk'}
            )
            fig_threshold.add_trace(
                go.Scatter(
                    x=threshold_df['threshold'],
                    y=threshold_df['revenue_at_risk'],
                    name="Approx. Revenue at Risk (£)",
                    yaxis='y2'
                )
            )
            fig_threshold.update_layout(
                yaxis2=dict(title='Approx. Revenue at Risk (£)', overlaying='y', side='right')
            )
            st.plotly_chart(fig_threshold, use_container_width=True)
    
    st.markdown("---")
    st.markdown("**💡 Pro Tip:** Upload your own events.csv or events.parquet to explore Revenue at Risk scenarios! Parquet files load faster! For large datasets, select a smaller sample size to speed up processing.")

if __name__ == "__main__":
    main()