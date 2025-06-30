import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
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
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_retail_rocket_data(uploaded_file=None, sample_size="All data"):
    """Load Retail Rocket dataset from uploaded file, local file, or sample data"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['timestamp', 'visitorid', 'event']
            if not all(col in df.columns for col in required_columns):
                st.error(f"Uploaded file must contain columns: {', '.join(required_columns)}")
                return generate_sample_data(), False
            # Convert timestamp
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
                if df['timestamp'].isna().any():
                    st.warning("Some timestamps could not be parsed. Using sample data.")
                    return generate_sample_data(), False
            except Exception:
                st.error("Invalid timestamp format. Expected milliseconds. Using sample data.")
                return generate_sample_data(), False
            # Sample users if needed
            if sample_size != "All data":
                sample_users = df['visitorid'].drop_duplicates().sample(n=min(sample_size, df['visitorid'].nunique()), random_state=42)
                df = df[df['visitorid'].isin(sample_users)]
            st.success("✅ Uploaded dataset loaded successfully!")
            return df, True
        except Exception as e:
            st.error(f"Error loading uploaded file: {str(e)}. Using sample data.")
            return generate_sample_data(), False
    
    # Try loading local events.csv
    try:
        df = pd.read_csv('events.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        if df['timestamp'].isna().any():
            st.error("Invalid timestamps in events.csv. Using sample data.")
            return generate_sample_data(), False
        if sample_size != "All data":
            sample_users = df['visitorid'].drop_duplicates().sample(n=min(sample_size, df['visitorid'].nunique()), random_state=42)
            df = df[df['visitorid'].isin(sample_users)]
        st.success(f"✅ Loaded {len(df):,} events from local events.csv")
        return df, True
    except FileNotFoundError:
        st.error("Could not load events.csv. Using sample data.")
        return generate_sample_data(), False
    except Exception as e:
        st.error(f"Error loading local events.csv: {str(e)}. Using sample data.")
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
    current_date = df['timestamp'].max()
    
    # Filter users with minimum events
    user_event_counts = df.groupby('visitorid').size()
    active_users = user_event_counts[user_event_counts >= min_events].index
    df = df[df['visitorid'].isin(active_users)]
    
    # Basic user aggregations
    user_features = df.groupby('visitorid').agg({
        'timestamp': ['count', 'min', 'max', 'nunique'],
        'event': lambda x: x.nunique(),
        'itemid': 'nunique',
        'transactionid': lambda x: x.notna().sum()
    }).round(2)
    
    user_features.columns = [
        'total_events', 'first_activity', 'last_activity', 'unique_sessions',
        'event_types', 'unique_items', 'total_transactions'
    ]
    
    # Calculate temporal features
    user_features['activity_span'] = (user_features['last_activity'] - user_features['first_activity']).dt.days
    user_features['recency'] = (current_date - user_features['last_activity']).dt.days
    user_features['frequency'] = user_features['total_events'] / (user_features['activity_span'] + 1)
    
    # Event type breakdown
    event_breakdown = df.groupby(['visitorid', 'event']).size().unstack(fill_value=0)
    for col in ['view', 'addtocart', 'transaction']:
        if col not in event_breakdown.columns:
            event_breakdown[col] = 0
    user_features = user_features.join(event_breakdown, how='left').fillna(0)
    
    # Calculate business metrics with fallback
    user_features['conversion_rate'] = np.where(
        user_features['total_events'] > 0,
        user_features['transaction'] / user_features['total_events'],
        0.01  # Default fallback to avoid zero conversion
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
    
    # Session-based features
    session_stats = df.groupby(['visitorid', df['timestamp'].dt.date]).size().reset_index()
    session_stats = session_stats.groupby('visitorid')[0].agg(['mean', 'std', 'max']).fillna(0)
    session_stats.columns = ['avg_events_per_day', 'std_events_per_day', 'max_events_per_day']
    user_features = user_features.join(session_stats, how='left').fillna(0)
    
    # Calculate CLV estimate
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
    
    # Engagement features
    user_features['bounce_rate'] = (user_features['total_events'] == min_events).astype(int)
    user_features['return_visitor'] = (user_features['unique_sessions'] > 1).astype(int)
    
    # Normalize some features
    user_features['events_per_session'] = user_features['total_events'] / user_features['unique_sessions']
    user_features = user_features.fillna(0)
    
    # Remove outliers (exclude clv_estimate)
    for col in ['total_events', 'frequency']:
        if col in user_features.columns:
            q99 = user_features[col].quantile(0.99)
            user_features[col] = user_features[col].clip(upper=q99)
    
    return user_features

@st.cache_data
def predict_churn(features):
    """Revised churn prediction model with capped recency and adjusted weights, handling NaN values"""
    # Handle NaN values
    features = features.fillna({
        'recency': features['recency'].max(),  # Use max recency for NaN
        'frequency': 0,  # Default to 0 frequency for NaN
        'conversion_rate': 0,  # Default to 0 conversion for NaN
        'cart_abandonment_rate': 0,  # Default to 0 abandonment for NaN
        'total_transactions': 0  # Default to 0 transactions for NaN
    })
    
    # Calculate churn score with element-wise minimum
    churn_score = (
        np.minimum(features['recency'] / 90, 1) * 0.20 +  # 20% recency, capped at 90 days
        (1 - (features['frequency'] / (features['frequency'].max() + 1))) * 0.40 +  # 40% frequency
        (1 - features['conversion_rate']) * 0.25 +  # 25% conversion rate
        (features['cart_abandonment_rate']) * 0.15   # 15% cart abandonment
    )
    
    # Calibration for high recency and no transactions
    condition = (features['recency'] > 90) & (features['total_transactions'] == 0)
    churn_score = np.where(condition, churn_score * 1.3, churn_score)
    
    # Normalize to 0-1, handling potential zero division
    churn_score_min = churn_score.min()
    churn_score_max = churn_score.max()
    churn_probability = np.where(
        churn_score_max > churn_score_min,
        (churn_score - churn_score_min) / (churn_score_max - churn_score_min),
        0  # Default to 0 if max equals min to avoid division by zero
    )
    
    return churn_probability

@st.cache_data
def simulate_revenue_at_risk(features, churn_threshold, avg_transaction_value, decay_factor, base_event_value, recency_reduction=0, transaction_increase=0):
    """Simulate Revenue at Risk with adjusted parameters"""
    sim_features = features.copy()
    
    # Apply simulation adjustments
    sim_features['recency'] = sim_features['recency'] * (1 - recency_reduction / 100)
    sim_features['transaction'] = sim_features['transaction'] * (1 + transaction_increase / 100)
    
    # Recalculate CLV
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
    
    # Recalculate churn probability
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
    
    # Calculate simulated metrics
    churned_users = len(sim_features[sim_features['churn_probability'] > churn_threshold])
    churn_rate = churned_users / len(sim_features) if len(sim_features) > 0 else 0
    revenue_at_risk = (sim_features[sim_features['churn_probability'] > churn_threshold]['clv_estimate'] * 
                       sim_features[sim_features['churn_probability'] > churn_threshold]['churn_probability']).sum()
    avg_clv = sim_features['clv_estimate'].mean()
    
    return churned_users, churn_rate, revenue_at_risk, avg_clv

def main():
    st.title("🛍️ E-Commerce Churn Prediction Dashboard")
    st.markdown("*Retail Rocket Dataset Analysis*")
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    uploaded_file = st.sidebar.file_uploader("Upload events.csv", type=["csv"], help="Upload a CSV with columns: timestamp, visitorid, event, itemid, transactionid")
    sample_size = st.sidebar.selectbox(
        "Data Sample Size (for faster processing)",
        options=[100, 1000, 5000, 10000, 50000, "All data"],
        index=5,
        help="Select number of events to process"
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
    
    # Load data
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
            st.sidebar.error("⚠️ Using sample data due to loading issues.")
        
        # Default parameters
        avg_transaction_value = 100
        decay_factor = 120
        base_event_value = 20
        features = extract_features(df, min_events, avg_transaction_value, decay_factor, base_event_value)
        churn_probs = predict_churn(features)
        features['churn_probability'] = churn_probs
    
    # Display dataset date range
    st.sidebar.markdown(f"**Dataset Date Range:** {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    # Date range filter
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
    
    # Calculate base metrics
    total_users = len(features)
    churned_users = len(features[features['churn_probability'] > churn_threshold])
    churn_rate = churned_users / total_users if total_users > 0 else 0
    revenue_at_risk = (features[features['churn_probability'] > churn_threshold]['clv_estimate'] * 
                       features[features['churn_probability'] > churn_threshold]['churn_probability']).sum()
    avg_clv = features['clv_estimate'].mean()
    avg_recency = features['recency'].mean()
    
    # Executive Overview
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
            label="💰 Revenue at Risk",
            value=f"${revenue_at_risk:,.0f}",
            delta=f"${revenue_at_risk/churned_users if churned_users > 0 else 0:.0f} per user"
        )
    
    with col4:
        st.metric(
            label="📅 Avg Days Since Last Visit",
            value=f"{avg_recency:.1f}",
            delta=f"${avg_clv:.0f} avg CLV"
        )
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Churn Analysis", 
        "📈 Feature Importance", 
        "👥 User Segments", 
        "🔧 Simulation Tools",
        "📋 Data Explorer"
    ])
    
    with tab1:
        st.subheader("Churn Probability Distribution")
        st.write(f"**Churn Rate at Threshold {churn_threshold:.2f}:** {churn_rate:.1%}")
        
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
            # Dynamic bins based on desired 60%/25%/15% split
            churn_probs = features['churn_probability']
            q60 = churn_probs.quantile(0.60)  # 60th percentile for Low Risk
            q85 = churn_probs.quantile(0.85)  # 85th percentile for Medium Risk
            features['risk_segment'] = pd.cut(churn_probs, 
                                            bins=[0, q60, q85, 1.0],
                                            labels=['Low Risk', 'Medium Risk', 'High Risk'],
                                            include_lowest=True)
            
            segment_counts = features['risk_segment'].value_counts()
            fig_pie = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="User Risk Distribution",
                color_discrete_map={
                    'Low Risk': '#48cab2',
                    'Medium Risk': '#feca57',
                    'High Risk': '#ff6b6b'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("Churn Trends Over Time")
        df_with_churn = df.merge(features[['churn_probability']].reset_index(), 
                                left_on='visitorid', right_on='visitorid', how='left')
        df_with_churn['week'] = df_with_churn['timestamp'].dt.to_period('W')
        
        weekly_churn = df_with_churn.groupby('week').agg({
            'churn_probability': 'mean',
            'visitorid': 'nunique'
        }).reset_index()
        weekly_churn['week'] = weekly_churn['week'].dt.start_time
        
        fig_trend = px.line(
            weekly_churn,
            x='week',
            y='churn_probability',
            title="Average Churn Probability Over Time",
            labels={'churn_probability': 'Avg Churn Probability', 'week': 'Week'}
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Importance Analysis")
        
        numeric_features = ['recency', 'frequency', 'total_events', 'conversion_rate', 
                           'view', 'addtocart', 'transaction', 'unique_items',
                           'view_to_cart_rate', 'cart_abandonment_rate', 'events_per_session']
        
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
        
        st.subheader("Feature Distributions by Risk Level")
        
        selected_feature = st.selectbox(
            "Select feature to analyze:",
            options=numeric_features,
            index=0
        )
        
        fig_box = px.box(
            features,
            x='risk_segment',
            y=selected_feature,
            title=f"{selected_feature} Distribution by Risk Level",
            color='risk_segment',
            color_discrete_map={
                'Low Risk': '#48cab2',
                'Medium Risk': '#feca57',
                'High Risk': '#ff6b6b'
            }
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab3:
        st.subheader("User Segmentation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(
                features,
                x='clv_estimate',
                y='churn_probability',
                color='risk_segment',
                size='total_events',
                title="Customer Lifetime Value vs Churn Risk",
                labels={'clv_estimate': 'Estimated CLV ($)', 'churn_probability': 'Churn Probability'},
                color_discrete_map={
                    'Low Risk': '#48cab2',
                    'Medium Risk': '#feca57',
                    'High Risk': '#ff6b6b'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            fig_rf = px.scatter(
                features,
                x='recency',
                y='frequency',
                color='churn_probability',
                size='clv_estimate',
                title="RFM Analysis: Recency vs Frequency",
                labels={'recency': 'Days Since Last Visit', 'frequency': 'Activity Frequency'},
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_rf, use_container_width=True)
        
        st.subheader("High-Value At-Risk Customers")
        
        high_risk_valuable = features[
            (features['churn_probability'] > churn_threshold) & 
            (features['clv_estimate'] > features['clv_estimate'].quantile(0.75))
        ].sort_values('clv_estimate', ascending=False)
        
        if not high_risk_valuable.empty:
            st.dataframe(
                high_risk_valuable[['clv_estimate', 'churn_probability', 'recency', 'total_events', 'conversion_rate']].head(10),
                use_container_width=True
            )
        else:
            st.info("No high-value customers at risk with current threshold.")
    
    with tab4:
        st.subheader("Simulation & Intervention Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Revenue at Risk Simulation")
            
            st.markdown("**Adjust CLV Parameters**")
            sim_avg_transaction_value = st.number_input(
                "Average Transaction Value ($)",
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
                "Base Event Value for Non-Transaction Users ($)",
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
            
            # Run simulation
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
                "Simulated Revenue at Risk",
                f"${sim_revenue_at_risk:,.0f}",
                delta=f"${sim_revenue_at_risk/sim_churned_users if sim_churned_users > 0 else 0:.0f} per user"
            )
            st.metric(
                "Simulated Avg CLV",
                f"${sim_avg_clv:.0f}",
                delta=f"Base: ${avg_clv:.0f}"
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
                    name='Revenue at Risk ($)',
                    yaxis='y2'
                )
            )
            fig_threshold.update_layout(
                yaxis2=dict(title='Revenue at Risk ($)', overlaying='y', side='right')
            )
            st.plotly_chart(fig_threshold, use_container_width=True)
    
    with tab5:
        st.subheader("Data Explorer")
        
        st.markdown("#### Sample Events Data")
        st.dataframe(df_filtered.head(10), use_container_width=True)
        
        st.markdown("#### User Features")
        st.dataframe(features.head(10), use_container_width=True)
        
        st.markdown("#### Feature Statistics")
        st.write(features[['recency', 'frequency', 'conversion_rate', 'cart_abandonment_rate', 'clv_estimate', 'churn_probability']].describe())
        
        st.markdown("#### Transaction Statistics")
        st.write(f"Users with transactions: {len(features[features['transaction'] > 0]):,}")
        st.write(f"Total transactions: {features['transaction'].sum():,}")
        st.write(f"Average CLV: ${features['clv_estimate'].mean():,.2f}")
        st.write(f"Users above churn threshold: {churned_users:,}")
        st.write(f"Revenue at Risk (weighted): ${revenue_at_risk:,.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            high_risk_users = features[features['churn_probability'] > churn_threshold]
            csv_high_risk = high_risk_users.to_csv(index=True)
            
            st.download_button(
                label="📥 Download High-Risk Users",
                data=csv_high_risk,
                file_name=f"high_risk_users_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_all_features = features.to_csv(index=True)
            
            st.download_button(
                label="📥 Download All User Features",
                data=csv_all_features,
                file_name=f"user_features_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    st.markdown("**💡 Pro Tip:** Upload your own events.csv or adjust simulation parameters to explore Revenue at Risk scenarios!")

if __name__ == "__main__":
    main()