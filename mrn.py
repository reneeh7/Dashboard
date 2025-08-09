import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

# Amazon-style CSS
def load_amazon_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Amazon+Ember:wght@300;400;500;700&display=swap');
    
    .main .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    .amazon-header {
        background: linear-gradient(to bottom, #232F3E, #131A22);
        color: white;
        padding: 10px 20px;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 4px solid #FF9900;
    }
    
    .amazon-header h1 {
        color: white;
        font-family: 'Amazon Ember', Arial, sans-serif;
        font-weight: 400;
        font-size: 28px;
        margin: 0;
        display: flex;
        align-items: center;
    }
    
    .amazon-logo {
        color: #FF9900;
        margin-right: 15px;
        font-size: 32px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: #F3F3F3;
        border-radius: 4px;
        padding: 5px;
        margin-bottom: 20px;
        border: 1px solid #DDD;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        color: #232F3E;
        border-radius: 4px;
        font-weight: 500;
        font-family: 'Amazon Ember', Arial, sans-serif;
        border: 1px solid transparent;
        margin: 2px;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #FF9900 !important;
        color: white !important;
        border-color: #FF9900 !important;
    }
    
    .amazon-card {
        background: white;
        border: 1px solid #DDD;
        border-radius: 4px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: 'Amazon Ember', Arial, sans-serif;
    }
    
    .amazon-card h3 {
        color: #232F3E;
        font-size: 18px;
        font-weight: 500;
        margin-bottom: 15px;
        border-bottom: 1px solid #E7E7E7;
        padding-bottom: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(to bottom, #F7DFA5, #F0C14B);
        border: 1px solid #A88734 #9C7E31 #846A29;
        border-radius: 3px;
        color: #111;
        font-family: 'Amazon Ember', Arial, sans-serif;
        font-size: 13px;
        font-weight: 400;
        padding: 8px 15px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(to bottom, #F5D78E, #EDB932);
        border-color: #996633;
    }
    
    .amazon-metric {
        background: white;
        border: 1px solid #DDD;
        border-radius: 4px;
        padding: 15px;
        text-align: center;
        margin: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .amazon-metric h2 {
        color: #B12704;
        font-size: 24px;
        font-weight: 400;
        margin: 0;
        font-family: 'Amazon Ember', Arial, sans-serif;
    }
    
    .amazon-metric p {
        color: #565959;
        font-size: 12px;
        margin: 5px 0 0 0;
        font-family: 'Amazon Ember', Arial, sans-serif;
    }
    
    .css-1d391kg {
        background: #F3F3F3;
        border-right: 1px solid #DDD;
    }
    
    .product-card {
        background: white;
        border: 1px solid #DDD;
        border-radius: 4px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.2s;
        font-family: 'Amazon Ember', Arial, sans-serif;
    }
    
    .product-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        border-color: #FF9900;
    }
    
    .product-title {
        color: #007185;
        font-size: 14px;
        font-weight: 400;
        text-decoration: none;
        line-height: 1.3;
        margin-bottom: 8px;
    }
    
    .product-price {
        color: #B12704;
        font-size: 18px;
        font-weight: 400;
        margin: 5px 0;
    }
    
    .product-rating {
        color: #FF9900;
        font-size: 12px;
    }
    
    .stDataFrame {
        border: 1px solid #DDD;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .stAlert {
        border-radius: 4px;
        font-family: 'Amazon Ember', Arial, sans-serif;
    }
    
    .stSelectbox > div > div {
        background: white;
        border: 1px solid #DDD;
        border-radius: 3px;
        font-family: 'Amazon Ember', Arial, sans-serif;
    }
    
    .stTextInput > div > div > input {
        border: 1px solid #DDD;
        border-radius: 3px;
        font-family: 'Amazon Ember', Arial, sans-serif;
        padding: 8px 12px;
    }
    
    .stFileUploader {
        border: 2px dashed #DDD;
        border-radius: 4px;
        padding: 20px;
        text-align: center;
        background: #FAFAFA;
    }
    
    .email-template {
        background: #F9F9F9;
        border: 1px solid #DDD;
        border-radius: 4px;
        padding: 20px;
        margin: 15px 0;
        font-family: 'Amazon Ember', Arial, sans-serif;
    }
    
    .risk-high {
        background: #FFE6E6;
        border-left: 4px solid #D32F2F;
        padding: 10px;
        border-radius: 4px;
        margin: 5px 0;
    }
    
    .risk-medium {
        background: #FFF3E0;
        border-left: 4px solid #F57C00;
        padding: 10px;
        border-radius: 4px;
        margin: 5px 0;
    }
    
    .risk-low {
        background: #E8F5E8;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        border-radius: 4px;
        margin: 5px 0;
    }
    
    @media (max-width: 600px) {
        .amazon-card {
            padding: 10px;
        }
        .product-card {
            padding: 10px;
        }
        .amazon-header h1 {
            font-size: 20px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Database initialization
def init_database():
    conn = sqlite3.connect('amazon_churn.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visitor_id TEXT,
            email TEXT,
            subject TEXT,
            content TEXT,
            timestamp DATETIME,
            status TEXT,
            sector TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Mock data generators
def generate_mock_events(num_records=5000):
    events = []
    visitor_ids = [f"visitor_{i}" for i in range(1, 201)]
    item_ids = [f"item_{i}" for i in range(1, 501)]
    event_types = ['view', 'addtocart', 'transaction']

    start_date = datetime(2024, 4, 6)  # UK financial year start
    end_date = datetime(2025, 4, 5)   # UK financial year end

    # Fixed distribution: 10% High (20), 40% Medium (80), 50% Low (100)
    for i, visitor_id in enumerate(visitor_ids):
        if i < 20:  # High risk: 10% (20 customers)
            num_events = np.random.randint(1, 3)  # 1-2 events
            event_probs = [1.0, 0.0, 0.0]  # views only, no transactions
            days_offset = np.random.randint(61, 121)  # 61-120 days inactivity
        elif i < 100:  # Medium risk: 40% (80 customers)
            num_events = np.random.randint(3, 10)  # 3-10 events
            event_probs = [0.6, 0.3, 0.1]  # few transactions
            days_offset = np.random.randint(46, 62)  # 46-61 days inactivity
        else:  # Low risk: 50% (100 customers)
            num_events = np.random.randint(15, 50)  # 15-50 events
            event_probs = [0.4, 0.3, 0.3]  # more transactions
            days_offset = np.random.randint(1, 46)  # 1-45 days inactivity

        # Ensure the most recent activity sits exactly at the target inactivity window
        baseline_ts = end_date - timedelta(days=days_offset)
        # Create one event at the baseline (most recent)
        event_type = np.random.choice(event_types, p=event_probs)
        item_id = np.random.choice(item_ids)
        transaction_id = None
        if event_type == 'transaction':
            transaction_id = f"txn_{i}_{int(baseline_ts.timestamp())}"
        events.append({
            'timestamp': baseline_ts.strftime('%Y-%m-%d %H:%M:%S'),
            'visitorid': visitor_id,
            'event': event_type,
            'itemid': item_id,
            'transactionid': transaction_id
        })

        # Remaining events are older than baseline
        for _ in range(max(0, num_events - 1)):
            older_ts = baseline_ts - timedelta(days=np.random.randint(1, 31))
            event_type = np.random.choice(event_types, p=event_probs)
            item_id = np.random.choice(item_ids)
            transaction_id = None
            if event_type == 'transaction':
                transaction_id = f"txn_{i}_{int(older_ts.timestamp())}"
            events.append({
                'timestamp': older_ts.strftime('%Y-%m-%d %H:%M:%S'),
                'visitorid': visitor_id,
                'event': event_type,
                'itemid': item_id,
                'transactionid': transaction_id
            })

    return pd.DataFrame(events).sort_values('timestamp')

def generate_mock_item_properties():
    properties = []
    item_ids = [f"item_{i}" for i in range(1, 501)]
    brands = ['Amazon Basics', 'Samsung', 'Apple', 'Nike', 'Adidas', 'Sony', 'HP', 'Dell', 'Canon']
    
    for item_id in item_ids:
        base_time = datetime(2024, 4, 6) + timedelta(days=np.random.uniform(0, 365))
        
        properties.append({
            'timestamp': base_time.strftime('%Y-%m-%d %H:%M:%S'),
            'itemid': item_id,
            'property': 'categoryid',
            'value': f"cat_{np.random.choice(range(1, 21))}"
        })
        
        properties.append({
            'timestamp': base_time.strftime('%Y-%m-%d %H:%M:%S'),
            'itemid': item_id,
            'property': 'brand',
            'value': np.random.choice(brands)
        })
        
        properties.append({
            'timestamp': base_time.strftime('%Y-%m-%d %H:%M:%S'),
            'itemid': item_id,
            'property': 'price',
            'value': str(np.random.uniform(10, 1000))
        })
        
        properties.append({
            'timestamp': base_time.strftime('%Y-%m-%d %H:%M:%S'),
            'itemid': item_id,
            'property': 'available',
            'value': str(np.random.choice([0, 1], p=[0.1, 0.9]))
        })
    
    return pd.DataFrame(properties)

def generate_mock_category_tree():
    categories = [
        {'categoryid': 'cat_1', 'category_path': 'Electronics'},
        {'categoryid': 'cat_2', 'category_path': 'Electronics > Computers'},
        {'categoryid': 'cat_3', 'category_path': 'Electronics > Smartphones'},
        {'categoryid': 'cat_4', 'category_path': 'Electronics > Audio'},
        {'categoryid': 'cat_5', 'category_path': 'Fashion'},
        {'categoryid': 'cat_6', 'category_path': 'Fashion > Men'},
        {'categoryid': 'cat_7', 'category_path': 'Fashion > Women'},
        {'categoryid': 'cat_8', 'category_path': 'Books'},
        {'categoryid': 'cat_9', 'category_path': 'Books > Fiction'},
        {'categoryid': 'cat_10', 'category_path': 'Books > Non-Fiction'},
        {'categoryid': 'cat_11', 'category_path': 'Sports'},
        {'categoryid': 'cat_12', 'category_path': 'Sports > Fitness'},
        {'categoryid': 'cat_13', 'category_path': 'Sports > Outdoor'},
        {'categoryid': 'cat_14', 'category_path': 'Home & Garden'},
        {'categoryid': 'cat_15', 'category_path': 'Home & Garden > Kitchen'},
        {'categoryid': 'cat_16', 'category_path': 'Beauty'},
        {'categoryid': 'cat_17', 'category_path': 'Beauty > Skincare'},
        {'categoryid': 'cat_18', 'category_path': 'Automotive'},
        {'categoryid': 'cat_19', 'category_path': 'Automotive > Parts'},
        {'categoryid': 'cat_20', 'category_path': 'Automotive > Accessories'},
    ]
    return pd.DataFrame(categories)

def generate_sector_mapping(events_df):
    sectors = ['Premium', 'Standard', 'Budget', 'Enterprise', 'Student']
    visitor_ids = events_df['visitorid'].unique()
    mapping = []
    
    for visitor_id in visitor_ids:
        visitor_events = events_df[events_df['visitorid'] == visitor_id]
        event_count = len(visitor_events)
        sector = np.random.choice(sectors, p=[0.4, 0.3, 0.2, 0.05, 0.05]) if event_count >= 10 else np.random.choice(sectors)
        mapping.append({
            'visitor_id': visitor_id,
            'sector': sector,
            'name': f"Customer {visitor_id.split('_')[-1]}",
            'email': f"customer{visitor_id.split('_')[-1]}@example.com",
            'registration_date': (datetime(2024, 4, 6) - timedelta(days=np.random.uniform(30, 730))).strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(mapping)

# Churn prediction functions
def calculate_customer_features(events_df, visitor_id):
    customer_events = events_df[events_df['visitorid'] == visitor_id].copy()
    
    if customer_events.empty:
        return {
            'days_since_last_purchase': 100,
            'total_events': 0,
            'total_purchases': 0,
            'view': 0,
            'addtocart': 0,
            'transaction': 0,
            'recency_score': 0,
            'frequency_score': 0,
            'monetary_score': 0,
            'reason_for_churn': 'No activity recorded'
        }
    
    customer_events['timestamp'] = pd.to_datetime(customer_events['timestamp'])
    reference_date = pd.to_datetime(events_df['timestamp']).max()
    # Inactivity measured by last activity
    last_activity = customer_events['timestamp'].max()
    days_since_inactivity = min(100, max(1, (reference_date - last_activity).days))

    purchase_events = customer_events[customer_events['event'] == 'transaction']
    if purchase_events.empty:
        reason = 'No purchases recorded'
    else:
        # Keep reason aligned with inactivity windows for consistency in UI
        if days_since_inactivity > 60:
            reason = 'Long period since last activity (>60 days)'
        elif days_since_inactivity > 45:
            reason = 'Moderate period since last activity (>45 days)'
        else:
            reason = 'Recent activity (≤45 days)'
    
    total_events = len(customer_events)
    total_purchases = len(purchase_events)
    event_counts = customer_events['event'].value_counts()
    view_count = event_counts.get('view', 0)
    addtocart_count = event_counts.get('addtocart', 0)
    transaction_count = event_counts.get('transaction', 0)
    
    # Use inactivity-based days for recency
    days_since_last_purchase = days_since_inactivity
    recency_score = max(0, 100 - days_since_inactivity * 2)
    frequency_score = min(100, total_events * 2)
    monetary_score = min(100, total_purchases * 20)
    
    return {
        'days_since_last_purchase': days_since_last_purchase,
        'total_events': total_events,
        'total_purchases': total_purchases,
        'view': view_count,
        'addtocart': addtocart_count,
        'transaction': transaction_count,
        'recency_score': recency_score,
        'frequency_score': frequency_score,
        'monetary_score': monetary_score,
        'reason_for_churn': reason
    }

def predict_churn(events_df, sector_mapping_df):
    customers = []
    
    for visitor_id in events_df['visitorid'].unique():
        features = calculate_customer_features(events_df, visitor_id)
        
        churn_score = 0.0
        
        # Days component (0.3 or 0.2)
        if features['days_since_last_purchase'] > 60:
            churn_score += 0.3
        elif features['days_since_last_purchase'] > 45:
            churn_score += 0.2
        
        # Activity component (0.4 or 0.2)
        if features['total_events'] < 3:
            churn_score += 0.4
        elif features['total_events'] < 10:
            churn_score += 0.2
        
        # Purchase component (0.3 or 0.1)
        if features['total_purchases'] == 0:
            churn_score += 0.3
        elif features['total_purchases'] < 2:
            churn_score += 0.1
        
        churn_score = min(1.0, churn_score)
        
        # Determine risk level based on churn score
        if churn_score >= 0.7:
            risk_level = 'High'
        elif churn_score >= 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        customer_info = sector_mapping_df[sector_mapping_df['visitor_id'] == visitor_id]
        sector = customer_info['sector'].iloc[0] if not customer_info.empty else 'Unknown'
        name = customer_info['name'].iloc[0] if not customer_info.empty else f"Customer {visitor_id}"
        email = customer_info['email'].iloc[0] if not customer_info.empty else f"{visitor_id}@example.com"
        
        customers.append({
            'visitor_id': visitor_id,
            'name': name,
            'email': email,
            'sector': sector,
            'churn_score': churn_score,
            'risk_level': risk_level,
            **features
        })
    
    return pd.DataFrame(customers)

# Recommendation function
def generate_recommendations(events_df, item_properties_df, category_tree_df, visitor_id, top_n=5):
    customer_events = events_df[events_df['visitorid'] == visitor_id]
    
    if customer_events.empty:
        popular_items = events_df['itemid'].value_counts().head(top_n)
        recommendations = []
        
        for item_id in popular_items.index:
            if item_properties_df is not None and category_tree_df is not None:
                item_props = item_properties_df[item_properties_df['itemid'] == item_id]
                category_id = item_props[item_props['property'] == 'categoryid']['value'].iloc[0] if not item_props[item_props['property'] == 'categoryid'].empty else 'Unknown'
                brand = item_props[item_props['property'] == 'brand']['value'].iloc[0] if not item_props[item_props['property'] == 'brand'].empty else 'Generic'
                price = item_props[item_props['property'] == 'price']['value'].iloc[0] if not item_props[item_props['property'] == 'price'].empty else '0'
                
                category_info = category_tree_df[category_tree_df['categoryid'] == category_id]
                category = category_info['category_path'].iloc[0] if not category_info.empty else 'Unknown'
            else:
                category, brand, price = 'General', 'Generic', 0
            
            recommendations.append({
                'item_id': item_id,
                'category': category,
                'brand': brand,
                'price': float(price) if str(price).replace('.', '').isdigit() else 0,
                'reason': 'Trending Product',
                'score': popular_items[item_id] / popular_items.max() if not popular_items.empty else 0.5
            })
        
        return recommendations[:top_n]
    
    customer_items = customer_events['itemid'].unique()
    
    if item_properties_df is not None and category_tree_df is not None:
        customer_categories = []
        for item_id in customer_items:
            item_props = item_properties_df[item_properties_df['itemid'] == item_id]
            category_id = item_props[item_props['property'] == 'categoryid']['value'].iloc[0] if not item_props[item_props['property'] == 'categoryid'].empty else None
            if category_id:
                customer_categories.append(category_id)
        
        recommendations = []
        for category_id in set(customer_categories[:3]):
            category_items = item_properties_df[
                (item_properties_df['property'] == 'categoryid') & 
                (item_properties_df['value'] == category_id)
            ]['itemid'].unique()
            
            new_items = [item for item in category_items if item not in customer_items]
            
            item_popularity = events_df[events_df['itemid'].isin(new_items)]['itemid'].value_counts()
            
            for item_id in item_popularity.head(2).index:
                item_props = item_properties_df[item_properties_df['itemid'] == item_id]
                brand = item_props[item_props['property'] == 'brand']['value'].iloc[0] if not item_props[item_props['property'] == 'brand'].empty else 'Generic'
                price = item_props[item_props['property'] == 'price']['value'].iloc[0] if not item_props[item_props['property'] == 'price'].empty else '0'
                
                category_info = category_tree_df[category_tree_df['categoryid'] == category_id]
                category = category_info['category_path'].iloc[0] if not category_info.empty else 'Unknown'
                
                recommendations.append({
                    'item_id': item_id,
                    'category': category,
                    'brand': brand,
                    'price': float(price) if str(price).replace('.', '').isdigit() else 0,
                    'reason': f'Based on your interest in {category}',
                    'score': item_popularity[item_id] / item_popularity.max() if not item_popularity.empty else 0.5
                })
    else:
        related_visitors = events_df[events_df['itemid'].isin(customer_items)]['visitorid'].unique()
        related_items = events_df[
            (events_df['visitorid'].isin(related_visitors)) &
            (~events_df['itemid'].isin(customer_items))
        ]['itemid'].value_counts()
        
        recommendations = []
        for item_id in related_items.head(top_n).index:
            recommendations.append({
                'item_id': item_id,
                'category': 'General',
                'brand': 'Generic',
                'price': 0,
                'reason': 'Based on your activity',
                'score': related_items[item_id] / related_items.max() if not related_items.empty else 0.5
            })
    
    recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    return recommendations[:top_n]

# Email sending function
def send_email(to_email, subject, body):
    try:
        conn = sqlite3.connect('amazon_churn.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO email_logs (visitor_id, email, subject, content, timestamp, status, sector)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (to_email.split('@')[0], to_email, subject, body, datetime.now().isoformat(), 'Sent (Simulated)', 'Unknown'))
        conn.commit()
        conn.close()
        
        return True, "Email sent successfully (simulated)"
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

# Main dashboard
def main():
    load_amazon_css()
    init_database()
    
    st.markdown("""
    <div class="amazon-header">
        <h1><span class="amazon-logo">📦</span>Amazon Customer Intelligence Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize default data
    if 'events_df' not in st.session_state or st.session_state.events_df.empty:
        st.session_state.events_df = generate_mock_events()
        st.session_state.item_properties_df = generate_mock_item_properties()
        st.session_state.category_tree_df = generate_mock_category_tree()
        st.session_state.sector_mapping_df = generate_sector_mapping(st.session_state.events_df)
        st.session_state.churn_predictions = None
    
    st.info(f"📊 Using {'mock data' if 'item_properties_df' in st.session_state else 'uploaded data'} with {len(st.session_state.events_df):,} events for {st.session_state.events_df['visitorid'].nunique():,} customers")
    
    st.sidebar.markdown("### 🏪 Navigation")
    selected_tab = st.sidebar.radio(
        "Select Section:",
        ["📊 Dashboard & Churn Analysis", "📤 Data Upload", "💡 Recommendations", "📧 Email Campaigns", "📋 Reports & Logs"],
        label_visibility="collapsed"
    )
    
    if selected_tab == "📊 Dashboard & Churn Analysis":
        dashboard_churn_section()
    elif selected_tab == "📤 Data Upload":
        data_upload_section()
    elif selected_tab == "💡 Recommendations":
        recommendations_section()
    elif selected_tab == "📧 Email Campaigns":
        email_campaigns_section()
    elif selected_tab == "📋 Reports & Logs":
        reports_logs_section()

# Combined Dashboard and Churn Analysis section
def dashboard_churn_section():
    st.markdown("## 📊 Business Intelligence & Churn Analysis")
    st.markdown("**UK Financial Year 2024-25** (6 April 2024 - 5 April 2025)", unsafe_allow_html=True)
    
    if 'events_df' not in st.session_state or st.session_state.events_df.empty:
        st.warning("⚠️ No events data loaded. Please reset to default data or upload events.csv in the Data Upload section.")
        return
    
    # Filter data for UK financial year
    events_df = st.session_state.events_df.copy()
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    start_date = datetime(2024, 4, 6)
    end_date = datetime(2025, 4, 5)
    events_df = events_df[(events_df['timestamp'] >= start_date) & (events_df['timestamp'] <= end_date)]
    
    if events_df.empty:
        st.warning("⚠️ No events found for the UK Financial Year 2024-25. Resetting to default data.")
        st.session_state.events_df = generate_mock_events()
        st.session_state.item_properties_df = generate_mock_item_properties()
        st.session_state.category_tree_df = generate_mock_category_tree()
        st.session_state.sector_mapping_df = generate_sector_mapping(st.session_state.events_df)
        st.session_state.churn_predictions = None
        events_df = st.session_state.events_df.copy()
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
        events_df = events_df[(events_df['timestamp'] >= start_date) & (events_df['timestamp'] <= end_date)]
    
    # Churn predictions
    if st.session_state.churn_predictions is None:
        with st.spinner("🔍 Analyzing customer behavior patterns..."):
            st.session_state.churn_predictions = predict_churn(
                st.session_state.events_df, 
                st.session_state.sector_mapping_df
            )
    
    churn_df = st.session_state.churn_predictions
    
    # Validate risk level distribution
    risk_counts = churn_df['risk_level'].value_counts()
    high_risk = risk_counts.get('High', 0)
    medium_risk = risk_counts.get('Medium', 0)
    low_risk = risk_counts.get('Low', 0)
    
    if high_risk < 10 and 'item_properties_df' in st.session_state:
        st.warning("⚠️ Insufficient High Risk customers detected. Resetting mock data to ensure all risk levels.")
        st.session_state.events_df = generate_mock_events()
        st.session_state.sector_mapping_df = generate_sector_mapping(st.session_state.events_df)
        st.session_state.churn_predictions = predict_churn(
            st.session_state.events_df, 
            st.session_state.sector_mapping_df
        )
        events_df = st.session_state.events_df.copy()
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
        events_df = events_df[(events_df['timestamp'] >= start_date) & (events_df['timestamp'] <= end_date)]
        churn_df = st.session_state.churn_predictions
        risk_counts = churn_df['risk_level'].value_counts()
        high_risk = risk_counts.get('High', 0)
        medium_risk = risk_counts.get('Medium', 0)
        low_risk = risk_counts.get('Low', 0)
    
    # Key metrics
    st.markdown('<div class="amazon-card"><h3>📈 Key Metrics</h3></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = events_df['visitorid'].nunique()
        st.markdown(f"""
        <div class="amazon-metric">
            <h2>👥 {total_customers:,}</h2>
            <p>Total Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_events = len(events_df)
        st.markdown(f"""
        <div class="amazon-metric">
            <h2>📈 {total_events:,}</h2>
            <p>Total Events</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_purchases = len(events_df[events_df['event'] == 'transaction'])
        st.markdown(f"""
        <div class="amazon-metric">
            <h2>🛒 {total_purchases:,}</h2>
            <p>Total Purchases</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        conversion_rate = (total_purchases / total_events * 100) if total_events > 0 else 0
        st.markdown(f"""
        <div class="amazon-metric">
            <h2>🎯 {conversion_rate:.1f}%</h2>
            <p>Conversion Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="amazon-metric risk-high">
            <h2>{high_risk}</h2>
            <p>High Risk Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="amazon-metric risk-medium">
            <h2>{medium_risk}</h2>
            <p>Medium Risk Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="amazon-metric risk-low">
            <h2>{low_risk}</h2>
            <p>Low Risk Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_score = churn_df['churn_score'].mean()
        st.markdown(f"""
        <div class="amazon-metric">
            <h2>{avg_score:.2f}</h2>
            <p>Avg Churn Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Filters
    st.markdown('<div class="amazon-card"><h3>🔍 Filters</h3></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        events_df['month'] = events_df['timestamp'].dt.to_period('M').astype(str)
        available_months = sorted(events_df['month'].unique(), reverse=True)
        month_options = [m for m in available_months if '2024-04' <= m <= '2025-03']
        selected_month = st.selectbox("Select Month (2024-25 Financial Year):", month_options, index=0)
    
    with col2:
        sectors = st.session_state.sector_mapping_df['sector'].unique()
        selected_sectors = st.multiselect("Select Sectors:", sectors, default=sectors)
    
    with col3:
        risk_levels = ['High', 'Medium', 'Low']
        selected_risk_levels = st.multiselect("Select Risk Levels:", risk_levels, default=risk_levels)
    
    # Filter events and churn data
    filtered_events = events_df[events_df['month'] == selected_month]
    filtered_churn = churn_df[(churn_df['sector'].isin(selected_sectors)) & (churn_df['risk_level'].isin(selected_risk_levels))]
    
    # Month-wise activity chart
    st.markdown('<div class="amazon-card"><h3>📈 Monthly Activity Trend (2024-25)</h3></div>', unsafe_allow_html=True)
    monthly_data = events_df.groupby(['month', 'event']).size().unstack(fill_value=0).reset_index()
    monthly_data = monthly_data[monthly_data['month'].between('2024-04', '2025-03')]
    fig = px.line(monthly_data, x='month', y=['view', 'addtocart', 'transaction'],
                  color_discrete_sequence=['#FF9900', '#232F3E', '#146EB4'])
    fig.update_layout(height=300, xaxis_title="Month", yaxis_title="Number of Events",
                      legend_title="Event Type")
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk level bar graph
    st.markdown('<div class="amazon-card"><h3>👥 Customer Risk Levels</h3></div>', unsafe_allow_html=True)
    risk_counts = filtered_churn['risk_level'].value_counts().reindex(['High', 'Medium', 'Low'], fill_value=0)
    fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                 color=risk_counts.index,
                 color_discrete_map={'High': '#D32F2F', 'Medium': '#F57C00', 'Low': '#4CAF50'})
    fig.update_layout(height=300, xaxis_title="Risk Level", yaxis_title="Number of Customers")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector-wise risk level bar graph
    st.markdown('<div class="amazon-card"><h3>🏷️ Sector-wise Risk Levels</h3></div>', unsafe_allow_html=True)
    sector_risk = filtered_churn.groupby(['sector', 'risk_level']).size().unstack(fill_value=0)
    fig = px.bar(sector_risk, barmode='stack',
                 color_discrete_map={'High': '#D32F2F', 'Medium': '#F57C00', 'Low': '#4CAF50'})
    fig.update_layout(height=300, xaxis_title="Sector", yaxis_title="Number of Customers")
    st.plotly_chart(fig, use_container_width=True)
    
    # Activity summary for selected month
    st.markdown(f'<div class="amazon-card"><h3>📊 Activity for {selected_month}</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        active_customers = filtered_events['visitorid'].nunique()
        st.markdown(f"""
        <div class="amazon-metric">
            <h2>👥 {active_customers:,}</h2>
            <p>Active Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_events_filtered = len(filtered_events)
        st.markdown(f"""
        <div class="amazon-metric">
            <h2>📈 {total_events_filtered:,}</h2>
            <p>Total Events</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_purchases_filtered = len(filtered_events[filtered_events['event'] == 'transaction'])
        st.markdown(f"""
        <div class="amazon-metric">
            <h2>🛒 {total_purchases_filtered:,}</h2>
            <p>Total Purchases</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Activity table
    activity_summary = filtered_events.merge(
        st.session_state.sector_mapping_df[['visitor_id', 'name', 'sector']],
        left_on='visitorid',
        right_on='visitor_id',
        how='left'
    )
    activity_summary = activity_summary.groupby(['visitorid', 'name', 'sector', 'event']).size().unstack(fill_value=0).reset_index()
    activity_summary['Total Events'] = activity_summary[['view', 'addtocart', 'transaction']].sum(axis=1)
    display_columns = ['name', 'sector', 'view', 'addtocart', 'transaction', 'Total Events']
    
    st.markdown('<div class="amazon-card"><h3>👥 Customer Activity Details</h3></div>', unsafe_allow_html=True)
    st.dataframe(
        activity_summary[display_columns],
        use_container_width=True,
        column_config={
            'view': st.column_config.NumberColumn('Views'),
            'addtocart': st.column_config.NumberColumn('Cart Adds'),
            'transaction': st.column_config.NumberColumn('Purchases'),
            'Total Events': st.column_config.NumberColumn('Total Events')
        }
    )
    
    # Churn analysis table
    st.markdown(f'<div class="amazon-card"><h3>👥 Customer Risk Analysis (Filtered: {len(filtered_churn)} customers)</h3></div>', unsafe_allow_html=True)
    
    display_columns = ['name', 'email', 'sector', 'risk_level', 'churn_score', 
                       'days_since_last_purchase', 'total_events', 'total_purchases',
                       'view', 'addtocart', 'transaction', 'reason_for_churn']
    
    st.dataframe(
        filtered_churn[display_columns].round(3),
        use_container_width=True,
        column_config={
            'churn_score': st.column_config.NumberColumn('Churn Score', format="%.3f"),
            'risk_level': st.column_config.TextColumn('Risk Level'),
            'days_since_last_purchase': st.column_config.NumberColumn('Days Since Last Purchase'),
            'reason_for_churn': st.column_config.TextColumn('Reason for Churn'),
            'view': st.column_config.NumberColumn('Views'),
            'addtocart': st.column_config.NumberColumn('Cart Adds'),
            'transaction': st.column_config.NumberColumn('Purchases')
        }
    )
    
    if st.button("📥 Download Churn Analysis", type="primary"):
        csv = filtered_churn.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"churn_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# Data upload section
def data_upload_section():
    st.markdown("## 📤 Data Management")
    
    st.markdown('<div class="amazon-card"><h3>Upload Events Data</h3></div>', unsafe_allow_html=True)
    st.info("Upload events.csv with columns: timestamp (YYYY-MM-DD HH:MM:SS or Unix ms), visitorid, event (view/addtocart/transaction), itemid, transactionid")
    
    uploaded_events = st.file_uploader("Choose events.csv", type=['csv'], key="events")
    if uploaded_events:
        try:
            events_df = pd.read_csv(uploaded_events)
            required_columns = ['timestamp', 'visitorid', 'event', 'itemid']
            if not all(col in events_df.columns for col in required_columns):
                st.error(f"❌ events.csv must contain columns: {', '.join(required_columns)}")
                return
            valid_events = {'view', 'addtocart', 'transaction'}
            if not events_df['event'].isin(valid_events).all():
                st.error(f"❌ event column must contain only: {', '.join(valid_events)}")
                return
            try:
                events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
            except:
                try:
                    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], unit='ms')
                except:
                    st.error("❌ timestamp column must be in 'YYYY-MM-DD HH:MM:SS' or Unix milliseconds format")
                    return
            events_df['visitorid'] = events_df['visitorid'].astype(str)
            events_df['itemid'] = events_df['itemid'].astype(str)
            st.session_state.events_df = events_df
            st.session_state.item_properties_df = None
            st.session_state.category_tree_df = None
            st.session_state.sector_mapping_df = generate_sector_mapping(events_df)
            st.session_state.churn_predictions = None
            st.success(f"✅ Loaded {len(events_df):,} events from uploaded file")
            st.dataframe(events_df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error loading events.csv: {str(e)}")
    
    st.markdown('<div class="amazon-card"><h3>Reset Data</h3></div>', unsafe_allow_html=True)
    if st.button("🔄 Reset to Default Mock Data", type="primary"):
        with st.spinner("Loading default mock data..."):
            st.session_state.events_df = generate_mock_events(5000)
            st.session_state.item_properties_df = generate_mock_item_properties()
            st.session_state.category_tree_df = generate_mock_category_tree()
            st.session_state.sector_mapping_df = generate_sector_mapping(st.session_state.events_df)
            st.session_state.churn_predictions = None
        st.success("✅ Default mock data loaded!")

# Recommendations section
def recommendations_section():
    st.markdown("## 💡 Personalized Product Recommendations")
    
    if 'events_df' not in st.session_state or st.session_state.events_df.empty:
        st.warning("⚠️ No events data loaded. Please reset to default data or upload events.csv in the Data Upload section.")
        return
    
    customer_options = st.session_state.sector_mapping_df['visitor_id'].tolist()
    selected_customer = st.selectbox("👤 Select Customer:", customer_options)
    
    if selected_customer:
        customer_info = st.session_state.sector_mapping_df[
            st.session_state.sector_mapping_df['visitor_id'] == selected_customer
        ].iloc[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="amazon-card">
                <h3>👤 Customer Profile</h3>
                <p><strong>Name:</strong> {customer_info['name']}</p>
                <p><strong>Email:</strong> {customer_info['email']}</p>
                <p><strong>Sector:</strong> {customer_info['sector']}</p>
                <p><strong>Registration:</strong> {customer_info['registration_date']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            customer_events = st.session_state.events_df[
                st.session_state.events_df['visitorid'] == selected_customer
            ]
            
            st.markdown(f"""
            <div class="amazon-card">
                <h3>📊 Activity Summary</h3>
                <p><strong>Total Events:</strong> {len(customer_events)}</p>
                <p><strong>Purchases:</strong> {len(customer_events[customer_events['event'] == 'transaction'])}</p>
            </div>
            """, unsafe_allow_html=True)
        
        recommendations = generate_recommendations(
            st.session_state.events_df,
            st.session_state.item_properties_df,
            st.session_state.category_tree_df,
            selected_customer
        )
        
        st.markdown('<div class="amazon-card"><h3>🛍️ Recommended Products</h3></div>', unsafe_allow_html=True)
        
        for i, rec in enumerate(recommendations):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                price_display = f"£{rec['price']:.2f}" if rec['price'] > 0 else "N/A"
                st.markdown(f"""
                <div class="product-card">
                    <div class="product-title">🎁 {rec['item_id']} - {rec['brand']}</div>
                    <div class="product-price">{price_display}</div>
                    <div class="product-rating">⭐⭐⭐⭐⭐ ({int(rec['score'] * 100)}% match)</div>
                    <p style="color: #565959; font-size: 12px; margin-top: 8px;">
                        📂 {rec['category']}<br>
                        💡 {rec['reason']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button(f"🛒 Add to Campaign", key=f"add_{i}"):
                    st.success(f"✅ {rec['item_id']} added to email campaign!")

# Email campaigns section
def email_campaigns_section():
    st.markdown("## 📧 Automated Email Campaigns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="amazon-card"><h3>🎯 Campaign Targeting</h3></div>', unsafe_allow_html=True)
        
        target_sectors = st.multiselect(
            "Select Target Sectors:",
            options=st.session_state.sector_mapping_df['sector'].unique(),
            default=['Premium']
        )
        
        risk_levels = st.multiselect(
            "Select Risk Levels:",
            options=['High', 'Medium', 'Low'],
            default=['High', 'Medium']
        )
        
        if st.session_state.churn_predictions is not None:
            target_customers = st.session_state.churn_predictions[
                (st.session_state.churn_predictions['sector'].isin(target_sectors)) &
                (st.session_state.churn_predictions['risk_level'].isin(risk_levels))
            ]
            
            st.info(f"📊 Target Audience: {len(target_customers)} customers")
        else:
            st.warning("⚠️ Please run churn analysis first")
            target_customers = pd.DataFrame()
    
    with col2:
        st.markdown('<div class="amazon-card"><h3>📧 Email Configuration</h3></div>', unsafe_allow_html=True)
        
        email_subject = st.text_input("Email Subject:", value="We Miss You! Special Offers Inside 🎁")
        
        sender_name = st.text_input("Sender Name:", value="Amazon Customer Service")
        
        campaign_name = st.text_input("Campaign Name:", value=f"Churn_Campaign_{datetime.now().strftime('%Y%m%d')}")
    
    st.markdown('<div class="amazon-card"><h3>✏️ Email Template Editor</h3></div>', unsafe_allow_html=True)
    
    default_template = """
    <div style="font-family: Amazon Ember, Arial, sans-serif; max-width: 600px; margin: 0 auto; background: white;">
        <div style="background: linear-gradient(to right, #232F3E, #131A22); padding: 20px; text-align: center;">
            <h1 style="color: #FF9900; margin: 0; font-size: 28px;">📦 Amazon</h1>
        </div>
        
        <div style="padding: 30px 20px;">
            <h2 style="color: #232F3E; font-size: 24px;">Hi {customer_name},</h2>
            
            <p style="color: #111; font-size: 16px; line-height: 1.5;">
                We noticed you haven't visited us recently, and we miss you! 
                As one of our valued {sector} customers, we have some special recommendations just for you.
            </p>
            
            <div style="background: #F7F7F7; padding: 20px; margin: 20px 0; border-left: 4px solid #FF9900;">
                <h3 style="color: #232F3E; margin-top: 0;">🎯 Personalized Recommendations</h3>
                <p style="color: #111;">Based on your interests, you might like:</p>
                <ul style="color: #111;">
                    {recommendations}
                </ul>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="#" style="background: linear-gradient(to bottom, #F7DFA5, #F0C14B); 
                   border: 1px solid #A88734; border-radius: 3px; color: #111; 
                   padding: 12px 24px; text-decoration: none; font-weight: bold;">
                    🛍️ Shop Now
                </a>
            </div>
            
            <p style="color: #565959; font-size: 14px; margin-top: 30px;">
                Thank you for being a loyal Amazon customer. We look forward to serving you again soon!
            </p>
            
            <p style="color: #565959; font-size: 14px;">
                Best regards,<br>
                The Amazon Team
            </p>
        </div>
        
        <div style="background: #232F3E; padding: 15px 20px; text-align: center; color: #DDD; font-size: 12px;">
            This email was sent to {customer_email}. 
            <a href="#" style="color: #FF9900;">Unsubscribe</a> | 
            <a href="#" style="color: #FF9900;">Privacy Policy</a>
        </div>
    </div>
    """
    
    email_template = st.text_area(
        "Email Template (HTML):",
        value=default_template,
        height=400,
        help="Use placeholders: {customer_name}, {customer_email}, {sector}, {recommendations}"
    )
    
    if st.button("👀 Preview Email"):
        if not target_customers.empty:
            sample_customer = target_customers.iloc[0]
            
            recommendations = generate_recommendations(
                st.session_state.events_df,
                st.session_state.item_properties_df,
                st.session_state.category_tree_df,
                sample_customer['visitor_id'],
                3
            )
            
            rec_html = ""
            for rec in recommendations:
                price_display = f"£{rec['price']:.2f}" if rec['price'] > 0 else "N/A"
                rec_html += f"<li>{rec['brand']} - {rec['item_id']} ({price_display})</li>"
            
            preview_html = email_template.format(
                customer_name=sample_customer['name'],
                customer_email=sample_customer['email'],
                sector=sample_customer['sector'],
                recommendations=rec_html
            )
            
            st.markdown("### 📧 Email Preview")
            st.markdown(preview_html, unsafe_allow_html=True)
    
    st.markdown('<div class="amazon-card"><h3>🚀 Launch Campaign</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Send Campaign", type="primary"):
            if not target_customers.empty:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                sent_count = 0
                failed_count = 0
                
                for idx, customer in target_customers.iterrows():
                    recommendations = generate_recommendations(
                        st.session_state.events_df,
                        st.session_state.item_properties_df,
                        st.session_state.category_tree_df,
                        customer['visitor_id'],
                        3
                    )
                    
                    rec_html = ""
                    for rec in recommendations:
                        price_display = f"£{rec['price']:.2f}" if rec['price'] > 0 else "N/A"
                        rec_html += f"<li>{rec['brand']} - {rec['item_id']} ({price_display})</li>"
                    
                    personalized_email = email_template.format(
                        customer_name=customer['name'],
                        customer_email=customer['email'],
                        sector=customer['sector'],
                        recommendations=rec_html
                    )
                    
                    success, message = send_email(
                        customer['email'],
                        email_subject,
                        personalized_email
                    )
                    
                    if success:
                        sent_count += 1
                    else:
                        failed_count += 1
                    
                    progress = (idx + 1) / len(target_customers)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {idx + 1}/{len(target_customers)} customers")
                
                st.success(f"✅ Campaign completed! Sent: {sent_count}, Failed: {failed_count}")
            else:
                st.error("❌ No target customers selected")
    
    with col2:
        if st.button("💾 Save Template"):
            st.success("✅ Template saved successfully!")
    
    with col3:
        if st.button("📋 View Campaign History"):
            st.info("📊 Campaign history feature coming soon!")

# Reports and logs section
def reports_logs_section():
    st.markdown("## 📋 Reports & Activity Logs")
    
    tabs = st.tabs(["📧 Email Logs", "👥 Customer Activity"])
    
    with tabs[0]:
        st.markdown("### 📧 Email Campaign Logs")
        
        conn = sqlite3.connect('amazon_churn.db')
        try:
            email_logs = pd.read_sql_query("SELECT * FROM email_logs ORDER BY timestamp DESC", conn)
            
            if not email_logs.empty:
                st.dataframe(email_logs, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    total_emails = len(email_logs)
                    st.metric("Total Emails", total_emails)
                
                with col2:
                    sent_emails = len(email_logs[email_logs['status'].str.contains('Sent', na=False)])
                    st.metric("Sent Successfully", sent_emails)
            else:
                st.info("📭 No email logs found. Send some campaigns to see data here.")
        
        except Exception as e:
            st.error(f"❌ Error loading email logs: {str(e)}")
        finally:
            conn.close()
    
    with tabs[1]:
        st.markdown("### 👥 Customer Activity Tracking")
        
        if 'events_df' not in st.session_state or st.session_state.events_df.empty:
            st.warning("⚠️ No events data loaded. Please reset to default data or upload events.csv in the Data Upload section.")
            return
        
        recent_events = st.session_state.events_df.sort_values('timestamp', ascending=False).head(100)
        
        recent_events_with_names = recent_events.merge(
            st.session_state.sector_mapping_df[['visitor_id', 'name', 'sector']], 
            left_on='visitorid', 
            right_on='visitor_id', 
            how='left'
        )
        
        display_cols = ['timestamp', 'name', 'sector', 'event', 'itemid']
        st.dataframe(recent_events_with_names[display_cols], use_container_width=True)
        
        st.markdown("### 📥 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Churn Analysis"):
                if st.session_state.churn_predictions is not None:
                    csv = st.session_state.churn_predictions.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Churn Data",
                        data=csv,
                        file_name=f"churn_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📈 Export Activity Data"):
                csv = st.session_state.events_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Activity Data",
                    data=csv,
                    file_name=f"customer_activity_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
    