# Amazon Sales and Review Analysis

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st
import warnings

warnings.filterwarnings("ignore")
nltk.download('vader_lexicon')

# Load data
data = pd.read_csv("amazon.csv")

# Preprocess columns
data['discounted_price'] = data['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
data['actual_price'] = data['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
data['discount_percentage'] = data['discount_percentage'].str.replace('%', '').astype(float)
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data['rating_count'] = data['rating_count'].str.replace(',', '').astype(float)
data.dropna(subset=['rating', 'rating_count'], inplace=True)
data['revenue'] = data['rating_count'] * data['discounted_price']
data['engagement'] = data['rating_count'] / data['discounted_price']

# Generate synthetic dates if missing
data['date'] = pd.date_range(start='2021-01-01', periods=len(data), freq='D')

# Streamlit app
st.set_page_config(page_title="Amazon Insights Dashboard", layout="wide", page_icon="📊")
st.title("🛍️ Amazon Product Analytics Dashboard")
st.markdown("""
<style>
    .reportview-container {
        background: #F8F9FA;
    }
    .sidebar .sidebar-content {
        background: #EAF2F8;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("🔍 Filter Data")
category = st.sidebar.multiselect("Select Category", options=data['category'].unique(), default=None)
if category:
    data = data[data['category'].isin(category)]

# Tabs
main_tabs = st.tabs(["Overview", "Clustering", "Time Series Forecasting", "Sentiment Analysis", "Market Basket", "Word Cloud", "Advanced Insights"])

with main_tabs[0]:
    st.subheader("📊 Dataset Overview")
    st.dataframe(data.head(20))
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products", len(data))
    col2.metric("Average Rating", round(data['rating'].mean(), 2))
    col3.metric("Total Revenue", f"${data['revenue'].sum():,.2f}")
    fig = px.histogram(data, x='rating', nbins=20, title="Rating Distribution", color_discrete_sequence=['indigo'])
    st.plotly_chart(fig, use_container_width=True)

with main_tabs[1]:
    st.subheader("🧠 Clustering on Discounted Prices + PCA")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    features = data[['discounted_price', 'actual_price', 'rating']]
    data['price_cluster'] = kmeans.fit_predict(features)

    pca = PCA(n_components=2)
    components = pca.fit_transform(features)
    data['PC1'], data['PC2'] = components[:, 0], components[:, 1]
    fig = px.scatter(data, x='PC1', y='PC2', color='price_cluster', title="PCA Clusters", color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

with main_tabs[2]:
    st.subheader("📈 Sales Forecasting")
    monthly = data.groupby(data['date'].dt.to_period('M')).agg({'revenue': 'sum'}).reset_index()
    monthly['date'] = monthly['date'].dt.to_timestamp()
    model = ExponentialSmoothing(monthly['revenue'], seasonal='add', seasonal_periods=12).fit()
    monthly['forecast'] = model.fittedvalues
    fig = px.line(monthly, x='date', y=['revenue', 'forecast'], title="Monthly Revenue vs Forecast")
    st.plotly_chart(fig, use_container_width=True)

with main_tabs[3]:
    st.subheader("🗣️ Sentiment Analysis on Reviews")
    if 'review_content' in data.columns:
        sid = SentimentIntensityAnalyzer()
        data['sentiment_score'] = data['review_content'].fillna('').apply(lambda x: sid.polarity_scores(x)['compound'])
        data['sentiment'] = data['sentiment_score'].apply(lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral')
        fig = px.pie(data, names='sentiment', title="Sentiment Distribution", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(data[['product_name', 'sentiment', 'sentiment_score']].sort_values(by='sentiment_score', ascending=False).head(10))
    else:
        st.warning("Review content column is missing.")

with main_tabs[4]:
    st.subheader("🛒 Market Basket Analysis")
    if 'user_id' in data.columns and 'category' in data.columns:
        basket = pd.crosstab(data['user_id'], data['category'])
        basket = basket.applymap(lambda x: x > 0).astype(bool)
        freq_items = apriori(basket, min_support=0.01, use_colnames=True)
        rules = association_rules(freq_items, metric='lift', min_threshold=1.0)
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))
    else:
        st.warning("Required columns missing for Market Basket Analysis.")

with main_tabs[5]:
    st.subheader("☁️ Word Cloud from Reviews")
    if 'review_content' in data.columns:
        text = " ".join(data['review_content'].dropna())
        wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("Review content column is missing.")

with main_tabs[6]:
    st.subheader("📌 Advanced Insights")
    st.markdown("### 🔎 Top Performing Products")
    top_products = data.sort_values(by='revenue', ascending=False).head(10)
    st.dataframe(top_products[['product_name', 'revenue', 'rating', 'rating_count']])

    st.markdown("### 🛍️ Engagement vs Revenue")
    fig = px.scatter(data, x='engagement', y='revenue', size='rating_count', color='rating', hover_data=['product_name'], title="Engagement vs Revenue")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | Advanced Analytics for Amazon Data Insights")