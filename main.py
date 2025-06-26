# Amazon Sales and Review Analysis (Ultra Advanced Version)

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
from sklearn.preprocessing import StandardScaler
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
data['discount_ratio'] = data['discounted_price'] / data['actual_price']

# Generate synthetic dates if missing
data['date'] = pd.date_range(start='2021-01-01', periods=len(data), freq='D')

# Streamlit app setup
st.set_page_config(page_title="Amazon Insights Dashboard", layout="wide", page_icon="📊")
st.title("🛍️ Amazon Product Analytics Dashboard")

# Custom CSS for better visuals
st.markdown("""
<style>
    .reportview-container {
        background: #FBFCFC;
    }
    .sidebar .sidebar-content {
        background: #EBF5FB;
    }
    h1, h2, h3, h4, h5 {
        color: #154360;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
category = st.sidebar.multiselect("Select Category", options=data['category'].unique(), default=data['category'].unique())
if category:
    data = data[data['category'].isin(category)]

# Tabs
main_tabs = st.tabs(["Overview", "Clustering", "Time Series Forecasting", "Sentiment Analysis", "Market Basket", "Word Cloud", "Advanced Insights"])

# Overview Tab
with main_tabs[0]:
    st.subheader("📊 Dataset Overview")
    st.dataframe(data.head(20))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", len(data))
    col2.metric("Average Rating", round(data['rating'].mean(), 2))
    col3.metric("Total Revenue", f"₹{data['revenue'].sum():,.0f}")
    col4.metric("Avg Discount %", f"{data['discount_percentage'].mean():.2f}%")
    fig = px.box(data, x='rating', y='discounted_price', title="Price Distribution by Rating")
    st.plotly_chart(fig, use_container_width=True)

# Clustering Tab
with main_tabs[1]:
    st.subheader("🧠 Clustering with PCA & Standardization")
    features = data[['discounted_price', 'actual_price', 'rating', 'rating_count', 'discount_ratio']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    data['price_cluster'] = kmeans.fit_predict(scaled_features)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_features)
    data['PC1'], data['PC2'] = components[:, 0], components[:, 1]
    fig = px.scatter(data, x='PC1', y='PC2', color='price_cluster', title="KMeans Clusters with PCA", color_continuous_scale='Turbo')
    st.plotly_chart(fig, use_container_width=True)

# Time Series Forecasting
with main_tabs[2]:
    st.subheader("📈 Revenue Trend Forecast")
    monthly = data.groupby(data['date'].dt.to_period('M')).agg({'revenue': 'sum'}).reset_index()
    monthly['date'] = monthly['date'].dt.to_timestamp()
    model = ExponentialSmoothing(monthly['revenue'], seasonal='add', seasonal_periods=12).fit()
    monthly['forecast'] = model.fittedvalues
    fig = px.line(monthly, x='date', y=['revenue', 'forecast'], title="Actual vs Forecasted Revenue")
    st.plotly_chart(fig, use_container_width=True)

# Sentiment Analysis
with main_tabs[3]:
    st.subheader("🗣️ Sentiment Analysis of Reviews")
    if 'review_content' in data.columns:
        sid = SentimentIntensityAnalyzer()
        data['sentiment_score'] = data['review_content'].fillna('').apply(lambda x: sid.polarity_scores(x)['compound'])
        data['sentiment'] = data['sentiment_score'].apply(lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral')
        sentiment_counts = data['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts, names=sentiment_counts.index, title="Sentiment Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(data[['product_name', 'sentiment', 'sentiment_score']].sort_values(by='sentiment_score', ascending=False).head(10))
    else:
        st.warning("Review content column is missing.")

# Market Basket
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

# Word Cloud
with main_tabs[5]:
    st.subheader("☁️ Word Cloud from Customer Reviews")
    if 'review_content' in data.columns:
        text = " ".join(data['review_content'].dropna())
        wc = WordCloud(width=1000, height=400, background_color='white', colormap='plasma').generate(text)
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("Review content column is missing.")

# Advanced Insights
with main_tabs[6]:
    st.subheader("📌 Advanced Product Insights")
    st.markdown("### 🔎 Top 10 Revenue Generators")
    top_products = data.sort_values(by='revenue', ascending=False).head(10)
    st.dataframe(top_products[['product_name', 'revenue', 'rating', 'rating_count']])

    st.markdown("### 🛍️ Engagement vs Revenue Analysis")
    fig = px.scatter(data, x='engagement', y='revenue', size='rating_count', color='rating', hover_data=['product_name'], title="Engagement vs Revenue")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔁 Discount Ratio vs Rating")
    fig = px.scatter(data, x='discount_ratio', y='rating', color='rating', size='rating_count', hover_data=['product_name'], title="Discount Ratio vs Rating")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | Ultra Advanced Analytics for Amazon Data Insights")