Imports and Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
import seaborn as sns
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from wordcloud import WordCloud
import plotly.express as px
import nltk
import warnings

Pandas: Handles data manipulation and analysis.

NumPy: Provides numerical operations, especially for arrays.

Matplotlib & Seaborn: Used for creating visualizations.

KMeans: Implements clustering algorithms from scikit-learn.

SentimentIntensityAnalyzer: Performs sentiment analysis using VADER.

Metrics (mean_squared_error, r2_score): Evaluate model performance.

Apriori and Association Rules: Perform market basket analysis using mlxtend.

ExponentialSmoothing: Implements time-series forecasting.

WordCloud: Generates word cloud visualizations.

Plotly Express: Creates interactive visualizations.

Warnings: Suppresses deprecation or irrelevant warnings.

nltk.download('vader_lexicon'): Downloads VADER lexicon for sentiment analysis.


Warning Suppression

warnings.filterwarnings("ignore", category=UserWarning)  # Suppresses irrelevant warnings from mlxtend
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)  # Suppresses matplotlib deprecation warnings
These lines prevent unnecessary warnings from cluttering the output.

Data Loading

data = pd.read_csv('amazon.csv')
Loads the dataset amazon.csv into a Pandas DataFrame named data.


Data Cleaning
Check for Required Columns:

required_columns = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']
if not all(col in data.columns for col in required_columns):
    raise KeyError("One or more required columns are missing from the dataset.")
Ensures all necessary columns exist in the dataset. If any are missing, it raises an error.

Clean and Convert Columns:

data['discounted_price'] = data['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
data['actual_price'] = data['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
data['discount_percentage'] = data['discount_percentage'].str.replace('%', '').astype(float)
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data['rating_count'] = data['rating_count'].str.replace(',', '').astype(float)
data.dropna(subset=['rating', 'rating_count'], inplace=True)
Removes symbols like ₹ and % from the price and discount columns, converting them to numeric types.
Converts rating_count to numeric by removing commas.
Drops rows with missing values in rating and rating_count.

Create New Features:

data['revenue'] = data['rating_count'] * data['discounted_price']
data['engagement'] = data['rating_count'] / data['discounted_price']
Revenue: Estimated revenue by multiplying rating_count with discounted_price.
Engagement: Measures user interaction (number of ratings per unit price).

1. Clustering Analysis

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['price_cluster'] = kmeans.fit_predict(data[['discounted_price']])
Uses the KMeans algorithm to group products into 3 clusters based on discounted_price.
Adds a new column, price_cluster, to store the cluster each product belongs to.

Visualize Clusters:

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='discounted_price', y='rating', hue='price_cluster', palette='viridis')
plt.title("Price Clusters and Ratings", fontsize=16)
plt.xlabel("Discounted Price (₹)")
plt.ylabel("Rating")
plt.show()
Creates a scatterplot showing the relationship between discounted_price and rating across clusters.

2. Time-Series Analysis
Handle Missing Dates:

if 'date' not in data.columns:
    print("The 'date' column is missing. Generating synthetic dates...")
    data['date'] = pd.date_range(start='2021-01-01', periods=len(data), freq='D')
If the date column is missing, generates synthetic dates starting from 2021-01-01.

Aggregate Monthly Sales:

data['date'] = pd.to_datetime(data['date'])
monthly_sales = data.groupby(data['date'].dt.to_period('M')).agg({'revenue': 'sum'}).reset_index()
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
Converts date to datetime format.
Groups data by month and calculates total revenue.

Forecast Sales:

model = ExponentialSmoothing(monthly_sales['revenue'], seasonal='add', seasonal_periods=12).fit()
monthly_sales['forecast'] = model.fittedvalues
Uses Exponential Smoothing for time-series forecasting.

Plot Sales and Forecast:

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['date'], monthly_sales['revenue'], label='Actual Sales')
plt.plot(monthly_sales['date'], monthly_sales['forecast'], label='Forecasted Sales', linestyle='--')
plt.legend()
plt.show()
Visualizes actual and forecasted sales trends.

3. Sentiment Analysis

if 'review_content' in data.columns:
    analyzer = SentimentIntensityAnalyzer()
    data['sentiment_score'] = data['review_content'].fillna('').apply(lambda x: analyzer.polarity_scores(x)['compound'])
    data['sentiment'] = data['sentiment_score'].apply(lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral')
else:
    print("The 'review_content' column is missing.")
Analyzes sentiment of review_content using VADER and classifies reviews as Positive, Negative, or Neutral.

4. Market Basket Analysis

if 'user_id' in data.columns and 'category' in data.columns:
    basket = pd.crosstab(data['user_id'], data['category'])
    basket = basket.applymap(lambda x: x > 0).astype(bool)
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    if not rules.empty:
        print("\nMarket Basket Analysis Rules:\n", rules)
    else:
        print("\nNo association rules found. Try lowering 'min_support' or checking the data.")
else:
    print("Required columns for market basket analysis are missing.")
Uses Apriori to find frequent itemsets and generate association rules.

5. Word Cloud

if 'review_content' in data.columns:
    review_text = " ".join(data['review_content'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(review_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Customer Reviews Word Cloud", fontsize=16)
    plt.show()
else:
    print("The 'review_content' column is missing.")
Creates a word cloud from review_content.

6. Automated Reporting

print("\n--- Executive Summary ---")
print("1. Price clusters reveal high-rating ranges for better pricing strategies.")
print("2. Time-series analysis predicts sales trends, enabling effective inventory planning.")
print("3. Sentiment analysis shows X% positive, Y% neutral, Z% negative reviews.")
print("4. Market basket analysis highlights product bundling opportunities.")
print("5. Enhanced feature engineering (revenue, engagement) enables precise decision-making.")
Provides a summary of the analysis for technical and non-technical stakeholders.