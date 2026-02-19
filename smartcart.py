# ================================
# SMARTCART CUSTOMER CLUSTERING APP
# ================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="SmartCart Clustering", layout="centered")

st.title("🛒 SmartCart Customer Clustering System")

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("smartcart_customers.csv")
    return df

df = load_data()

st.write("Dataset Shape:", df.shape)

# ================================
# SELECT FEATURES
# ================================
features = [
    'Income','MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds',
    'NumWebPurchases','NumStorePurchases','Recency'
]

X = df[features]

# ================================
# FIX MISSING VALUES  🔥 IMPORTANT
# ================================
X = X.fillna(X.median())

# ================================
# SCALE DATA
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# TRAIN MODEL
# ================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# ================================
# SHOW CLUSTER DISTRIBUTION
# ================================
st.subheader("Cluster Distribution")

df["Cluster"] = kmeans.labels_

fig, ax = plt.subplots()
sns.countplot(x=df["Cluster"], ax=ax)
st.pyplot(fig)

# ================================
# USER INPUT
# ================================
st.subheader("👉 Enter Customer Details")

income = st.number_input("Income", 0)
wine = st.number_input("Wine Spending", 0)
fruit = st.number_input("Fruit Spending", 0)
meat = st.number_input("Meat Spending", 0)
fish = st.number_input("Fish Spending", 0)
sweet = st.number_input("Sweet Spending", 0)
gold = st.number_input("Gold Spending", 0)
web = st.number_input("Web Purchases", 0)
store = st.number_input("Store Purchases", 0)
recency = st.number_input("Days Since Last Purchase", 0)

# ================================
# PREDICT
# ================================
if st.button("Predict Customer Cluster"):

    user_data = np.array([[income,wine,fruit,meat,fish,sweet,gold,web,store,recency]])
    user_scaled = scaler.transform(user_data)
    cluster = kmeans.predict(user_scaled)

    st.success(f"✅ Customer belongs to Cluster: {cluster[0]}")

# ================================
# SHOW DATA
# ================================
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())
