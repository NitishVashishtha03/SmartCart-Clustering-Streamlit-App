# =====================================
# SmartCart Customer Clustering App
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SmartCart Clustering", layout="wide")

st.title("🛒 SmartCart Customer Clustering System")

# =====================================
# 1️⃣ LOAD DATA
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("smartcard_customers.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =====================================
# 2️⃣ DATA CLEANING
# =====================================
df["Income"] = df["Income"].fillna(df["Income"].median())

features = [
    'Income','MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds',
    'NumWebPurchases','NumStorePurchases','Recency'
]

df_model = df[features].copy()
df_model = df_model.fillna(df_model.median())

# =====================================
# 3️⃣ SCALING
# =====================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)

# =====================================
# 4️⃣ TRAIN MODEL
# =====================================
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

df_model["Cluster"] = kmeans.labels_

# =====================================
# 5️⃣ VISUALIZATION
# =====================================
st.subheader("Customer Count per Cluster")

fig1, ax1 = plt.subplots()
sns.countplot(data=df_model, x="Cluster", hue="Cluster", ax=ax1)
st.pyplot(fig1)

st.subheader("Income vs Spending")

fig2, ax2 = plt.subplots()
sns.scatterplot(data=df_model, x="Income", y="Recency", hue="Cluster", ax=ax2)
st.pyplot(fig2)

# =====================================
# 6️⃣ USER INPUT
# =====================================
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

if st.button("Predict Customer Cluster"):

    user_data = np.array([[income,wine,fruit,meat,fish,sweet,gold,web,store,recency]])
    user_scaled = scaler.transform(user_data)
    cluster = kmeans.predict(user_scaled)

    st.success(f"Customer belongs to Cluster: {cluster[0]}")

    if cluster[0] == 0:
        st.info("💡 Low spending customer")
    elif cluster[0] == 1:
        st.info("💡 Medium spending customer")
    elif cluster[0] == 2:
        st.info("💡 High value loyal customer")
    else:
        st.info("💡 New or irregular customer")

# =====================================
# 7️⃣ CLUSTER SUMMARY
# =====================================
st.subheader("Cluster Summary")
summary = df_model.groupby("Cluster").mean()
st.dataframe(summary)
