# ==========================================
# SmartCart Customer Clustering Streamlit App
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SmartCart Clustering", layout="wide")

st.title("🛒 SmartCart Customer Clustering System")

# ==========================================
# 1️⃣ LOAD DATA (⚠️ CHECK DATASET NAME HERE)
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("smartcart_customers.csv")  # 👈 CHANGE NAME IF DIFFERENT
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==========================================
# 2️⃣ DATA CLEANING
# ==========================================
if "Income" in df.columns:
    df["Income"] = df["Income"].fillna(df["Income"].median())

# Features used for clustering
features = [
    'Income','MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds',
    'NumWebPurchases','NumStorePurchases','Recency'
]

# Keep only available features
features = [f for f in features if f in df.columns]

df_model = df[features].copy()
df_model = df_model.fillna(df_model.median())

# ==========================================
# 3️⃣ SCALE DATA
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)

# ==========================================
# 4️⃣ TRAIN KMEANS MODEL
# ==========================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df_model["Cluster"] = kmeans.labels_

# ==========================================
# 5️⃣ VISUALIZATION (NO 3D ERROR)
# ==========================================
st.subheader("Customer Count per Cluster")

fig1, ax1 = plt.subplots()
sns.countplot(data=df_model, x="Cluster", ax=ax1)
st.pyplot(fig1)

st.subheader("Income vs Recency by Cluster")

if "Income" in df_model.columns and "Recency" in df_model.columns:
    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        data=df_model,
        x="Income",
        y="Recency",
        hue="Cluster",
        ax=ax2
    )
    st.pyplot(fig2)

# ==========================================
# 6️⃣ USER INPUT SECTION
# ==========================================
st.subheader("👉 Enter Customer Details")

user_inputs = []

for feature in features:
    value = st.number_input(f"{feature}", 0)
    user_inputs.append(value)

# ==========================================
# 7️⃣ PREDICTION
# ==========================================
if st.button("Predict Customer Cluster"):

    user_array = np.array([user_inputs])
    user_scaled = scaler.transform(user_array)
    cluster = kmeans.predict(user_scaled)

    st.success(f"Customer belongs to Cluster: {cluster[0]}")

    if cluster[0] == 0:
        st.info("💡 Budget / Low Spending Customer")
    elif cluster[0] == 1:
        st.info("💡 Moderate Spending Customer")
    elif cluster[0] == 2:
        st.info("💡 High Value Loyal Customer")
    else:
        st.info("💡 Occasional / New Customer")

# ==========================================
# 8️⃣ CLUSTER SUMMARY
# ==========================================
st.subheader("Cluster Summary (Average Values)")
summary = df_model.groupby("Cluster").mean()
st.dataframe(summary)
