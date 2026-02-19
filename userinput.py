import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="SmartCart Customer Segmentation", layout="centered")

st.title("🛒 SmartCart Customer Segmentation System")
st.write("Enter customer details to find customer cluster")

# ----------------------------
# USER INPUT
# ----------------------------

st.subheader("Customer Demographics")

income = st.number_input("Income", 0, 1000000, 50000)
kidhome = st.number_input("Kids at Home", 0, 5, 1)
teenhome = st.number_input("Teenagers at Home", 0, 5, 0)

st.subheader("Purchase Behaviour")

mnt_wines = st.number_input("Amount Spent on Wine", 0, 5000, 100)
mnt_meat = st.number_input("Amount Spent on Meat", 0, 5000, 100)
mnt_fruits = st.number_input("Amount Spent on Fruits", 0, 5000, 50)
mnt_gold = st.number_input("Amount Spent on Gold", 0, 5000, 20)

st.subheader("Purchase Frequency")

web_purchases = st.number_input("Web Purchases", 0, 50, 5)
store_purchases = st.number_input("Store Purchases", 0, 50, 3)
web_visits = st.number_input("Website Visits per Month", 0, 50, 10)

recency = st.number_input("Days Since Last Purchase", 0, 1000, 30)

# ----------------------------
# LOAD DATA
# ----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("smartcart_data.csv")  # Upload dataset to GitHub
    return df

df = load_data()

features = [
    'Income','Kidhome','Teenhome',
    'MntWines','MntMeatProducts','MntFruits','MntGoldProds',
    'NumWebPurchases','NumStorePurchases','NumWebVisitsMonth',
    'Recency'
]

X = df[features]

# ----------------------------
# MODEL TRAINING
# ----------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_pca)

# ----------------------------
# USER PREDICTION
# ----------------------------

if st.button("Predict Customer Cluster"):

    user = np.array([[income,kidhome,teenhome,
                      mnt_wines,mnt_meat,mnt_fruits,mnt_gold,
                      web_purchases,store_purchases,web_visits,
                      recency]])

    user_scaled = scaler.transform(user)
    user_pca = pca.transform(user_scaled)
    cluster = kmeans.predict(user_pca)

    st.success(f"🎯 Customer belongs to Cluster: {cluster[0]}")

    if cluster[0] == 0:
        st.info("Low engagement customer")
    elif cluster[0] == 1:
        st.info("High value loyal customer")
    elif cluster[0] == 2:
        st.info("Discount-seeking customer")
    else:
        st.info("Churn-risk customer")
