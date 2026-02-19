🚀 Live Demo:https://smartcart-clustering-app-app-fsdnjwefqkn3m9meurg6wz.streamlit.app/

📌 Project Overview

SmartCart is an intelligent customer segmentation system built using unsupervised machine learning.
The system analyses customer demographics, spending behaviour, and engagement patterns to group customers into meaningful clusters. 
These clusters help businesses create personalised marketing strategies, improve retention, and identify high‑value or churn‑risk customers early.

Problem Statement

SmartCart is a growing e‑commerce platform serving customers across multiple countries. 
The company collected extensive customer data including demographics, purchase behaviour, website activity, and feedback. However, SmartCart currently uses generic marketing strategies without understanding different customer behaviour patterns.

This leads to:

Inefficient marketing campaigns

Missed high‑value customer opportunities

Late identification of churn‑risk users

The goal is to build an intelligent customer segmentation system using clustering algorithms to support data‑driven decision‑making.

Dataset Description

The dataset contains 2240 customer records with 22 attributes, including:

1️⃣ Customer Demographics

Year_Birth

Education

Marital_Status

Income

Kidhome, Teenhome

Dt_Customer

2️⃣ Purchase Behaviour (Spending)

MntWines

MntFruits

MntMeatProducts

MntFishProducts

MntSweetProducts

MntGoldProds

3️⃣ Purchase Behaviour (Frequency)

NumWebPurchases

NumStorePurchases

NumDealsPurchases

NumCatalogPurchases

NumWebVisitsMonth

4️⃣ Customer Feedback

Recency

Complain

Response

⚙️ Technologies Used

Python

Pandas

NumPy

Scikit‑learn

Matplotlib & Seaborn

Streamlit

GitHub

🤖 Machine Learning Workflow

Data Cleaning & Missing Value Handling

Feature Engineering (Age, Total Spending, Customer Tenure)

Encoding Categorical Variables

Feature Scaling

Dimensionality Reduction using PCA

Clustering using KMeans & Agglomerative Clustering

Cluster Analysis & Visualisation

Streamlit Deployment for User Input Prediction

📈 Features of the App

User can enter customer details manually

Predicts customer cluster instantly

Helps identify high‑value customers

Supports personalised marketing strategies

Interactive Streamlit interface

📌 Future Improvements

Deploy trained model using .pkl file

Add customer recommendation system

Add dashboard analytics

Use DBSCAN / Hierarchical clustering comparison

Real‑time database integration
