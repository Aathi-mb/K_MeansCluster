import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Page title
st.title("🛍️ Mall Customers Segmentation using KMeans")

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select number of clusters
k = st.slider("Select Number of Clusters (K)", 2, 6, 3)

# Train KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Show clustered data
st.subheader("Clustered Customers")
st.dataframe(df)

# Plot clusters
st.subheader("Customer Segmentation Visualization")

fig, ax = plt.subplots()
scatter = ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
st.pyplot(fig)

# User input section
st.subheader("🔍 Find Customer Cluster")

income = st.number_input("Enter Annual Income (k$)", 10, 150, 50)
score = st.number_input("Enter Spending Score", 1, 100, 50)

if st.button("Predict Cluster"):
    input_scaled = scaler.transform([[income, score]])
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"✅ Customer belongs to Cluster {cluster}")
