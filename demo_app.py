
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(layout="wide", page_title="Customer Segmentation")

st.title("Wholesale Customer Segmentation Analysis")
st.markdown("*Data Mining Unsupervised Learning - Final Evaluation Project*")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("final_results_with_clusters.csv")

df = load_data()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select Page:", ["Dashboard", "Clusters", "Insights", "Prediction"])

spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

if page == "Dashboard":
    st.header("Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Clusters Found", df["Cluster"].nunique())
    col3.metric("Average Spending", f"${df[spending_cols].sum(axis=1).mean():,.0f}")
    col4.metric("HoReCa Customers", (df["Channel"]==1).sum())
    
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_dist = df["Cluster"].value_counts().reset_index()
        cluster_dist.columns = ["Cluster", "Count"]
        fig = px.bar(cluster_dist, x="Cluster", y="Count", 
                    title="Customer Distribution by Cluster")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        channel_dist = pd.crosstab(df["Cluster"], df["Channel"])
        fig = px.bar(channel_dist, x=channel_dist.index, y=[channel_dist[1], channel_dist[2]],
                    title="Channel Distribution by Cluster",
                    labels={'value': 'Customers', 'x': 'Cluster', 'variable': 'Channel'})
        st.plotly_chart(fig, use_container_width=True)

elif page == "Clusters":
    st.header("Cluster Analysis")
    
    selected_cluster = st.selectbox("Select Cluster:", sorted(df["Cluster"].unique()))
    cluster_data = df[df["Cluster"] == selected_cluster]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Cluster Size", len(cluster_data))
        st.metric("Avg Total Spending", f"${cluster_data[spending_cols].sum(axis=1).mean():,.0f}")
    
    with col2:
        horeca_pct = (cluster_data["Channel"] == 1).sum() / len(cluster_data) * 100
        st.metric("HoReCa %", f"{horeca_pct:.1f}%")
        st.metric("Retail %", f"{100-horeca_pct:.1f}%")
    
    # Spending profile
    spending_profile = cluster_data[spending_cols].mean()
    fig = px.bar(x=spending_profile.index, y=spending_profile.values,
                title=f"Cluster {selected_cluster} - Spending Profile")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Insights":
    st.header("Business Insights")
    st.markdown("""
    ### Key Findings:
    1. **Optimal clusters identified**: """ + str(optimal_k) + """ customer segments
    2. **Model quality**: Silhouette Score = """ + str(round(silhouette_final, 4)) + """ (Excellent)
    3. **Channel patterns**: HoReCa focuses on Fresh products
    4. **Correlations**: Grocery-Milk-Detergents highly correlated (0.75+)
    
    ### Actionable Recommendations:
    - High-spending clusters: Premium service tier
    - Category-focused: Targeted promotions
    - HoReCa-heavy: Fresh product inventory priority
    - Retail-heavy: Grocery and staples focus
    """)

elif page == "Prediction":
    st.header("New Customer Prediction")
    st.markdown("Enter annual spending to predict segment:")
    
    col1, col2, col3 = st.columns(3)
    fresh = col1.number_input("Fresh ($)", 0, 50000, 10000)
    milk = col2.number_input("Milk ($)", 0, 30000, 8000)
    grocery = col3.number_input("Grocery ($)", 0, 30000, 9000)
    
    col4, col5, col6 = st.columns(3)
    frozen = col4.number_input("Frozen ($)", 0, 20000, 3000)
    detergents = col5.number_input("Detergents ($)", 0, 20000, 4000)
    delicassen = col6.number_input("Delicassen ($)", 0, 15000, 2000)
    
    if st.button("Predict Cluster", type="primary"):
        # Create input data
        new_customer = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])
        new_customer_scaled = scaler.transform(new_customer)
        prediction = kmeans_final.predict(new_customer_scaled)[0]
        total_spending = new_customer.sum()
        
        st.success(f"Predicted Cluster: **Cluster {prediction}**")
        st.info(f"Total Annual Spending: ${total_spending[0]:,.0f}")
        
        # Compare with cluster average
        cluster_avg = cluster_centers_original[prediction]
        comparison = pd.DataFrame({
            'Customer': new_customer[0],
            'Cluster Avg': cluster_avg,
            'Difference': new_customer[0] - cluster_avg
        }, index=spending_cols)
        
        st.dataframe(comparison.round(0))
