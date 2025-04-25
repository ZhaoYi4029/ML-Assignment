import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    DBSCAN, AgglomerativeClustering, MeanShift, OPTICS, Birch, 
    SpectralClustering, estimate_bandwidth
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.impute import KNNImputer
import base64
from io import BytesIO

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, delimiter='\t')
    return data

def create_download_link(content, filename, text):
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'

file_path = st.text_input("Enter the file path of the dataset:", r"C:\Degree Year 2 Semester 3\Machine Learning\Assignment\marketing_campaign.csv")

if file_path:
    data = load_data(file_path)
    st.write(f"Data loaded from: {file_path}")

    def preprocess_data(data):
        # Convert columns to numeric
        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
        
        # Drop fully NaN columns
        data.dropna(axis=1, how='all', inplace=True)
        
        # KNN Imputation
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(data.select_dtypes(include=[np.number]))
        imputed_df = pd.DataFrame(imputed_data, columns=data.columns)
        
        # Standardization
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(imputed_df)
        return scaled_data, imputed_df

    try:
        scaled_data, processed_data = preprocess_data(data.copy())

        if scaled_data is None:
            st.warning("Preprocessing failed")
        else:
            # Apply PCA
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(scaled_data)
            processed_data['PCA1'] = pca_components[:, 0]
            processed_data['PCA2'] = pca_components[:, 1]

            st.title("📊 Interactive Clustering Dashboard")
            
            # Sidebar controls
            enable_data_download = st.sidebar.checkbox("Enable Data Download", True)
            clustering_method = st.sidebar.selectbox(
                "Choose Clustering Method",
                ["DBSCAN", "Hierarchical", "GMM", "Mean Shift", "OPTICS", "BIRCH", "Spectral"]
            )

            # Clustering parameters and model setup
            params = {}
            if clustering_method == "DBSCAN":
                params['eps'] = st.sidebar.slider("EPS", 0.1, 2.0, 0.5)
                params['min_samples'] = st.sidebar.slider("Min Samples", 1, 20, 5)
                cluster_model = DBSCAN(**params)
            
            elif clustering_method == "Hierarchical":
                params['n_clusters'] = st.sidebar.slider("Number of Clusters", 2, 10, 3)
                cluster_model = AgglomerativeClustering(**params)
            
            elif clustering_method == "GMM":
                params['n_components'] = st.sidebar.slider("Number of Clusters", 2, 10, 3)
                cluster_model = GaussianMixture(**params)
            
            elif clustering_method == "Mean Shift":
                bandwidth = estimate_bandwidth(processed_data[['PCA1', 'PCA2']], quantile=0.2)
                params['bandwidth'] = st.sidebar.slider(
                    "Bandwidth",
                    min_value=0.1,
                    max_value=5.0,
                    value=float(np.round(bandwidth, 1)),
                    step=0.1
                )
                cluster_model = MeanShift(**params)
            
            elif clustering_method == "OPTICS":
                params['min_samples'] = st.sidebar.slider("Min Samples", 1, 20, 5)
                params['max_eps'] = st.sidebar.slider("Max Epsilon", 0.1, 5.0, 2.0, step=0.1)
                cluster_model = OPTICS(**params)
            
            elif clustering_method == "BIRCH":
                params['n_clusters'] = st.sidebar.slider("Number of Clusters", 2, 10, 3)
                params['threshold'] = st.sidebar.slider("Threshold", 0.1, 1.0, 0.5, step=0.1)
                cluster_model = Birch(**params)
            
            elif clustering_method == "Spectral":
                params['n_clusters'] = st.sidebar.slider("Number of Clusters", 2, 10, 3)
                params['affinity'] = st.sidebar.selectbox("Affinity", ["rbf", "nearest_neighbors"])
                if params['affinity'] == "rbf":
                    params['gamma'] = st.sidebar.slider("Gamma", 0.1, 2.0, 1.0, step=0.1)
                cluster_model = SpectralClustering(**params)

            # Perform clustering
            if clustering_method == "GMM":
                cluster_model.fit(processed_data[['PCA1', 'PCA2']])
                clusters = cluster_model.predict(processed_data[['PCA1', 'PCA2']])
            else:
                clusters = cluster_model.fit_predict(processed_data[['PCA1', 'PCA2']])
            
            processed_data['Cluster'] = clusters

            # Calculate and display Silhouette Score
            unique_clusters = len(np.unique(clusters))
            if unique_clusters > 1:
                silhouette_avg = silhouette_score(processed_data[['PCA1', 'PCA2']], clusters)
                st.success(f"**Silhouette Score**: {silhouette_avg:.3f}")
                st.write(f"Number of clusters: {unique_clusters}")
            else:
                st.warning("Silhouette Score requires at least 2 clusters")

            # Main visualization
            fig = px.scatter(
                processed_data,
                x='PCA1',
                y='PCA2',
                color='Cluster',
                title=f'{clustering_method} Clustering'
            )
            st.plotly_chart(fig)

            # Pair Plots Section
            st.subheader("Pairwise Analysis")
            generate_pair_plot = st.checkbox("Generate Pairwise Plots")
            
            if generate_pair_plot:
                features = st.multiselect(
                    "Select features for pair plot",
                    processed_data.columns,
                    default=['PCA1', 'PCA2', 'Cluster']
                )
                
                if len(features) > 1:
                    pair_fig = px.scatter_matrix(
                        processed_data,
                        dimensions=features,
                        color="Cluster",
                        height=800
                    )
                    st.plotly_chart(pair_fig)
                    
                    # Pair plot download
                    download_pair = st.checkbox("Download Pairwise Plots")
                    if download_pair:
                        pair_img = pair_fig.to_image(format="png")
                        st.download_button(
                            label="Download Pair Plots",
                            data=pair_img,
                            file_name="pair_plots.png",
                            mime="image/png"
                        )

            # Data download section
            if enable_data_download:
                st.subheader("Data Export")
                st.download_button(
                    "Download Clustered Data",
                    processed_data.to_csv(index=False),
                    file_name="clustered_data.csv",
                    mime="text/csv"
                )

            # Report generation
            st.subheader("Analysis Report")
            report_content = f"""
            Clustering Report
            -----------------
            Algorithm: {clustering_method}
            Parameters: {params}
            Clusters: {unique_clusters}
            Silhouette Score: {silhouette_avg if unique_clusters > 1 else 'N/A'}
            """
            st.download_button(
                "Download Report",
                report_content,
                file_name="clustering_report.txt"
            )

    except ValueError as e:
        st.error(f"Error: {e}")
else:
    st.warning("Please provide a valid file path.")