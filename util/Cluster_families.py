
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd

def extract_family_features(family, cat_family, e_feat, flag='both'):
    # Extract features from a  family with Event-based features.

    if flag == 'both' or flag == 'str':
        # Structural features
        structural_features = {
            'num_nodes': family.number_of_nodes(),
            'num_edges': family.number_of_edges(),
            'density': nx.density(family),
            'time_len': cat_family['Time[d.s]'].max() - cat_family['Time[d.s]'].min(),
            'spatial_size': np.sqrt((cat_family['UTM_Easting[m]'].max() - cat_family['UTM_Easting[m]'].min())**2 +\
                                    (cat_family['UTM_Northing[m]'].max() - cat_family['UTM_Northing[m]'].min())**2 +\
                                    (cat_family['Depth[m]'].max() - cat_family['Depth[m]'].min())**2),
            'radius': nx.radius(family),
            'diameter': nx.diameter(family),
        }
        s_feat = ['num_nodes', 'num_edges', 'density', 'time_len', 'spatial_size', 'radius', 'diameter']
    
    if flag == 'both' or flag == 'node':
    # Node feature statistics
        node_features = []
        f_feat = [col+'_mean' for col in e_feat] #+ [col.split('_')[0]+'_std' for col in e_feat]
        for node in family.nodes():
            node_features.append(list(family.nodes[node].values()))
        node_features = np.array(node_features)
        feature_stats = {
            f'feature_{i}_mean': np.mean(node_features[:, i])
            for i in range(node_features.shape[1])
        }
        # feature_stats.update({
        #     f'feature_{i}_std': np.std(node_features[:, i])
        #     for i in range(node_features.shape[1])
        # })
    
    # Combine all features
    if flag == 'str':
        all_features = {**structural_features}
        f_label = s_feat
    elif flag == 'node':
        all_features = {**feature_stats}
        f_label = f_feat
    else:
        all_features = {**structural_features, **feature_stats}
        f_label = s_feat + f_feat
    
    return np.array(list(all_features.values())), f_label

def Cluster_families(families, catalog_data, e_feat, n_clusters=3, method='kmeans', family_feature_flag = 'both'):
    """
    Cluster a list of families based on their features.
    
    Parameters:
    families: List of families
    n_clusters: Number of desired clusters
    method: Clustering method ('kmeans' or 'ward')
    
    Returns:
    cluster_labels: label of each family
    feature_matrix: Family-based features
    cluster_centroids: Family-based features value on te centroid of each cluster
    """
    # Extract features for each family
    feature_vectors = []
    for family in families:
        g_idx = np.array(list(family)).astype('int')
        cat_family = catalog_data[catalog_data['GENIE_ID'].isin(g_idx)]
        features,_  = extract_family_features(family, cat_family, e_feat, flag=family_feature_flag) 
        feature_vectors.append(features)
    _, f_label = extract_family_features(family, cat_family, e_feat, flag=family_feature_flag)

    # Convert to array and normalize
    feature_matrix = np.array(feature_vectors)
    scaler = MinMaxScaler()
    normalized_features = feature_matrix
    normalized_features[:,:7] = scaler.fit_transform(feature_matrix[:,:7])
    normalized_features[:,-len(e_feat):] = feature_matrix[:,-len(e_feat):]
    
    # Perform clustering
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(normalized_features) 
    else:
        from sklearn.cluster import AgglomerativeClustering
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(normalized_features)
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, 
                                          metric="euclidean",
                                          linkage='ward')
        cluster_labels = clusterer.fit_predict(1 - similarity_matrix)
    
    cluster_centroids = clusterer.cluster_centers_
    
    return cluster_labels, pd.DataFrame(np.array(feature_vectors), columns=f_label), cluster_centroids