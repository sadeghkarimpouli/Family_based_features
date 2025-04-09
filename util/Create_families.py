import networkx as nx
import pandas as pd
import numpy as np

def Create_families(features, df_parents, catalog_data, undisered_features = ['enum']):
    """
    Create a list of families.
    
    Parameters:
    features: Event-based features
    df_parents: Parent of each event
    catalog_data: Catalog data
    
    Returns:
    All_f: All families
    e_feat: Event-based features
    time_column: Time value of each family (which is equal to the time of family's mainshock)
    """

    undisered_features = ['enum']
    undisered_clm = ['Time[d.s]']
    if undisered_features:
        for prefix in undisered_features:
            [undisered_clm.append(col) for col in features.columns if col.split('_')[0] == prefix]
        features = features.drop(columns=undisered_clm)

    # Merg all dfs based on Index
    features = features[features.columns.sort_values()]
    merged_df = df_parents.merge(features, on='Index', how='inner', suffixes=('', '_duplicate'))
    merged_df = merged_df[merged_df['CB'] == 1]
    g_df = merged_df
    g_df = g_df.drop(columns=['CB'])
    g_df.columns.values[0] = 'target'
    g_df.columns.values[1] = 'source'
    e_feat = [col for col in g_df.columns][2:]

    # Create a dictionary to map events to their Event-based features (from the 'target' rows)
    node_attributes = {}

    for _, row in g_df.iterrows():
        target = row['target']
        attributes = row.drop(['target', 'source']).to_dict()
        node_attributes[target] = attributes  # Store attributes for the target node

    # Create familes
    G = nx.Graph()

    for _, row in g_df.iterrows():
        target = row['target']
        source = row['source']
        if target in node_attributes:
            G.add_node(target, **node_attributes[target])
        if source in node_attributes:
            G.add_node(source, **node_attributes[source])
        else:
            G.add_node(source, **node_attributes[target])  # Add the source without attributes if not found
        G.add_edge(source, target)

    All_f = list(G.subgraph(c) for c in nx.connected_components(G) if len(c) >= 3)

    # Time value of each family
    time_column = []
    for i in range(len(All_f)):
        g_i = np.array(list(All_f[i]))
        m_i = catalog_data[catalog_data['GENIE_ID'].isin(g_i)]['Magnitude']
        t_i = catalog_data[catalog_data['GENIE_ID']==g_i[np.argmax(m_i)]]['Time[d.s]']
        time_column.append(t_i.values[0])
    time_column = pd.Series(time_column)

    return All_f, e_feat, time_column
