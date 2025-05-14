# # ------------- NOTE ----------------- NOTE : Certaines fonctions que nous allons utiliser dans ce projet sont
# codées dans le dossier iads, plus précisement dans le fichier Clustering (kmoyenne, CHA, distance euclienne etc..)
# Nous avons fait le choix d'utiliser et de coder nos fonctions dès que nous pouvons au lieu d'utiliser les
# bibliothèques python. Ceci explique la première case de code importante qui sert à charger les fonctions depuis
# notre dossier iads.

# Importation de librairies standards:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import yfinance as yf
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score
import glob
import os

file_path_finance = "./financial_ratios.csv"
folder_path = "Companies_historical_data/"

# Pour mesurer le temps
import time

# Importation de votre librairie iads:
# La ligne suivante permet de préciser le chemin d'accès à la librairie iads
import sys

sys.path.append('../')  # iads doit être dans le répertoire père du répertoire courant !

# Importation de la librairie iads
import iads as iads

# importation de utils
from iads import utils as ut

# importation de evaluation
from iads import evaluation as ev

# importation de Clustering
from iads import Clustering as clust

import multiprocessing

# Importation des librairies supplémentaires :
import seaborn as sns
from sklearn.preprocessing import StandardScaler


## ----- PARTIE 1.1 -----
def preprocess_risk_clustering(file_path):
    df = pd.read_csv(file_path)

    risk_features = ['debtToEquity', 'currentRatio', 'quickRatio', 'returnOnAssets']
    df_pertinent = df[risk_features].copy()
    df_pertinent = df_pertinent.dropna()

    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(df_pertinent), columns=df_pertinent.columns,
                               index=df_pertinent.index)
    return data_scaled


def preprocess_for_financial_clustering(file_path):
    pertinent_columns_return = ["forwardPE", "beta", "priceToBook", "returnOnEquity", "operatingMargins",
                                "profitMargins"]
    df = pd.read_csv(file_path, index_col=0)
    df_pertinent = df[pertinent_columns_return].copy()
    df_pertinent = df_pertinent.dropna()
    scaler = StandardScaler()
    df_pertinent_scaled = pd.DataFrame(scaler.fit_transform(df_pertinent), columns=df_pertinent.columns,
                                       index=df_pertinent.index)
    return df_pertinent_scaled


df_scaled = preprocess_for_financial_clustering(file_path_finance)

inerties = clust.elbow_method(df_scaled, max_k=15)


def do_kmeans_clustering():
    # Appliquez KMeans grâce à notre focntion dans notre bibliothèque CLust
    Centres, Affect = clust.kmoyennes(4, df_scaled, epsilon=1e-4, iter_max=100, verbose=False)

    df_scaled['clusters'] = np.nan
    for cluster_id, indices in Affect.items():
        df_scaled.iloc[indices, df_scaled.columns.get_loc('clusters')] = cluster_id

    # Normalisation des données pour TSNE
    df_normalized = clust.normalisation(df_scaled.drop('clusters', axis=1))

    # Appliquer t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(df_normalized)

    df_scaled['tsne_1'] = tsne_results[:, 0]
    df_scaled['tsne_2'] = tsne_results[:, 1]

    plt.figure(figsize=(10, 8))
    plt.scatter(df_scaled['tsne_1'], df_scaled['tsne_2'], c=df_scaled['clusters'], cmap='tab20', s=50)
    plt.title('Visualisation des entreprises par clusters (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.colorbar(label='Cluster')
    plt.close()


do_kmeans_clustering()


## ------ PARTIE 1.2 -----
def do_hierarchical_clustering(df, linkage='centroides', dendrogramme=False, verbose=False):
    histo = clust.CHA(df, linkage=linkage, verbose=verbose, dendrogramme=dendrogramme)
    return histo


data_risk_scaled = preprocess_risk_clustering(file_path_finance)
result_risk_df_plot = do_hierarchical_clustering(data_risk_scaled, linkage='complete', verbose=True, dendrogramme=False)


## ------ PARTIE 1.3 -----
def prepare_returns_data(folder_path):
    filepaths = glob.glob(f"{folder_path}/*.csv")
    returns_dict = {}

    for path in filepaths:
        df = pd.read_csv(path)
        company_name = path.split("/")[-1].replace(".csv", "")

        if "Rendement" in df.columns:
            returns_dict[company_name] = df["Rendement"]

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.fillna(returns_df.mean())

    return returns_df


def plot_correlation_dendrogram(returns_df):
    corr_matrix = clust.correlation_matrix(returns_df)
    distance_matrix = clust.correlation_to_distance(corr_matrix)

    linked = linkage(distance_matrix, method='ward')

    plt.figure(figsize=(12, 6))
    scipy.cluster.hierarchy.dendrogram(
        linked,
        labels=returns_df.columns,
        orientation='top',
        distance_sort='descending',
        leaf_rotation=90,  # labels droits
        leaf_font_size=10  # taille lisible
    )
    plt.title("Dendrogramme des corrélations de rendements journaliers")
    plt.xlabel("Entreprises")
    plt.ylabel("Distance (1 - Corrélation)")
    plt.grid(True)
    plt.tight_layout()  # ajuste automatiquement les marges
    plt.close()


returns_df = prepare_returns_data(folder_path)
plot_correlation_dendrogram(returns_df)


## ------ PARTIE 1.4 -----

def do_dbscan_clustering(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)

    df_clustered = pd.DataFrame(data, columns=data.columns)
    df_clustered['Cluster'] = clusters

    return df_clustered


def evaluate_clustering(data, clustering_labels):
    if len(set(clustering_labels)) > 1:  # Silhouette Score nécessite au moins 2 clusters
        score = silhouette_score(data, clustering_labels)
    else:
        score = -1  # Mauvais clustering si un seul cluster détecté

    return score


def compare_algorithms(data_financial, data_risk, data_returns):
    results = []

    # ---------- K-MEANS ----------
    Centres_finance, Affect_finance = clust.kmoyennes(4, data_financial.copy(), epsilon=1e-4, iter_max=100,
                                                      verbose=False)
    Centres_risk, Affect_risk = clust.kmoyennes(4, data_risk.copy(), epsilon=1e-4, iter_max=100, verbose=False)
    Centres_rend, Affect_rend = clust.kmoyennes(4, data_returns.copy(), epsilon=1e-4, iter_max=100, verbose=False)

    data_financial['clusters'] = np.nan
    for cluster_id, indices in Affect_finance.items():
        data_financial.iloc[indices, data_financial.columns.get_loc('clusters')] = cluster_id

    data_risk['clusters'] = np.nan
    for cluster_id, indices in Affect_risk.items():
        data_risk.iloc[indices, data_risk.columns.get_loc('clusters')] = cluster_id

    data_returns['clusters'] = np.nan
    for cluster_id, indices in Affect_rend.items():
        data_returns.iloc[indices, data_returns.columns.get_loc('clusters')] = cluster_id

    results.append(["K-Means", "Finance", evaluate_clustering(data_financial, data_financial['clusters'])])
    results.append(["K-Means", "Risk", evaluate_clustering(data_risk, data_risk['clusters'])])
    results.append(["K-Means", "Returns", evaluate_clustering(data_returns, data_returns['clusters'])])

    # Clustering Hiérarchique
    hier_fin = AgglomerativeClustering(n_clusters=3).fit_predict(data_financial)
    hier_risk = AgglomerativeClustering(n_clusters=3).fit_predict(data_risk)
    hier_returns = AgglomerativeClustering(n_clusters=3).fit_predict(data_returns)

    results.append(["Hierarchical", "Finance", evaluate_clustering(data_financial, hier_fin)])
    results.append(["Hierarchical", "Risk", evaluate_clustering(data_risk, hier_risk)])
    results.append(["Hierarchical", "Returns", evaluate_clustering(data_returns, hier_returns)])

    # DBSCAN
    dbscan_fin = do_dbscan_clustering(data_financial)['Cluster']
    dbscan_risk = do_dbscan_clustering(data_risk)['Cluster']
    dbscan_returns = do_dbscan_clustering(data_returns)['Cluster']

    results.append(["DBSCAN", "Finance", evaluate_clustering(data_financial, dbscan_fin)])
    results.append(["DBSCAN", "Risk", evaluate_clustering(data_risk, dbscan_risk)])
    results.append(["DBSCAN", "Returns", evaluate_clustering(data_returns, dbscan_returns)])

    return pd.DataFrame(results, columns=["Algorithm", "Dataset", "Silhouette Score"])


data_financial = preprocess_for_financial_clustering(file_path_finance)
data_risk = preprocess_risk_clustering(file_path_finance)
data_returns = prepare_returns_data(folder_path)

# Comparer les algorithmes
results_df = compare_algorithms(data_financial, data_risk, data_returns)
