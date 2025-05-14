

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque
# ------------------------ 

def normalisation (df) :
    norm_df = df.copy()
    for x in df :
        max_x = norm_df[x].max()
        min_x = norm_df[x].min()
        norm_df[x] = (norm_df[x] - min_x)/(max_x - min_x)
    return norm_df

def centroide (df):
    df_np = np.array(df)
    return np.mean(df_np, axis=0)

def dist_euclidienne(x,y):
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x - y)

def dist_centroides(x,y):
    return dist_euclidienne(centroide(x),centroide(y))

def dist_complete(x,y):
    max_dist = 0
    max_i,max_j = None, None
    for i in range(len(x)):
        for j in range(len(y)):
            dist_ij = dist_euclidienne(x.iloc[i],y.iloc[j])
            if dist_ij > max_dist:
                max_dist = dist_ij
                max_i , max_j = i , j
    return max_dist,(max_i,max_j)

def dist_simple(x,y):
    min_dist = np.inf
    min_i,min_j = None, None
    for i in range(len(x)):
        for j in range(len(y)):
            dist_ij = dist_euclidienne(x.iloc[i],y.iloc[j])
            if dist_ij < min_dist:
                min_dist = dist_ij
                min_i , min_j = i , j
    return min_dist,(min_i,min_j)

def dist_average(x,y):
    xv = x.values
    yv = y.values

    # Calculer toutes les distances entre les points des deux ensembles
    distances = np.sqrt(np.sum((xv[:, np.newaxis] - yv) ** 2, axis=2))

    # Calculer la distance d'average linkage
    average_distance = distances.mean()

    return average_distance


def initialise_CHA(df):
    dico = {}
    for i in range(len(df.index)):
        dico[i]=[df.index[i]]
    return dico

def fusionne(df, P0, verbose=False):
    dist_min = float('inf')
    min_x, min_y = None, None
    
    for x in P0:
        for y in P0:
            if x != y:
                # Utiliser df.loc[] si les indices sont des labels
                dist_x_y = dist_centroides(df.loc[P0[x]], df.loc[P0[y]])
                if dist_x_y < dist_min:
                    dist_min = dist_x_y
                    min_x = x
                    min_y = y
    
    # Fusionner les deux clusters
    P1 = P0.copy()
    P1[str(min_x) + '-' + str(min_y)] = P0.pop(min_x) + P0.pop(min_y)

    
    if verbose:
        print(f"Fusion des clusters {min_x} et {min_y} avec distance {dist_min}")
    
    return P1, min_x, min_y, dist_min


def CHA_centroid(df,verbose=False,dendrogramme=False):
    P0 = initialise_CHA(df)
    histo = []
    if verbose :
        print("CHA_centroid: clustering hiérarchique ascendant, version Centroid Linkage")
    while len(P0) > 1:
        P0 , x , y , dist = fusionne(df,P0,verbose)
        nb_elem = len(P0[max(P0.keys())])
        histo.append([x,y,dist,nb_elem])
        if verbose:
            print(f"CHA_centroid: une fusion réalisée de  {x}  avec {y} de distance  {dist}")
            print(f"CHA_centroid: le nouveau cluster contient  {nb_elem}  exemples")
    if verbose:
        print("CHA_centroid: plus de fusion possible, il ne reste qu'un cluster unique.")
    
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            histo, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    return histo

def CHA(df,linkage='centroides', verbose=False,dendrogramme=False):
    """  ##### donner une documentation à cette fonction
    """
    fun_dist = None
    if linkage=='centroides' :
        fun_dist = lambda x,y : dist_centroides(x,y)
    elif linkage=='complete':
        fun_dist = lambda x,y : dist_complete(x,y)[0]
    elif linkage=='simple':
        fun_dist = lambda x,y : dist_simple(x,y)[0]
    elif linkage=='average':
        fun_dist = lambda x,y : dist_average(x,y)
    else:
        raise ErrorDistance ("Distance not found")
    
    
    def fusionne(df,P0,verbose=False):
        P1 = P0.copy()
        dist_min = np.inf
        min_x,min_y = None,None
        for x in P0:
            for y in P0:
                if x!=y:
                    dist_x_y = fun_dist(df.loc[P0[x]],df.loc[P0[y]])
                    if dist_x_y < dist_min:
                        dist_min = dist_x_y
                        min_x = x
                        min_y = y

        if max(P0.keys())+1 in P0:
            raise IndexError ("Index len(P0) already exist in P0") 
        P1[max(P0.keys())+1] = P0[min_x] + P0[min_y]
        del P1[min_x]
        del P1[min_y]
        if verbose :
            print(f"Distance mininimale trouvée entre  [{min_x}, {min_y}]  =  {dist_min}")

        return P1,min_x,min_y,dist_min


    if verbose:
        print(f"Clustering Hiérarchique Ascendant : approche  {linkage}")
        print(f"Clustering hiérarchique ascendant, version {linkage} Linkage")
    P0 = initialise_CHA(df)
    histo = []
    while len(P0) > 1:
        P0 , x , y , dist = fusionne(df,P0,verbose)
        nb_elem = len(P0[max(P0.keys())])
        histo.append([x,y,dist,nb_elem])

    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        Z = np.array(histo)  # Convertit la liste en matrice numpy

        scipy.cluster.hierarchy.dendrogram(
        Z,
        leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    return histo

def inertie_cluster(Ens):
    """ Array -> float
        Ens: array qui représente un cluster
        Hypothèse: len(Ens)> >= 2
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    centro = centroide(Ens)
    return np.sum([dist_euclidienne(Ens.loc[i],centro)**2 for i in Ens.index])


def init_kmeans(K, Ens):
    """ int * DataFrame -> Array
        K : entier >1 et <=n (le nombre d'exemples de Ens)
        Ens: DataFrame contenant n exemples
    """
    n = Ens.shape[0]
    indices_rnd = np.random.choice(n, K, replace=False)
    return Ens.iloc[indices_rnd].values

def plus_proche(Exe, Centres):
    """ Array * Array -> int
        Exe : Array contenant un exemple
        Centres : Array contenant les K centres
    """
    distances = [dist_euclidienne(cent,Exe) for cent in Centres]
    return np.argmin(distances)


def affecte_cluster(Base,Centres):
    """ Array * Array -> dict[int,list[int]]
        Base: Array contenant la base d'apprentissage
        Centres : Array contenant des centroides
    """
    dico = {}
    for i in range(Base.shape[0]):
        pp = plus_proche(Base.iloc[i],Centres)
        if pp not in dico:
            dico[pp] = [i]
        else:
            dico[pp].append(i)
    return dico


def nouveaux_centroides(Base,U):
    """ Array * dict[int,list[int]] -> DataFrame
        Base : Array contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    new_cent = []
    for j in U:
        new_cent.append(centroide(Base.iloc[U[j]]))
    return np.array(new_cent)


def inertie_globale(Base, U):
    """ Array * dict[int,list[int]] -> float
        Base : Array pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    return np.sum([inertie_cluster(Base.iloc[U[j]]) for j in U])


def kmoyennes(K, Base, epsilon, iter_max,verbose=False):
    """ int * Array * float * int -> tuple(Array, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : Array pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    Centres = init_kmeans(K,Base)
    mat_affec = None
    inert_glob = 0
    for i in range(1,iter_max+1):
        
        mat_affec = affecte_cluster(Base,Centres)
        Centres = nouveaux_centroides(Base,mat_affec)
        new_inert = inertie_globale(Base,mat_affec)
        if verbose : print(f"iteration {i} Inertie : {new_inert:.4f} Difference: {abs(inert_glob - new_inert):.4f}")
        
        if abs(inert_glob - new_inert)<epsilon:
            break
        
        inert_glob = inertie_globale(Base,mat_affec)
        
    return Centres,mat_affec
    

def affiche_resultat(Base,Centres,Affect):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """
    couleur = cm.tab20(np.linspace(0, 1, len(Affect)))
    for j in Affect:
        for i in Affect[j]:
            x,y = Base.iloc[i]
            plt.scatter(x,y,color=couleur[j])
    for x,y in Centres:
        plt.scatter(x,y,color='red',marker='x')
    plt.show()

def elbow_method(data, max_k=10):
    inerties = []  # Liste pour stocker les inerties pour chaque nombre de clusters
    
    # Parcourir les différentes valeurs de K
    for k in range(1, max_k+1):
        Centres, Affect = kmoyennes(k, data, epsilon=1e-4, iter_max=100, verbose=False)
        inertie = inertie_globale(data, Affect)
        inerties.append(inertie)
        
        print(f"K={k}, Inertie={inertie}")
    
    # Plot des inerties
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k+1), inerties, marker='o', linestyle='-', color='b')
    plt.title("Méthode du Coude")
    plt.xlabel("Nombre de Clusters (K)")
    plt.ylabel("Inertie")
    plt.grid(True)
    plt.show()
    return inerties

def compute_affinity(X, perplexity=30.0):
    """Calcule les probabilités d'affinité (distributions gaussiennes)"""
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))

    # Calculer les distances euclidiennes entre tous les points
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = dist_euclidienne(X[i], X[j])
            distances[i, j] = dist
            distances[j, i] = dist
    P = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        # Estimation de la largeur de la gaussienne par l'échelle de perplexité
        beta = 1.0
        p_row = np.exp(-distances[i] * beta)
        p_row[i] = 0  # pas de probabilité pour un point avec lui-même
        p_row /= np.sum(p_row) 
        P[i] = p_row

    # Symétriser la matrice de probabilité
    P = (P + P.T) / (2 * n_samples)
    return P



def correlation_matrix(df):
    """Calculer la matrice de corrélation ."""
    n = df.shape[1]
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xi = df.iloc[:, i]
            xj = df.iloc[:, j]
            cov = np.cov(xi, xj, bias=True)[0, 1]
            std_xi = np.std(xi)
            std_xj = np.std(xj)
            corr_matrix[i, j] = cov / (std_xi * std_xj)
    return pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)

def correlation_to_distance(corr_matrix):
    """Transformer la corrélation en distance : d = 1 - corr"""
    return 1 - corr_matrix



