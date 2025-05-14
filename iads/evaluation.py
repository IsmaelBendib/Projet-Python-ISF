

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy



def crossval(X, Y, n_iterations, iteration):
    Xtest,Ytest = [],[]
    add_app = np.zeros(len(X))
    for k in range(iteration*(len(X)//n_iterations),(iteration+1)*(len(X)//(n_iterations))):
        Xtest.append(X[k])
        Ytest.append(Y[k])
        add_app[k] = 1
    Xapp = np.array(X[add_app==0])
    Yapp = np.array(Y[add_app==0])
    return Xapp, Yapp, np.array(Xtest), np.array(Ytest)

# code de la validation croisée (version qui respecte la distribution des classes)

def crossval_strat(X, Y, n_iterations, iteration):
    # Calcul du nombre total de données
    nb_data = len(X)
    
    # Calcul du nombre de classes uniques
    labels = np.unique(Y)
    nb_labels = len(labels)
    
    # Initialisation des listes pour les ensembles d'apprentissage et de test
    Xapp, Yapp = [], []
    Xtest, Ytest = [], []
    
    # Calcul de la taille des sous-ensembles de test pour chaque classe
    test_size_per_class = nb_data // (n_iterations * nb_labels)
    
    # Index des données utilisées pour l'itération actuelle
    start_index = iteration * test_size_per_class
    end_index = (iteration + 1) * test_size_per_class
    
    # Parcours de chaque classe pour diviser les données en ensembles d'apprentissage et de test
    for label in labels:
        # Sélection des indices correspondant à la classe en cours
        indices = np.where(Y == label)[0]
        np.random.shuffle(indices)  # Mélange aléatoire des indices
        
        # Séparation des indices pour les ensembles d'apprentissage et de test
        test_indices = indices[start_index:end_index]
        train_indices = np.setdiff1d(indices, test_indices)
        
        # Ajout des données aux ensembles d'apprentissage et de test
        Xtest.extend(X[test_indices])
        Ytest.extend(Y[test_indices])
        Xapp.extend(X[train_indices])
        Yapp.extend(Y[train_indices])
    
    return np.array(Xapp), np.array(Yapp), np.array(Xtest), np.array(Ytest)


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return np.mean(L), np.std(L)  

def validation_croisee(C, DS, nb_iter,verbose=False):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    perf = []
    for i in range(nb_iter):
        Xapp,Yapp,Xtest,Ytest = crossval_strat(DS[0],DS[1], nb_iter, i)
        classifieur = copy.deepcopy(C)
        classifieur.train(Xapp, Yapp)
        perf.append(classifieur.accuracy(Xtest, Ytest))
        if verbose:
            print(f"Itération {i}: taille base app.= {len(Xapp)}	taille base test= {len(Xtest)}	Taux de bonne classif: {perf[i]:.4f}")
    taux_moyen, taux_ecart = analyse_perfs(perf)
    return perf,taux_moyen,taux_ecart



def validation_croisee_arbre_decision(DS, nb_iter,eps_min,eps_max):
    """ tuple[array, array, array] * int * float * float -> tuple[ list[float], float, float]
    """
    
    def validation_croisee_adapt_to_arbre(C, DS, nb_iter,verbose=False):
            """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
            """
            perf = []
            for i in range(nb_iter):
                Xapp,Yapp,Xtest,Ytest = crossval(DS[0],DS[1], nb_iter, i)
                classifieur = copy.deepcopy(C)
                classifieur.train(Xapp, Yapp)
                perf.append(classifieur.accuracy(Xtest, Ytest))
                if verbose:
                    print(f"Itération {i}: taille base app.= {len(Xapp)}	taille base test= {len(Xtest)}	Taux de bonne classif: {perf[i]}")
            taux_moyen, taux_ecart = analyse_perfs(perf)
            return perf,taux_moyen,taux_ecart,classifieur.number_leaves()
    
    xs = np.arange(eps_min,eps_max+0.01,0.01)
    perfs, taux_moyens, taux_ecarts, nb_feuilles = [], [], [], []
    for x in xs:
        arbre = ClassifierArbreDecision(len(DS[2]),x,DS[2])        
        a,b,c,d = validation_croisee_adapt_to_arbre(arbre,DS[:2],nb_iter)#,verbose=True)
        perfs.append(a)
        taux_moyens.append(b)
        taux_ecarts.append(c)
        nb_feuilles.append(d)
        
    fig, ax = plt.subplots()
    ax.plot(xs,taux_moyens)
    ax.plot(xs,taux_ecarts)
    ax.set(xlabel='epsilon',ylabel='function values')
    plt.legend(['taux_moyens','taux_ecarts'])
    ax.grid()
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(xs,nb_feuilles)
    ax.set(xlabel='epsilon',ylabel='nb_feuilles')
    plt.legend(['nb_feuilles'])
    ax.grid()
    plt.show()
    return perfs, taux_moyens, taux_ecarts, nb_feuilles

