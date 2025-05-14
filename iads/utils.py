

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# genere_dataset_uniform:
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    data_desc = np.random.uniform(binf,bsup,(2*n,p))
    data_label = np.asarray([-1 for _ in range(n)]+[1 for _ in range(n)])
    return data_desc , data_label

# genere_dataset_gaussian:
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    # COMPLETER ICI (remplacer la ligne suivante)    
    data_desc_negative = np.random.multivariate_normal(negative_center,negative_sigma,nb_points)
    data_desc_positive = np.random.multivariate_normal(positive_center,positive_sigma,nb_points)
    data_label = np.asarray([-1 for _ in range(nb_points)]+[1 for _ in range(nb_points)])
    data_desc = np.concatenate((data_desc_negative,data_desc_positive))
    return data_desc,data_label

# plot2DSet:
def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    # COMPLETER ICI (remplacer la ligne suivante)
    data_negatifs = desc[labels == -1]
    data_positifs = desc[labels == +1]
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color="red")
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color="blue")

# plot_frontiere:
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    #colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])

# create_XOR:
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    negative_points_1 = np.random.multivariate_normal(np.array([0,0]), np.array([[var,0],[0,var]]), n)
    negative_points_2 = np.random.multivariate_normal(np.array([1,1]), np.array([[var,0],[0,var]]), n)
    positive_points_1 = np.random.multivariate_normal(np.array([1,0]), np.array([[var,0],[0,var]]), n)
    positive_points_2 = np.random.multivariate_normal(np.array([0,1]), np.array([[var,0],[0,var]]), n)
    
    desc = np.vstack((negative_points_1, negative_points_2, positive_points_1, positive_points_2))
    labels = np.asarray([-1 for i in range(2*n)] + [+1 for i in range(2*n)])
    
    return desc, labels



# desc_gradient:
def desc_gradient(all_w,X,Y):
	# Visualisation de la descente de gradient

	allw = np.array(all_w)

	# construction d'une grille de 'toutes' les valeurs possibles de w
	mmax=allw.max(0)
	mmin=allw.min(0)
	x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],30),np.linspace(mmin[1],mmax[1],30))
	grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))

	# evaluation du cout pour toutes ces solutions potentielles

	##########
	# construction de res = calcul du cout du perceptron pour tous les couples
	# (w1,w2) définis dans grid
	print(f"mmax : {mmax}, mmin : {mmin}")

	res = np.array([np.sum([(np.dot(w,X[i])-Y[i])**2 for i in range(len(Y))]) for w in grid])

	res = res.reshape(x1grid.shape)

	#####################################################
	fig, ax = plt.subplots() # pour 1 et 2
	ax.set_xlabel('$w_1$')
	ax.set_ylabel('$w_2$')
	CS = ax.contour(x1grid,x2grid,res)
	ax.clabel(CS, inline=1, fontsize=10)

	# ajoute de la couleur: jaune = plus grande itération
	ax.scatter(allw[:,0], allw[:,1], c=np.arange(len(allw)))

	#plt.savefig("out/espace_param_MC.png")









    
