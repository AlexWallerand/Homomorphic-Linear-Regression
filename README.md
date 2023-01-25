# INFO731-Homomorphic-Linear-Regression

L'objectif de ce projet est de mesuer l'efficacité du calcul d'une régression linéaire effecuée sur des données encryptées homomorphiquement. Pour cela nous utiliserons la biblothèque python [Pyfhel](Xhttps://pyfhel.readthedocs.io/en/latest/index.html).

<br>

## Régression linéaire classique

Nous commençons par implémenter un régression linéaire classique qui nous servira de base pour la suite. Pour cela, nous utiliserons le set de donnée *car_sales.csv* suivant :

<p align="center"><img src="src\scatter.svg" width="500"/></p>
<p align="center"><em>Taille des moteurs en fonction de leur puissance en chevaux</em></p>

Nous nous avons besoin de récupérer respectivement dans les variables *x* et *y* la liste des tailles moteur et la liste des puissances. Nous cherchons ensuite à obtenir les matrices *X* et *Y* de la forme où *m* est le nombre de données étudiées : <p align="center"><img src="src\X_Y.png" width="300"/></p> 

Pour ce faire, nous avons créé la fonction suivante :

```py
def load_data(path, column1, column2):
    data = pd.read_csv(path)
    x, y = data[column1].values, data[column2].values
    X = np.column_stack((x,np.ones(x.shape)))
    y = y.reshape(y.shape[0], 1)
    return x, X, y
```
<br>

L'étape suivante est d'initialiser *theta*, le vecteur qui caractérisera notre modèle. Theta est de la forme : <p align="center"><img src="src\theta.png" width="100"/></p> 
```py
def init_theta(data):
    return np.zeros((data.shape[1], 1))
```
Nous ne connaissons pas la valeur de *theta* au début, ce sera à notre algorithme de trouver le theta qui minimise la fonction de coût, c'est-à-dire les coefficiants *a* et *b* permettant de créer la droite affine minimisant la distance de chacun des points à la droite.

<br>

Nous passons ensuite à la définition du modèle linéaire. Celui-ci correspond simplement au produit matriciel de *X* et de *theta*.<p align="center"><img src="src\modele.png" width="100"/></p> 
```py
def model(X, theta):
    return np.dot(X, theta)
```

<br>

La fonction de coût du modèle est defini selon la formule suivante :<p align="center"><img src="src\cout.png" width="250"/></p>
Comme dit précédemment, nous allons devoir minimiser cette fonction de coût. Nous allons donc avoir besoin de calculer le gradient de cette fonction. Nous avons fait le choix de ne pas coder la fonction en elle-même mais de directement coder le calcul de son gradient.

<br>

Le gradient de la fonction coût est donnée par cette formule :<p align="center"><img src="src\grad.png" width="250"/></p>
Grâce à Numpy, l'implémentation de la formule est très simple.
```py
def grad(X ,y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)
```

<br>

Il ne nous reste plus qu'à effectuer un algorithme de descente de gradient pour approximer la valeur de theta idéale. Une descente de gradient est le calcul suivant répété un nombre *n* d'itérations :<p align="center"><img src="src\descente.png" width="250"/></p> 
*α* représente la vitesse de modification des paramètres à chaque itération. Plus *α* est grand, plus la modification des paramètres entre deux itérations successives est grande (donc plus la probabilité de « rater » le minimum ou de diverger est grand. C'est pourquoi nous prévilégions l'utilisation d'un *α* de l'ordre du centième.
```py
def gradient_descent(X, y, theta, learning_rate, n_iteration):
    for i in range(0, n_iteration):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta
```

<br>

Nous avons à présent tout ce qu'il nous faut pour effectuer une régression linéaire sur nos données. Nous exécutons le code et nous obtenons la courbe suivante :
```py
x, X, y = load_data('Car_sales.csv','Engine_size','Horsepower')
theta = init_theta(X)
theta_f = gradient_descent(X, y, theta, learning_rate=0.001, n_iteration=1000)
model_regression = model(X, theta_f)

plt.scatter(x, y)
plt.xlabel('Engine size')
plt.ylabel('Horse power')
plt.plot(x, model_regression, color='green')
plt.show()
```
<p align="center"><img src="src\regression.png" width="500"/></p> 






