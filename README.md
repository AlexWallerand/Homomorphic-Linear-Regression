# INFO731-Homomorphic-Linear-Regression

L'objectif de ce projet est de mesuer l'efficacité du calcul d'une régression linéaire effecuée sur des données encryptées homomorphiquement. Pour cela nous utiliserons la biblothèque python [Pyfhel](https://pyfhel.readthedocs.io/en/latest/index.html).

<br>

## Régression linéaire classique

Nous commençons par implémenter un régression linéaire classique qui nous servira de base pour la suite. Pour cela, nous utiliserons le set de donnée _car_sales.csv_ suivant :

<p align="center"><img src="src\scatter.svg" width="500"/></p>
<p align="center"><em>Taille des moteurs en fonction de leur puissance en chevaux</em></p>

Nous nous avons besoin de récupérer respectivement dans les variables _x_ et _y_ la liste des tailles moteur et la liste des puissances. Nous cherchons ensuite à obtenir les matrices _X_ et _Y_ de la forme où _m_ est le nombre de données étudiées : <p align="center"><img src="src\X_Y.png" width="300"/></p>

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

L'étape suivante est d'initialiser _theta_, le vecteur qui caractérisera notre modèle. Theta est de la forme : <p align="center"><img src="src\theta.png" width="100"/></p>

```py
def init_theta(data):
    return np.zeros((data.shape[1], 1))
```

Nous ne connaissons pas la valeur de _theta_ au début, ce sera à notre algorithme de trouver le theta qui minimise la fonction de coût, c'est-à-dire les coefficiants _a_ et _b_ permettant de créer la droite affine minimisant la distance de chacun des points à la droite.

<br>

Nous passons ensuite à la définition du modèle linéaire. Celui-ci correspond simplement au produit matriciel de _X_ et de _theta_.<p align="center"><img src="src\modele.png" width="100"/></p>

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

Il ne nous reste plus qu'à effectuer un algorithme de descente de gradient pour approximer la valeur de theta idéale. Une descente de gradient est le calcul suivant répété un nombre _n_ d'itérations :<p align="center"><img src="src\descente.png" width="250"/></p>
_α_ représente la vitesse de modification des paramètres à chaque itération. Plus _α_ est grand, plus la modification des paramètres entre deux itérations successives est grande (donc plus la probabilité de « rater » le minimum ou de diverger est grand. C'est pourquoi nous prévilégions l'utilisation d'un _α_ de l'ordre du centième.

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

<br>

## Régression sans Numpy

Nous avons maintenant un algorithme permettant d'effectuer une régression linéaire sur un set de donnée. Notre code utilise la bibliothèque [Numpy](https://numpy.org/doc/1.22/index.html) pour effectuer les calculs de multiplication, de soustraction, de multiplication par un scalaire et de transposée de matrice. Cependant, les méthodes de calculs matriciels de Numpy de sont pas compatibles avec Pyfhel. Afin de réaliser une régression linéaire sur des données encryptées homomorphiquement, nous avons donc besoin de coder nous même les fonctions de calcul matriciel.

En guise d'exemple, voici la fonction de multiplication matricielle que nous avons codé :

```py
def test_matrice_mul(mat1, mat2):
    assert mat1.shape[1] == mat2.shape[0]
    lignes, colonnes = mat1.shape[0], mat2.shape[1]
    res = np.zeros((lignes, colonnes))
    for i in range(lignes):
        for j in range(colonnes):
            sum = 0
            for k in range(mat1.shape[1]):
                sum = sum + mat1[i][k] * mat2[k][j]
            res[i][j] = sum
    return res
```

Nous allons tester notre régression linéaire en remplaçant tout les calculs de Numpy par nos propres fonctions. Le calcul du gradient devient par exemple :

```py
def grad(X ,y, theta):
    m = len(y)
    return test_matrice_scalar_mul(test_matrice_mul(test_matrice_transpose(X), test_matrice_sou(model(X, theta) , y)), 1/m)
```

Nous vérifions bien que nous obtenons exactement le même résultat que préfédement :<p align="center"><img src="src\comparaison.png" width="500"/></p>

Tout fonctionne correctement nous pouvons passer à la partie d'encryption.

<br>

## Régression linéaire encryptée

Nous arrivons maintenant à la partie crutiale du projet, la partie d'encryption.
Nous commençons tout d'abord par initialiser le modèle d'encryption de Pyfhel. Ce modèle sera utilisé à chaque fois qu'il sera nécessaire de chiffrer et déchiffrer une donnée puisque qu'il contient la clé privée et publique.

```py
HE = Pyfhel()
HE.contextGen(scheme="ckks", n=2**14, scale=2**30, qi_sizes=[60, 30, 30, 30, 60])
HE.keyGen()
```

Nous créons ensuire un fonction pour encrypter une matrice et un fonction pour la décrypter. La fonction d'encryption parcoure les élements de la matrice et les encrypte un par un avec la fonction de Pyfhel permettant le cryptage d'un float.

```py
def matrice_encrypt(mat, HE):
    '''
    :param mat: doit être une matrice numpy de float
    :return: matrice encryptée
    '''
    lignes, colonnes = mat.shape[0], mat.shape[1]
    res_mat = []
    for i in range(lignes):
        val_ligne = []
        for j in range(colonnes):
            val_ligne.append(HE.encryptFrac(
                np.array([mat[i, j]], dtype=np.float64)))
        res_mat.append(val_ligne)
    res_mat = np.asarray(res_mat)
    return res_mat.reshape(lignes, colonnes)


def decrypt(enc_mat, HE):
    '''
    :param enc_mat: doit être une matrice encryptée
    :return: matrice décryptée
    '''
    lignes, colonnes = enc_mat.shape[0], enc_mat.shape[1]
    res_mat = []
    for i in range(lignes):
        val_ligne = []
        for j in range(colonnes):
            val_ligne.append(HE.decryptFrac(enc_mat[i][j]))
        res_mat.append(val_ligne)
    res_mat = np.asarray(res_mat)
    return res_mat.reshape(lignes, colonnes)
```
