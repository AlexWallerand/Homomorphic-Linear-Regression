# INFO731-Homomorphic-Linear-Regression

L'objectif de ce projet est de mesuer l'efficacité du calcul d'une régression linéaire effecuée sur des données encryptées homomorphiquement. Pour cela nous utiliserons la biblothèque Python [Pyfhel](https://pyfhel.readthedocs.io/en/latest/index.html), basée elle même sur la librairie C++ [SEAL](https://github.com/microsoft/SEAL).

## Installation de l'environnement

Ce projet nécessite l'installation de Pyfhel, qui nécessite lui même l'environnement de [SEAL](https://github.com/Huelse/SEAL-Python) adapté pour le langage Python. Pour cela, se réferrer aux documentations des deux librairies correspondantes. Dans notre cas, nous avons créé un container Docker depuis l'image de SEAL, dans lequel nous exécutons nos scripts avec Python et Pyfhel installé préalablement. 

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
Nous commençons tout d'abord par initialiser le modèle d'encryption de Pyfhel. Ce modèle sera utilisé à chaque fois qu'il sera nécessaire de chiffrer et déchiffrer une donnée puisque qu'il contient la clé privée et publique. Nous l'initialisons avec le schéma CKKS, qui permet de crypter des valeurs flottantes contrairement au schéma BFV. Puis nous définissons une échelle de 2**30, ce qui nous autorise au maximum 10 multiplications sur une donnée cryptée. Sont ensuite générés les clés de relinéarisation et de rotation, qui nous permettent par la suite de pouvoir utiliser des opérations indispensables.

```py
HE = Pyfhel()
qi_sizes = [60] + [30]*10 + [60]
HE.contextGen(scheme="ckks", n=2**14, scale=2**30, qi_sizes=qi_sizes)
HE.keyGen()
HE.relinKeyGen()
HE.rotateKeyGen()
```

Nous créons ensuite une fonction pour encrypter une matrice et une fonction pour la décrypter. La fonction d'encryption parcoure les élements de la matrice et les encrypte un par un avec la fonction de Pyfhel permettant le cryptage d'un float.

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

Puis, il faut maintenant créer les fonctions permettant les calculs élémentaires entre matrices. Pour cela, il faut passer en paramètre de chacune de ces fonctions le modèle Pyfhel *HE*, afin de pouvoir opérer sur les propriétés des données cryptées. Elles sont pour la plupart triviales, sauf pour la multiplication. En effet, il faut rélinéariser chaque multiplication avec l'opérateur ~, et la somme se fait à l'aide de la méthode *cumul_add*.

```py
def matrice_mul(ctxt1, ctxt2, HE):
    '''
    :param ctxt1 & ctxt2: doivent être des matrices encryptées
    :return: la matrice encryptée de la mutiplication entre ctxt1 & ctxt2
    '''
    assert ctxt1.shape[1] == ctxt2.shape[0]
    lignes, colonnes = ctxt1.shape[0], ctxt2.shape[1]
    res = matrice_encrypt(np.zeros((lignes, colonnes)), HE)
    for i in range(lignes):
        for j in range(colonnes):
            mul_res = [~(ctxt1[i][k] * ctxt2[k][j]) for k in range(ctxt1.shape[1])]
            for k in range(len(mul_res)):
                mul_res[0] += mul_res[k]
            res[i][j] = HE.cumul_add(mul_res[0])
    return res
```

Maintenant que toutes les fonctions de bases sont codées, il ne reste plus qu'à adapter les fonctions de descente de gradient avec des données cryptées. Pour cela, on remplace les opérations de base par les opérations de matrices développés ci-dessus.

## Résultats obtenus

Les résultats obtenus suite à l'exécution du script homomorphique sont les suivants : <p align="center"><img src="src\resultat.png" width="500"/></p>

Nous avons fait ce test sur les 10 premières données du dataset. On observe que la courbe obtenue est bien loin des valeurs réelles, et des résultats obtenus dans les essais classiques précédent. Ceci peut s'expliquer par le fait que notre descente de gradient ne peut se faire seulement que sur 2 itérations, à cause du nombre de multipications limité à 10 sur chaque cyphertext. De plus, il faut prendre en compte les incertitudes inhérentes au modèle CKKS, qui augmente plus on effectue des calculs sur la donnée.
En ce qui concerne le temps d'exécutiion, il faut compter environ 4 minutes pour l'ensemble du dataset, alors qu'avec l'algorithme classisque il ne fallait que quelques secondes.
Nous pensons donc ne pas avoir d'erreurs de calcul lors du calcul homoprhique. La solution serait de trouver le bon paramétrage de la descente de gradient en fonction du contexte défini pour le modèle homomorphique.

## Conclusion

Le calcul homomorphique nous semble être une technologie de demain, tant son intérêt en terme de protection de la donnée semble être important. Cependant, le temps de calcul et les erreurs ajoutés aux données sont les principals freins de ce modèle. Nous n'avons pas forcément non plus choisi l'environnement de développement le plus optimal. En effet, étant tous sur des systèmes d'exploitation différents, nous avons choisi d'utiliser la version conteneurisée de SEAL, pour pouvoir utiliser Pyfhel. Ainsi, il y a forcément une perte en performance du fait que le script soit exécuté via un conteneur Docker. Finalement, il aurait été plus judicieux de développer directement en C++ avec la librairie native SEAL pour améliorer l'efficacité du script.