import numpy as np

def matrice_encrypt(mat):
    '''
    :param mat: doit être une matrice numpy de float
    :return: matrice encryptée
    '''
    lignes, colonnes = mat.shape[0], mat.shape[1]
    res_mat = []
    for i in range(lignes):
        val_ligne = []
        for j in range(colonnes):
            val_ligne.append(HE.encryptFrac(mat[i][j]))
        res_mat.append(val_ligne)
    res_mat = np.asarray(res_mat)
    return res_mat.reshape(lignes, colonnes)


def decrypt(enc_mat):
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


def matrice_mul(ctxt1, ctxt2):
    '''
    :param ctxt1 & ctxt2: doivent être des matrices encryptées
    :return: la matrice encryptée de la mutiplication entre ctxt1 & ctxt2
    '''
    assert ctxt1.shape[1] == ctxt2.shape[0]
    lignes, colonnes = ctxt1.shape[0], ctxt2.shape[1]
    res = matrice_encrypt(np.zeros((lignes, colonnes)))
    for i in range(lignes):
        for j in range(colonnes):
            sum = HE.encryptFrac(0)
            for k in range(ctxt1.shape[1]):
                sum = sum + ctxt1[i][k] * ctxt2[k][j]
            res[i][j] = sum
    return res

def matrix_scalar_mul(self, ctxt1, enc_scalar):
        '''
        :param ctxt1: encrypted matrix, size (mxn)
        :return: encrypted matrix, size (mxn)
        '''
        lignes, colonnes = ctxt1.shape[0], ctxt1.shape[1]
        temp = np.zeros((lignes, colonnes))
        res = matrice_encrypt(np.zeros((lignes, colonnes)))
        for i in range(lignes):
            for j in range(colonnes):
                res[i][j] = enc_scalar*ctxt1[i][j]
        return res


def matrice_sou(ctxt1, ctxt2):
    '''
    :param ctxt1 & ctxt2: doivent être des matrices encryptées
    :return: la matrice encryptée de la soustraction entre ctxt1 & ctxt2
    '''
    assert ctxt1.shape[0] == ctxt2.shape[0] and ctxt1.shape[1]==ctxt2.shape[1] 
    lignes, colonnes = ctxt1.shape[0], ctxt2.shape[1]
    res = matrice_encrypt(np.zeros((lignes, colonnes)))
    for i in range(lignes):
        for j in range(colonnes):
            res[i][j] = ctxt1[i][j] - ctxt2[i][j]
    return res


def matrice_add(ctxt1, ctxt2):
    '''
    :param ctxt1 & ctxt2: doivent être des matrices encryptées
    :return: la matrice encryptée de la mutiplication entre ctxt1 & ctxt2
    '''
    assert ctxt1.shape[0] == ctxt2.shape[0] and ctxt1.shape[1]==ctxt2.shape[1] 
    lignes, colonnes = ctxt1.shape[0], ctxt2.shape[1]
    res = matrice_encrypt(np.zeros((lignes, colonnes)))
    for i in range(lignes):
        for j in range(colonnes):
            res[i][j] = ctxt1[i][j] + ctxt2[i][j]
    return res


def matrice_sqrt(ctxt1):
    '''
    :param ctxt1: doit être des matrices encryptées
    :return: la matrice ctxt1 au carré
    '''
    lignes, colonnes = ctxt1.shape[0], ctxt1.shape[1]
    res = matrice_encrypt(np.zeros((lignes, colonnes)))
    for i in range(lignes):
        for j in range(colonnes):
            res[i][j] = ctxt1[i][j] * ctxt1[i][j]
    return res




###################### TEST ##########################

def test_matrice_mul(ctxt1, ctxt2):
    '''
    :param ctxt1 & ctxt2: doivent être des matrices encryptées
    :return: la matrice encryptée de la mutiplication entre ctxt1 & ctxt2
    '''
    assert ctxt1.shape[1] == ctxt2.shape[0]
    lignes, colonnes = ctxt1.shape[0], ctxt2.shape[1]
    #res = matrice_encrypt(np.zeros((lignes, colonnes)))
    res = np.zeros((lignes, colonnes))
    for i in range(lignes):
        for j in range(colonnes):
            #sum = HE.encryptFrac(0)
            sum = 0
            for k in range(ctxt1.shape[1]):
                sum = sum + ctxt1[i][k] * ctxt2[k][j]
            res[i][j] = sum
    return res


def test_matrice_sou(ctxt1, ctxt2):
    '''
    :param ctxt1 & ctxt2: doivent être des matrices encryptées
    :return: la matrice encryptée de la soustraction entre ctxt1 & ctxt2
    '''
    assert ctxt1.shape[0] == ctxt2.shape[0] and ctxt1.shape[1]==ctxt2.shape[1] 
    lignes, colonnes = ctxt1.shape[0], ctxt2.shape[1]
    #res = matrice_encrypt(np.zeros((lignes, colonnes)))
    res = np.zeros((lignes, colonnes))
    for i in range(lignes):
        for j in range(colonnes):
            res[i][j] = ctxt1[i][j] - ctxt2[i][j]
    return res

def test_matrice_scalar_mul(ctxt1, enc_scalar):
    '''
    :param ctxt1: doit être une matrice encryptée et enc_scalar un nombre encrypté
    :return: la multiplication scalaire de la matrice encryptée avec enc_scalar
    '''
    lignes, colonnes = ctxt1.shape[0], ctxt1.shape[1]
    #temp = np.zeros((lignes, colonnes))
    #res = self.encrypt(temp)
    res = np.zeros((lignes, colonnes))
    for i in range(lignes):
        for j in range(colonnes):
            res[i][j] = enc_scalar*ctxt1[i][j]
    return res

def test_matrice_transpose(ctxt1):
    '''
    :param ctxt1: doit être une matrice encryptée
    :return: la matrice encryptée transposée
    '''
    lignes, colonnes = ctxt1.shape[0], ctxt1.shape[1]
    #temp = np.zeros((lignes, colonnes))
    #res = self.encrypt(temp)
    res = np.zeros((colonnes, lignes))
    for i in range(lignes):
        for j in range(colonnes):
            res[j][i] = ctxt1[i][j]
    return res

