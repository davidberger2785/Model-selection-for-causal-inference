import numpy as np
from numpy import linalg as la

import torch
from torch import nn

def softplus(tensor):

    m = nn.Softplus()

    return m(tensor)

def normalized(x):
    return (x-x.mean()) /x.std()

def preprocessing(x, y, u, batch_size):
    data = []
    for k in np.arange(x.shape[0]):
        data.append([x[k], y[k], u[k]])

    data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    return data

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

