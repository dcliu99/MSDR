import os
from typing import Union, List

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigsh
from torch import sparse

from utils.util import load_pickle

def normalized_laplacian(w: np.ndarray) -> np.matrix:
    d = np.array(w.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    print(d,d_inv_sqrt)
    d_mat_inv_sqrt = np.eye(d_inv_sqrt.shape[0]) * d_inv_sqrt.shape
    return np.identity(w.shape[0]) - d_mat_inv_sqrt.dot(w).dot(d_mat_inv_sqrt)

def random_walk_matrix(w) -> np.matrix:
    d = np.array(w.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.eye(d_inv.shape[0]) * d_inv
    return d_mat_inv.dot(w)


def reverse_random_walk_matrix(w) -> sp.coo_matrix:
    return random_walk_matrix(w.T)


def sparse_scipy2torch(w: sp.coo_matrix):
    shape = w.shape
    i = torch.tensor(np.vstack((w.row, w.col)).astype(int)).long()
    v = torch.tensor(w.data).float()
    return sparse.FloatTensor(i, v, torch.Size(shape))


def scaled_laplacian(w: np.ndarray, lambda_max: Union[float, None] = 2., undirected: bool = True) -> sp.coo_matrix:
    if undirected:
        w = np.maximum.reduce([w, w.T])
    lp = normalized_laplacian(w)
    if lambda_max is None:
        lambda_max, _ = eigsh(lp.todense(), 1, which='LM')
        lambda_max = lambda_max[0]
    lp = sp.csr_matrix(lp)
    m, _ = lp.shape
    i = sp.identity(m, format='csr', dtype=lp.dtype)
    lp = (2 / lambda_max * lp) - i
    return lp.astype(np.float32).tocoo()


def cheb_poly_approx(lp, k_hop, n):
    l0, l1 = np.identity(n), np.copy(lp)

    if k_hop > 1:
        l_list = [np.copy(l0), np.copy(l1)]
        for i in range(k_hop - 2):
            ln = 2 * np.matmul(lp, l1) - l0
            l_list.append(np.copy(ln))
            l0, l1 = np.copy(l1), np.copy(ln)
        return np.stack(l_list, axis=1)
    elif k_hop == 1:
        return l0.reshape((n, 1, n))
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{k_hop}".')


def first_approx(w, n):
    a = w + np.identity(n)
    d = np.sum(a, axis=1)
    sinv_d = np.sqrt(np.linalg.inv(np.diag(d)))
    # refer to Eq.5
    return np.identity(n) + np.matmul(np.matmul(sinv_d, a), sinv_d)


def sparse_scipy2torch(w: sp.coo_matrix):
    shape = w.shape
    i = torch.tensor(np.vstack((w.row, w.col)).astype(int)).long()
    v = torch.tensor(w.data).float()
    return sparse.FloatTensor(i, v, torch.Size(shape))


def load_graph_data(dataset: str, graph_type: str) -> List[sp.coo_matrix]:
    _, _, adj_mx = load_pickle(os.path.join('data', dataset, 'adj_mx.pkl'))
    if graph_type == "scalap":
        adj = [scaled_laplacian(adj_mx)]
    elif graph_type == "normlap":
        adj = [normalized_laplacian(adj_mx)]
    elif graph_type == "transition":
        adj = [random_walk_matrix(adj_mx)]
    elif graph_type == "doubletransition":
        adj = [random_walk_matrix(adj_mx), reverse_random_walk_matrix(adj_mx)]
    elif graph_type == "identity":
        adj = [sp.identity(adj_mx.shape[0], dtype=np.float32, format='coo')]
    else:
        raise ValueError(f"graph type {graph_type} not defined")
    return adj
