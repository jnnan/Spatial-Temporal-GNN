import scipy.sparse as sp
import numpy as np
import torch
import pickle


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def convert_to_gpu(data):
    if torch.cuda.is_available():
        data = data.cuda(0)
    return data


def build_sparse_matrix(L):
    """
    build pytorch sparse tensor from scipy sparse matrix
    reference: https://stackoverflow.com/questions/50665141
    :return:
    """
    shape = L.shape
    i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
    v = torch.FloatTensor(L.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def calculate_random_walk_matrix(adj_mx):
    # adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = torch.tensor(d_mat_inv.dot(adj_mx))
    return random_walk_mx


def create_diffusion_supports(mx, K, num_nodes):
    m_out = []
    for m in mx:
        x0 = torch.eye(num_nodes)
        m_out.append(x0)
        for _ in range(K):
            x0 = torch.sparse.mm(m, x0)
            m_out.append(x0)
    m_out = convert_to_gpu(torch.stack(m_out, 0))
    return m_out


def create_kernel(args):
    _, _, adj = load_graph_data(args['adj_dir'])
    K = args['max_diffusion_step']
    kernals = []
    kernals.append(calculate_random_walk_matrix(adj).T)
    kernals.append(calculate_random_walk_matrix(adj.T).T)
    supports = create_diffusion_supports(kernals, K, args['num_nodes'])
    return supports


def create_diffusion_supports_dense(mx, K, num_nodes):
    m_out = []
    for m in mx:
        x0 = convert_to_gpu(torch.eye(num_nodes))
        m_out.append(x0)
        for _ in range(K):
            x0 = torch.mm(m, x0)
            m_out.append(x0)
    m_out = convert_to_gpu(torch.stack(m_out, 0))
    return m_out
