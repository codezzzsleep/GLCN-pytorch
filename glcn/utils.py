import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch


def load_data(dataset_str, path="../data/"):
    """Load dataset."""
    path = path + dataset_str + "/"
    
    if dataset_str == "cora":
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    elif dataset_str == "citeseer":
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_test = sio.loadmat(path + "test.mat")
        idx_test = idx_test['array'].flatten()
        idx_train = range(120)
        idx_val = range(120, 620)
    else:  # pubmed and other datasets
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_test = sio.loadmat(path + "test.mat")
        idx_test = idx_test['matrix'].flatten()
        idx_train = range(60)
        idx_val = range(200, 500)
    
    # Make adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # Create masks
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    
    # Create label matrices
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix and conversion to edge list."""
    # Add self-loops
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    
    # Convert to edge list (matching TensorFlow's np.nonzero format)
    # np.nonzero returns (row_indices, col_indices) in row-major order
    adj_dense = adj_normalized.todense()
    nonzero = np.nonzero(adj_dense)
    edge = np.array(nonzero)
    
    return adj_normalized, edge


def preprocess_features(features):
    """Row-normalize feature matrix."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def to_torch(adj, features, labels, train_mask, val_mask, test_mask, edge, device):
    """Convert numpy arrays to torch tensors."""
    # Convert features - keep as sparse tensor (matching TensorFlow)
    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features).to(device)
    else:
        # Convert dense to sparse
        features = torch.FloatTensor(np.array(features)).to(device)
    
    # Convert labels
    labels = torch.FloatTensor(labels).to(device)
    
    # Convert masks
    train_mask = torch.BoolTensor(train_mask).to(device)
    val_mask = torch.BoolTensor(val_mask).to(device)
    test_mask = torch.BoolTensor(test_mask).to(device)
    
    # Convert edge list
    edge = torch.LongTensor(edge).to(device)
    
    return features, labels, train_mask, val_mask, test_mask, edge
