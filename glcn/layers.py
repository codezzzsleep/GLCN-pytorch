import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sparse_dropout(x, keep_prob, training):
    """
    Dropout for sparse tensors (matching TensorFlow's sparse_dropout).
    Only drops non-zero elements.
    
    Args:
        x: Sparse tensor
        keep_prob: Probability to keep (1 - dropout_rate)
        training: Whether in training mode
    """
    if not training or keep_prob >= 1.0:
        return x
    
    # Get non-zero values
    indices = x._indices()
    values = x._values()
    
    # Create dropout mask for non-zero elements only
    random_tensor = torch.rand(values.shape, device=values.device)
    dropout_mask = random_tensor < keep_prob
    
    # Apply mask and scale
    new_values = values * dropout_mask.float() / keep_prob
    
    return torch.sparse_coo_tensor(indices, new_values, x.size())


class SparseGraphLearn(nn.Module):
    """Sparse Graph learning layer."""
    
    def __init__(self, input_dim, output_dim, dropout=0., bias=False, sparse_inputs=True):
        super(SparseGraphLearn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.bias = bias
        self.sparse_inputs = sparse_inputs
        
        # Initialize weights
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.a = nn.Parameter(torch.FloatTensor(output_dim, 1))
        
        if self.bias:
            self.bias_param = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias_param', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Glorot initialization."""
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        stdv_a = math.sqrt(6.0 / (self.a.size(0) + self.a.size(1)))
        self.a.data.uniform_(-stdv_a, stdv_a)
        if self.bias:
            self.bias_param.data.zero_()
    
    def forward(self, x, edge, num_nodes):
        """
        Args:
            x: Input features (sparse or dense)
            edge: Edge indices [2, num_edges]
            num_nodes: Number of nodes
        Returns:
            h: Transformed features
            adj: Learned sparse adjacency matrix
        """
        # Dropout - use sparse dropout for sparse inputs
        if self.training and self.dropout > 0:
            if self.sparse_inputs and x.is_sparse:
                x = sparse_dropout(x, 1 - self.dropout, training=True)
            else:
                x = F.dropout(x, self.dropout, training=True)
        
        # Linear transformation
        if x.is_sparse:
            h = torch.sparse.mm(x, self.weight)
        else:
            h = torch.mm(x, self.weight)
        
        # Graph learning: compute edge weights
        edge_h_i = h[edge[0]]  # Features of source nodes
        edge_h_j = h[edge[1]]  # Features of target nodes
        edge_v = torch.abs(edge_h_i - edge_h_j)  # Absolute difference
        edge_v = F.relu(torch.mm(edge_v, self.a)).squeeze()  # [num_edges]
        
        # Create sparse adjacency matrix with learned weights
        edge_index = edge.t()  # [num_edges, 2]
        adj = torch.sparse_coo_tensor(
            edge_index.t(), 
            edge_v, 
            (num_nodes, num_nodes)
        )
        
        # Sparse softmax normalization
        adj = self.sparse_softmax(adj)
        
        return h, adj
    
    def sparse_softmax(self, adj):
        """Apply softmax to sparse tensor row-wise (matching TensorFlow's tf.sparse_softmax)."""
        indices = adj._indices()
        values = adj._values()
        shape = adj.size()
        
        # Get row indices
        row_indices = indices[0]
        
        # Compute max per row for numerical stability using scatter
        max_per_row = torch.full((shape[0],), float('-inf'), device=values.device, dtype=values.dtype)
        max_per_row.scatter_reduce_(0, row_indices, values, reduce='amax', include_self=False)
        max_per_row = torch.where(torch.isinf(max_per_row), torch.zeros_like(max_per_row), max_per_row)
        
        # Subtract max and exponentiate
        values_stable = values - max_per_row[row_indices]
        exp_values = torch.exp(values_stable)
        
        # Compute sum per row
        sum_per_row = torch.zeros(shape[0], device=values.device, dtype=values.dtype)
        sum_per_row.scatter_add_(0, row_indices, exp_values)
        
        # Normalize
        normalized_values = exp_values / (sum_per_row[row_indices] + 1e-16)
        
        return torch.sparse_coo_tensor(indices, normalized_values, shape)


class GraphConvolution(nn.Module):
    """Graph convolution layer."""
    
    def __init__(self, input_dim, output_dim, dropout=0., bias=False, sparse_inputs=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Glorot initialization."""
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
    
    def forward(self, x, adj):
        """
        Args:
            x: Input features
            adj: Adjacency matrix (sparse)
        Returns:
            output: Convolved features
        """
        # Dropout - use sparse dropout for sparse inputs
        if self.training and self.dropout > 0:
            if self.sparse_inputs and x.is_sparse:
                x = sparse_dropout(x, 1 - self.dropout, training=True)
            else:
                x = F.dropout(x, self.dropout, training=True)
        
        # Linear transformation
        if x.is_sparse:
            support = torch.sparse.mm(x, self.weight)
        else:
            support = torch.mm(x, self.weight)
        
        # Graph convolution
        output = torch.sparse.mm(adj, support)
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        return output
