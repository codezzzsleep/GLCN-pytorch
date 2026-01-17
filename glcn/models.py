import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SparseGraphLearn, GraphConvolution


class SGLCN(nn.Module):
    """Sparse Graph Learning-Convolutional Network."""
    
    def __init__(self, input_dim, hidden_gl, hidden_gcn, output_dim, 
                 dropout=0.6, weight_decay=1e-4, losslr1=0.01, losslr2=0.0001):
        super(SGLCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_gl = hidden_gl
        self.hidden_gcn = hidden_gcn
        self.output_dim = output_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.losslr1 = losslr1
        self.losslr2 = losslr2
        
        # Graph learning layer
        self.graph_learn = SparseGraphLearn(
            input_dim=input_dim,
            output_dim=hidden_gl,
            dropout=dropout,
            bias=False,
            sparse_inputs=True  # Input features are sparse
        )
        
        # Graph convolution layers
        self.gc1 = GraphConvolution(
            input_dim=input_dim,
            output_dim=hidden_gcn,
            dropout=dropout,
            bias=False,
            sparse_inputs=True  # First layer has sparse inputs
        )
        
        self.gc2 = GraphConvolution(
            input_dim=hidden_gcn,
            output_dim=output_dim,
            dropout=dropout,
            bias=False,
            sparse_inputs=False  # Second layer has dense inputs
        )
    
    def forward(self, x, edge, num_nodes):
        """
        Args:
            x: Input features
            edge: Edge indices [2, num_edges]
            num_nodes: Number of nodes
        Returns:
            output: Logits
            learned_adj: Learned adjacency matrix
        """
        # Graph learning
        h, learned_adj = self.graph_learn(x, edge, num_nodes)
        
        # Graph convolution
        x1 = F.relu(self.gc1(x, learned_adj))
        output = self.gc2(x1, learned_adj)
        
        return output, learned_adj, h
    
    def compute_loss(self, output, learned_adj, h, labels, mask):
        """
        Compute total loss including classification loss and graph learning regularization.
        
        Args:
            output: Model predictions
            learned_adj: Learned adjacency matrix
            h: Features from graph learning layer
            labels: Ground truth labels
            mask: Training mask
        Returns:
            loss: Total loss
            loss1: Graph learning loss
            loss2: Classification loss
        """
        num_nodes = h.size(0)
        
        # Loss 1: Graph learning loss
        loss1 = 0.0
        
        # Weight decay for graph learning layer (using L2 loss = sum(x^2)/2)
        for param in self.graph_learn.parameters():
            loss1 += self.weight_decay * torch.sum(param ** 2) / 2.0
        
        # Graph smoothness term: tr(X^T D X)
        # D = I - S (where S is the learned adjacency)
        indices = learned_adj._indices()
        values = learned_adj._values()
        
        # Create identity matrix
        eye_indices = torch.arange(num_nodes, device=h.device).unsqueeze(0).repeat(2, 1)
        eye_values = torch.ones(num_nodes, device=h.device)
        eye = torch.sparse_coo_tensor(eye_indices, eye_values, (num_nodes, num_nodes))
        
        # D = I - S
        D = torch.sparse_coo_tensor(
            torch.cat([eye_indices, indices], dim=1),
            torch.cat([eye_values, -values]),
            (num_nodes, num_nodes)
        ).coalesce()
        
        # Compute tr(X^T D X)
        Dh = torch.sparse.mm(D, h)
        smoothness = torch.sum(h * Dh)
        loss1 += self.losslr1 * smoothness
        
        # Graph sparsity term: -tr(S^T S)
        # For sparse matrix, tr(S^T S) = sum(S.values^2)
        sparsity = -torch.sum(values ** 2)
        loss1 += self.losslr2 * sparsity
        
        # Loss 2: Classification loss
        loss2 = 0.0
        
        # Weight decay for ONLY the first GCN layer (using L2 loss = sum(x^2)/2)
        for param in self.gc1.parameters():
            loss2 += self.weight_decay * torch.sum(param ** 2) / 2.0
        
        # Cross entropy loss
        loss2 += self.masked_cross_entropy(output, labels, mask)
        
        # Total loss
        loss = loss1 + loss2
        
        return loss, loss1, loss2
    
    def masked_cross_entropy(self, preds, labels, mask):
        """Softmax cross-entropy loss with masking (matching TensorFlow implementation)."""
        # Compute cross entropy for all nodes
        loss = F.cross_entropy(preds, labels.argmax(dim=1), reduction='none')
        
        # Apply mask
        mask = mask.float()
        mask = mask / mask.mean()  # Normalize mask
        loss = loss * mask
        
        return loss.mean()
    
    def accuracy(self, output, labels, mask):
        """Compute accuracy with masking (matching TensorFlow implementation)."""
        preds = output.argmax(dim=1)
        labels_idx = labels.argmax(dim=1)
        correct = preds.eq(labels_idx).float()
        
        # Apply mask
        mask = mask.float()
        mask = mask / mask.mean()
        correct = correct * mask
        
        return correct.mean().item()
