import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_data, preprocess_adj, preprocess_features, to_torch
from models import SGLCN


def train(args):
    """Train SGLCN model."""
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
    
    # Preprocess
    features = preprocess_features(features)
    adj_normalized, edge = preprocess_adj(adj)
    
    # Get dimensions
    num_nodes = features.shape[0]
    input_dim = features.shape[1]
    output_dim = y_train.shape[1]
    
    print(f"Nodes: {num_nodes}, Features: {input_dim}, Classes: {output_dim}")
    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    # Convert to torch tensors
    device = torch.device('cuda' if args.cuda else 'cpu')
    features, labels, train_mask, val_mask, test_mask, edge = to_torch(
        adj_normalized, features, y_train + y_val + y_test, 
        train_mask, val_mask, test_mask, edge, device
    )
    
    # Create model
    model = SGLCN(
        input_dim=input_dim,
        hidden_gl=args.hidden_gl,
        hidden_gcn=args.hidden_gcn,
        output_dim=output_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        losslr1=args.losslr1,
        losslr2=args.losslr2
    ).to(device)
    
    # Create optimizers
    params_gl = list(model.graph_learn.parameters())
    params_gcn = list(model.gc1.parameters()) + list(model.gc2.parameters())
    
    optimizer1 = optim.Adam(params_gl, lr=args.lr1)
    optimizer2 = optim.Adam(params_gcn, lr=args.lr2)
    
    # Learning rate schedulers (matching TensorFlow's exponential_decay with staircase=True)
    # TF: lr * decay_rate^(global_step // decay_steps)
    # This means lr decays by 0.9 every 100 steps
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=100, gamma=0.9)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=100, gamma=0.9)
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 0
    test_acc_list = []
    
    print("\nStart training...")
    for epoch in range(args.epochs):
        t = time.time()
        
        # Training
        model.train()
        
        # Forward pass
        output, learned_adj, h = model(features, edge, num_nodes)
        loss, loss1, loss2 = model.compute_loss(output, learned_adj, h, labels, train_mask)
        train_acc = model.accuracy(output, labels, train_mask)
        
        # Backward pass - matching TensorFlow's separate optimization
        # In TF: opt_op1 minimizes loss1 w.r.t. vars1 only
        #        opt_op2 minimizes loss2 w.r.t. vars2 only
        
        # Step 1: Update graph learning layer with loss1
        optimizer1.zero_grad()
        loss1.backward(retain_graph=True)
        optimizer1.step()
        
        # Step 2: Update GCN layers with loss2
        # Need to zero gradients for graph learning layer to prevent accumulation
        optimizer2.zero_grad()
        for param in model.graph_learn.parameters():
            if param.grad is not None:
                param.grad.zero_()
        loss2.backward()
        optimizer2.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            output, learned_adj, h = model(features, edge, num_nodes)
            val_loss, _, _ = model.compute_loss(output, learned_adj, h, labels, val_mask)
            val_acc = model.accuracy(output, labels, val_mask)
            test_acc = model.accuracy(output, labels, test_mask)
            test_acc_list.append(test_acc)
        
        # Print progress
        print(f"Epoch: {epoch+1:04d} "
              f"train_loss: {loss.item():.5f} "
              f"train_acc: {train_acc:.5f} "
              f"val_loss: {val_loss.item():.5f} "
              f"val_acc: {val_acc:.5f} "
              f"test_acc: {test_acc:.5f} "
              f"time: {time.time()-t:.5f}s")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        
        if patience >= args.early_stopping:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        # Step schedulers every epoch (matching TensorFlow's staircase decay)
        scheduler1.step()
        scheduler2.step()
    
    print("\nOptimization Finished!")
    print("----------------------------------------------")
    if len(test_acc_list) > 100:
        print(f"Final result (100 epochs before stopping): {test_acc_list[-101]:.5f}")
    else:
        print(f"Final result: {test_acc_list[-1]:.5f}")
    print("----------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', 
                        help='Dataset: cora or citeseer')
    parser.add_argument('--lr1', type=float, default=0.005,
                        help='Learning rate for graph learning layer')
    parser.add_argument('--lr2', type=float, default=0.005,
                        help='Learning rate for GCN layers')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs to train')
    parser.add_argument('--hidden_gcn', type=int, default=30,
                        help='Number of hidden units in GCN')
    parser.add_argument('--hidden_gl', type=int, default=70,
                        help='Number of hidden units in graph learning')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--early_stopping', type=int, default=100,
                        help='Tolerance for early stopping')
    parser.add_argument('--losslr1', type=float, default=0.01,
                        help='Weight for graph smoothness loss')
    parser.add_argument('--losslr2', type=float, default=0.0001,
                        help='Weight for graph sparsity loss')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA if available')
    
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    
    train(args)
