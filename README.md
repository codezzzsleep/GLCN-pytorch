# GLCN-PyTorch

PyTorch implementation of **Graph Learning-Convolutional Networks (GLCN)** for semi-supervised node classification.

This is a PyTorch reimplementation of the original TensorFlow version from the paper:
> **Semi-supervised Learning with Graph Learning-Convolutional Networks**  
> Bo Jiang, Ziyan Zhang, Doudou Lin, Jin Tang, Bin Luo  
> CVPR 2019

## Requirements

- Python 3.10+
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
GLCN-pytorch/
├── data/
│   ├── cora/           # Cora dataset
│   └── citeseer/       # Citeseer dataset
├── glcn/
│   ├── layers.py       # SparseGraphLearn and GraphConvolution layers
│   ├── models.py       # SGLCN model
│   ├── train.py        # Training script
│   ├── utils.py        # Data loading and preprocessing
│   ├── run_cora.py     # Run on Cora dataset
│   └── run_citeseer.py # Run on Citeseer dataset
├── requirements.txt
└── README.md
```

## Usage

### Quick Start

```bash
cd glcn

# Run on Cora dataset
python run_cora.py

# Run on Citeseer dataset
python run_citeseer.py
```

### Custom Training

```bash
python train.py --dataset cora --epochs 5000 --early_stopping 200
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | cora | Dataset: cora or citeseer |
| `--lr1` | 0.005 | Learning rate for graph learning layer |
| `--lr2` | 0.005 | Learning rate for GCN layers |
| `--epochs` | 10000 | Number of training epochs |
| `--hidden_gcn` | 30 | Hidden units in GCN layer |
| `--hidden_gl` | 70 | Hidden units in graph learning layer |
| `--dropout` | 0.6 | Dropout rate |
| `--weight_decay` | 1e-4 | L2 regularization weight |
| `--early_stopping` | 100 | Early stopping patience |
| `--losslr1` | 0.01 | Weight for graph smoothness loss |
| `--losslr2` | 0.0001 | Weight for graph sparsity loss |
| `--seed` | 123 | Random seed |

## Results

| Dataset | This Implementation | Paper |
|---------|---------------------|-------|
| Cora | 84.5% | 85.5% |
| Citeseer | 71.1% | 72.0% |

### Recommended Hyperparameters

**Cora:**
```bash
python train.py --dataset cora --early_stopping 200
```

**Citeseer:**
```bash
python train.py --dataset citeseer --weight_decay 1e-2 --dropout 0.5 --early_stopping 200
```

## Model Architecture

GLCN jointly learns the graph structure and performs graph convolution:

1. **Graph Learning Layer**: Learns an optimal graph structure from node features
2. **Graph Convolution Layers**: 2-layer GCN using the learned graph

The model optimizes two losses:
- **Loss1**: Graph learning loss (smoothness + sparsity regularization)
- **Loss2**: Classification loss (cross-entropy + weight decay)

## Citation

```bibtex
@inproceedings{jiang2019semi,
  title={Semi-supervised learning with graph learning-convolutional networks},
  author={Jiang, Bo and Zhang, Ziyan and Lin, Doudou and Tang, Jin and Luo, Bin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11313--11320},
  year={2019}
}
```

## License

MIT License
