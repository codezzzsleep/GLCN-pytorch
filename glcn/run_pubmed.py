import os
import sys

# Run GLCN on Pubmed dataset
cmd = 'python train.py --dataset pubmed --hidden_gcn 30 --hidden_gl 70'
os.system(cmd)
