import os
import sys

# Run GLCN on Citeseer dataset with optimized hyperparameters
# Best result: ~71% test accuracy (paper reports 72%)
cmd = 'python train.py --dataset citeseer --weight_decay 1e-2 --dropout 0.5 --early_stopping 200'
os.system(cmd)
