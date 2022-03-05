from argparse import ArgumentParser
import sys
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import auc


sys.path.append('.')
import face_lib.evaluation.plots as plots


parser = ArgumentParser()
parser.add_argument('--test_folder', default='/gpfs/gpfs0/k.fedyanin/space/figures/test')
parser.add_argument('--last_timestamp', action="store_true")
args = parser.parse_args()

FARs = [0.0001, 0.001, 0.05]
rejected_portions = np.arange(0, 0.51, 0.02)

config = {
    'scale': ('mean', 'cosine', 'mean'),
    'pfe': ('PFE', 'cosine', 'mean'),
    'magface': ('mean', 'cosine', 'mean')
}

folder = Path(args.test_folder)

all_results = OrderedDict()

for name, methods in config.items():
    if args.last_timestamp:
        files = os.listdir(folder)
        files = [f for f in files if f.startswith(f'table_{name}')]
        file = sorted(files)[-1]
    else:
        file = f'table_{name}.pt'
    local_results = torch.load(folder / file)
    all_results[name] = local_results[methods]

res_AUCs = OrderedDict()
for method, table in all_results.items():
    res_AUCs[method] = {
        far: auc(rejected_portions, TARs) for far, TARs in table.items()
    }

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plots.plot_TAR_FAR_different_methods(
    all_results,
    rejected_portions,
    res_AUCs,
    title="Template reject verification",
    save_figs_path=os.path.join(folder, f"all_methods_together_{timestamp}.jpg")
)
plt.show()
