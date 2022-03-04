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

FARs = [0.0001, 0.001, 0.05]
rejected_portions = np.arange(0, 0.51, 0.02)

config = {
    'scale': ('mean', 'cosine', 'mean'),
    'pfe': ('mean', 'cosine', 'mean'),
    'magface': ('mean', 'cosine', 'mean')
}

folder = Path('/gpfs/gpfs0/k.fedyanin/space/figures/test')
save_fig_path = "/gpfs/gpfs0/k.fedyanin/space/figures/test"

all_results = OrderedDict()

for name, methods in config.items():
    local_results = torch.load(folder / f'table_{name}.pt')
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
    save_figs_path=os.path.join(save_fig_path, f"all_methods_together_{timestamp}.jpg")
)
plt.show()
