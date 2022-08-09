"""
22'634 classes
120'053 images
"""


import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np


categories = [
    'bicycle', 'cabinet', 'chair', 'coffee_maker', 'fan', 'kettle',
    'lamp', 'mug', 'sofa', 'stapler', 'table', 'toaster'
]

data_dir = Path('/home/kirill/data/stanford/Stanford_Online_Products')

df_train = pd.read_csv(data_dir / 'Ebay_train.txt', delim_whitespace=True, index_col='image_id')
df_test = pd.read_csv(data_dir / 'Ebay_test.txt', delim_whitespace=True, index_col='image_id')

sizes = []
for path in df_train.path:
    size = Image.open(data_dir/path).size
    sizes.append(size[0] / size[1])

plt.hist(np.sort(sizes), bins=40, alpha=0.6)


sizes = []
for path in df_test.path:
    size = Image.open(data_dir/path).size
    sizes.append(size[0] / size[1])

plt.hist(np.sort(sizes), bins=40, alpha=0.6)
plt.show()






