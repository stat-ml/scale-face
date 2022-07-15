from pathlib import Path
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# base_dir = Path("/gpfs/data/gpfs0/k.fedyanin/space/calfw")
base_dir = Path("~/data/faces/cplfw").expanduser()

files = os.listdir(base_dir / 'aligned images')
print(len(files))

idxs = np.random.randint(0, len(files), 10)
print(idxs)

files = [files[i] for i in idxs]
print(files)

for f in files:
    image = Image.open(base_dir / 'aligned images' / f)
    print(image.size)
    