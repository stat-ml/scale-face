import sys

sys.path.append("../detector")
import torch

from retina import load_static_model, detect

device = torch.device("cuda")
net = load_static_model(device)

from dataset import CDataset

ijbc_data = CDataset(
    "/trinity/home/r.karimov/face-evaluation/scripts/data/ijbc_cropped"
)
from torch.utils.data import DataLoader

loader = DataLoader(ijbc_data, batch_size=256, num_workers=3)
from tqdm import tqdm

locs, confs, landmss = [], [], []
for idx, item in tqdm(enumerate(loader)):
    loc, conf, landms = detect(net, item.to(device), device, return_raw=True)
    locs.append(loc.detach().cpu())
    confs.append(loc.detach().cpu())
    landmss.append(loc.detach().cpu())
