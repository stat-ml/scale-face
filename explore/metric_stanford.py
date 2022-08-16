"""
Level five:
generate embeddings
"""


import os
from pathlib import Path
import sys

import torch
import pandas as pd
import numpy as np


sys.path.append('.')
from explore.random_model import ResNet9
from explore.stanford import ffcv_loader_by_df, NUM_CLASSES


def generate_embeddings(model, loader):
    criterion = torch.nn.CrossEntropyLoss()

    embeddings, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            preds = model(x)
            loss = criterion(preds, y)
            print(loss.item(), (torch.argmax(preds, dim=-1) == y).to(torch.float).mean().item())

            embeddings.extend(preds.detach().cpu().numpy())
            labels.extend(y.detach().cpu())

    return np.array(embeddings), np.array(labels)


def build_embeddings(base_dir):
    data_dir = base_dir / 'Stanford_Online_Products'
    small_dir = base_dir / 'small'
    checkpoint_dir = base_dir / 'models'

    model = ResNet9(NUM_CLASSES)

    model.load_state_dict(torch.load(checkpoint_dir / 'resnet9.pth'))
    model.eval().cuda()

    test_df = pd.read_csv(data_dir / 'Ebay_test.txt', delim_whitespace=True, index_col='image_id')
    test_df = test_df[test_df.super_class_id.isin(np.arange(NUM_CLASSES)+1)]
    test_df['labels'] = (test_df.super_class_id) - 1
    print(len(test_df))

    test_loader = ffcv_loader_by_df(
        test_df, small_dir, '/tmp/ds_test.beton', random_order=False, batch_size=500
    )

    x, y = generate_embeddings(model, test_loader)
    np.save('/tmp/stanford_x.npz', x)
    np.save('/tmp/stanford_y.npz', y)


def main():
    base_dir = Path('/home/kirill/data/stanford/')
    # build_embeddings(base_dir)
    


if __name__ == '__main__':
    main()
