"""
Level chetyre:
inference on test
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


def main():
    base_dir = Path('/home/kirill/data/stanford/')
    data_dir = base_dir / 'Stanford_Online_Products'
    small_dir = base_dir / 'small2'
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

    model.eval()
    epoch_losses = []
    correct = []
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            preds = model(x)
            loss = criterion(preds, y)
            epoch_losses.append(loss.item())
            correct.extend(list((torch.argmax(preds, dim=-1) == y).detach().cpu()))
        #
        print(np.mean(epoch_losses))
        print(np.mean(correct))


if __name__ == '__main__':
    main()
