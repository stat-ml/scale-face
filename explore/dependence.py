import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append('.')
from explore.cross import load_config, get_pairs, Preprocessor, load_model, Inferencer


def main():
    configs = {
        'ScaleFace': './configs/cross/scale.yaml',
    }
    dataset = 'cplfw'

    args = load_config(configs['ScaleFace'])
    args.images_path = f'{dataset}/aligned images'
    args.short = False
    data_directory = Path(args.data_directory)
    pairs = get_pairs(data_directory, dataset, short=args.short)
    photo_list = np.unique(pairs.photo_1.to_list() + pairs.photo_2.to_list())

    preprocessor = Preprocessor(data_directory / args.images_path, is_training=True)
    checkpoint_path = data_directory / args.checkpoint_path
    model = load_model(args.uncertainty_type, checkpoint_path)

    inferencer = Inferencer(preprocessor, model, 100)
    multi_mus = []
    multi_sigmas = []
    for _ in range(5):
        mus, sigmas = inferencer(photo_list)
        multi_mus.append(torch.stack(tuple(mus.values())))
        multi_sigmas.append(torch.tensor(tuple(sigmas.values())))

    multi_mus = torch.stack(multi_mus).numpy()
    multi_sigmas = torch.stack(multi_sigmas).numpy()

    sigmas = np.linalg.norm(np.std(multi_mus, axis=0), axis=-1)
    mean_scale = np.mean(multi_sigmas, axis=0)

    # mean_scale = mean_scale[sigmas < 0.1]
    # sigmas = sigmas[sigmas < 0.1]

    plt.scatter(mean_scale, sigmas, alpha=0.5, s=5)
    plt.show()




if __name__ == '__main__':
    main()