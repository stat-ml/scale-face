import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm as tqdm

path = str(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from face_lib.datasets import IJBDataset, IJBATest, IJBCTest
from face_lib import models as mlib, utils
from face_lib.utils import cfg
from face_lib.utils.imageprocessing import preprocess
from face_lib.utils.fusion_metrics import pair_euc_score, pair_cosine_score, pair_MLS_score
from face_lib.utils.fusion_metrics import l2_normalize, aggregate_PFE


def aggregate_templates(templates, features, method):
    sum_fuse_len = 0
    number_of_templates = 0
    for i, t in enumerate(templates):
        if len(t.indices) > 0:
            if method == "random":
                t.feature = l2_normalize(features[np.random.choice(t.indices)])
            if method == 'mean':
                t.feature = l2_normalize(np.mean(features[t.indices], axis=0))
            if method == 'PFE_fuse':
                t.mu, t.sigma_sq = aggregate_PFE(features[t.indices], normalize=True, concatenate=False)
                t.feature = t.mu
            if method == 'PFE_fuse_match':
                if not hasattr(t, 'mu'):
                    t.mu, t.sigma_sq = aggregate_PFE(features[t.indices], normalize=True, concatenate=False)
                t.feature = np.concatenate([t.mu, t.sigma_sq])
        else:
            t.feature = None
        if i % 1000 == 0:
            sys.stdout.write('Fusing templates {}/{}...\t\r'.format(i, len(templates)))

        sum_fuse_len += len(t.indices)
        number_of_templates += int(len(t.indices) > 0)
    print('')
    print("Mean aggregated size : ", sum_fuse_len / number_of_templates)


def force_compare(compare_func, verbose=False):
    def compare(t1, t2):
        score_vec = np.zeros(len(t1))
        for i in range(len(t1)):
            if t1[i] is None or t2[i] is None:
                score_vec[i] = -9999
            else:
                score_vec[i] = compare_func(t1[i][None], t2[i][None])
            if verbose and i % 1000 == 0:
                sys.stdout.write('Matching pair {}/{}...\t\r'.format(i, len(t1)))
        if verbose:
            print('')
        return score_vec

    return compare


def extract_features(
        backbone, head, images, batch_size, proc_func=None,
        verbose=False, device=torch.device("cpu")):
    num_images = len(images)
    mu = []
    sigma_sq = []
    start_time = time.time()
    for start_idx in tqdm(range(0, num_images, batch_size)):
        if verbose:
            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
            sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r'
                             % (num_images, start_idx, elapsed_time))
        end_idx = min(num_images, start_idx + batch_size)
        images_batch = images[start_idx:end_idx]

        batch = proc_func(images_batch)
        batch = (
            torch.from_numpy(batch)
                .permute(0, 3, 1, 2)
                .to(device)
        )
        feature, sig_feat = backbone(batch)
        log_sig_sq = head(sig_feat)
        mu.append(np.array(feature.detach().cpu()))
        sigma_sq.append(np.array(log_sig_sq.exp().detach().cpu()))

    mu = np.concatenate(mu, axis=0)
    sigma_sq = np.concatenate(sigma_sq, axis=0)

    if verbose:
        print('')
    return mu, sigma_sq


def eval_fusion_ijb(
        backbone,
        head,
        dataset_path,
        protocol_path,
        batch_size=64,
        protocol="ijbc",
        device=torch.device("cpu"),):

    proc_func = lambda images: preprocess(images, [112, 112], is_training=False)

    testset = IJBDataset(dataset_path)
    if protocol == 'ijba':
        tester = IJBATest(testset['abspath'].values)
        tester.init_proto(protocol_path)
    elif protocol == 'ijbc':
        tester = IJBCTest(testset['abspath'].values)
        tester.init_proto(protocol_path)
    else:
        raise ValueError('Unkown protocol. Only accept "ijba" or "ijbc".')

    backbone = backbone.eval().to(device)
    head = head.eval().to(device)

    mu, sigma_sq = extract_features(
        backbone,
        head,
        tester.image_paths,
        batch_size,
        proc_func=proc_func,
        verbose=True,
        device=device,)

    features = np.concatenate([mu, sigma_sq], axis=1)

    print('---- Random pooling (Cosine distance)')
    aggregate_templates(tester.verification_templates, features, 'random')
    TARs, std, FARs = tester.test_verification(force_compare(pair_cosine_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))
    print()

    print('---- Average pooling (Cosine distance)')
    aggregate_templates(tester.verification_templates, features, 'mean')
    TARs, std, FARs = tester.test_verification(force_compare(pair_cosine_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))
    print()

    print('---- Uncertainty pooling (Cosine distance)')
    aggregate_templates(tester.verification_templates, features, 'PFE_fuse')
    TARs, std, FARs = tester.test_verification(force_compare(pair_cosine_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))
    print()

    print('---- Uncertainty pooling (MLS distance)')
    aggregate_templates(tester.verification_templates, features, 'PFE_fuse_match')
    TARs, std, FARs = tester.test_verification(force_compare(pair_MLS_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", help="The path to the pre-trained model directory",
                        type=str, default=None)
    parser.add_argument("--protocol", help="The dataset to test",
                        type=str, default='ijbc')
    parser.add_argument("--dataset_path", help="The path to the IJB-A dataset directory",
                        type=str, default='data/ijba_mtcnncaffe_aligned')
    parser.add_argument("--protocol_path", help="The path to the IJB-A protocol directory",
                        type=str, default='proto/IJB-A')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=64)
    parser.add_argument("--config_path", help="The path to config .yaml file",
                        type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0")

    model_args = cfg.load_config(args.config_path)
    backbone = mlib.model_dict[model_args.backbone["name"]](
        **utils.pop_element(model_args.backbone, "name"))
    head = mlib.heads[model_args.head.name](
        **utils.pop_element(model_args.head, "name"))

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["uncertain"])

    eval_fusion_ijb(
        backbone,
        head,
        args.dataset_path,
        args.protocol_path,
        batch_size=64,
        protocol="ijbc",
        device=torch.device("cuda:0"),)
