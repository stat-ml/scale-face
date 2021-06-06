import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path

# from utils import utils
# from face_lib.datasets.ijb import Dataset
# from imageprocessing import preprocess

# from network import Network


# from evaluation.ijbc import IJBCTest
from tqdm import tqdm as tqdm

path = str(Path(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute()).parent.absolute())
# print(path)
sys.path.insert(0, path)
from face_lib.datasets import IJBDataset, IJBATest, IJBCTest
# from face_lib.datasets.ijba import IJBATest
from face_lib import models as mlib, utils
from face_lib.utils import cfg
from face_lib.utils.imageprocessing import preprocess

def l2_normalize(x, axis=None, eps=1e-8):
    x = x / (eps + np.linalg.norm(x, axis=axis))
    return x


def aggregate_PFE(x, sigma_sq=None, normalize=True, concatenate=False):
    if sigma_sq is None:
        D = int(x.shape[1] / 2)
        mu, sigma_sq = x[:, :D], x[:, D:]
    else:
        mu = x
    attention = 1. / sigma_sq
    attention = attention / np.sum(attention, axis=0, keepdims=True)

    mu_new = np.sum(mu * attention, axis=0)
    sigma_sq_new = np.min(sigma_sq, axis=0)

    if normalize:
        mu_new = l2_normalize(mu_new)

    if concatenate:
        return np.concatenate([mu_new, sigma_sq_new])
    else:
        return mu_new, sigma_sq_new

def pair_euc_score(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    dist = np.sum(np.square(x1 - x2), axis=1)
    return -dist

def pair_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    if sigma_sq1 is None:
        x1, x2 = np.array(x1), np.array(x2)
        assert sigma_sq2 is None, 'either pass in concated features, or mu, sigma_sq for both!'
        D = int(x1.shape[1] / 2)
        mu1, sigma_sq1 = x1[:,:D], x1[:,D:]
        mu2, sigma_sq2 = x2[:,:D], x2[:,D:]
    else:
        x1, x2 = np.array(x1), np.array(x2)
        sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
        mu1, mu2 = x1, x2
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1)
    return -dist

def aggregate_templates(templates, features, method):
    for i,t in enumerate(templates):
        if len(t.indices) > 0:
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
    print('')


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

def get_item_by_the_path(path: str, proc_func) -> np.ndarray:
    # abspath = list(self.data.query(f"path == '{path}'")["abspath"])[0:1]
    if len(path) == 0:
        raise NotImplementedError
    print("1 Path : ", path)
    image = proc_func(path)
    return image[0]

def process_images_batch(model, images_names, proc_func):
    # batch = [get_item_by_the_path(img_path, proc_func) for img_path in images_names]
    batch = proc_func(images_names)
    print(batch[0].shape)

def extract_features(
        model, images, batch_size, proc_func=None,
        verbose=False, device=torch.device("cpu")):
    num_images = len(images)
    # num_features = self.mu.shape[1]
    # mu = np.ndarray((num_images, num_features), dtype=np.float32)
    # sigma_sq = np.ndarray((num_images, num_features), dtype=np.float32)
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
        # if proc_func:
        #     images_batch = proc_func(images_batch)

        # feed_dict = {self.images: images_batch,
        #              self.phase_train: False,
        #              self.keep_prob: 1.0}
        # mu[start_idx:end_idx], sigma_sq[start_idx:end_idx] = self.sess.run([self.mu, self.sigma_sq],
        #                                                                    feed_dict=feed_dict)
        # process_images_batch(model, images_batch, proc_func)
        batch = proc_func(images_batch)
        batch = (
            torch.from_numpy(batch)
            .permute(0, 3, 1, 2)
            .to(device)
        )
        feature, sig_feat = model["backbone"](batch)
        log_sig_sq = model["head"](sig_feat)
        mu.append(np.array(feature.detach().cpu()))
        sigma_sq.append(np.array(log_sig_sq.exp().detach().cpu()))

    mu = np.concatenate(mu, axis=0)
    sigma_sq = np.concatenate(sigma_sq, axis=0)

    print(f"Mu : {mu.shape}; sigma_sq : {sigma_sq.shape}")
    print(f"Mu : {mu[[0, 5], :]}")
    print(f"Sigma : {sigma_sq[[0, 5], :]}")

    if verbose:
        print('')
    return mu, sigma_sq


def main(args):
    model_args = cfg.load_config(args.config_path)
    # network = Network()
    # network.load_model(args.model_dir)

    # model_path = "/trinity/home/r.kail/Faces/Probabilistic-Face-Embeddings/checkpoints/PFE_sphere64_casia_am"
    # # network_config = importlib.load_source('network_config', os.path.join(model_path, 'config.py'))
    # network_config = imp.load_source('network_config', os.path.join(model_path, 'config.py'))
    # proc_func = lambda x: preprocess(x, network_config, False)

    proc_func = lambda images: preprocess(images, [112, 112], is_training=False)

    testset = IJBDataset(args.dataset_path)
    if args.protocol == 'ijba':
        tester = IJBATest(testset['abspath'].values)
        tester.init_proto(args.protocol_path)
    elif args.protocol == 'ijbc':
        tester = IJBCTest(testset['abspath'].values)
        tester.init_proto(args.protocol_path)
        # raise NotImplementedError("This protocol is not implemented yet")
    else:
        raise ValueError('Unkown protocol. Only accept "ijba" or "ijbc".')

    # backbone = mlib.model_dict[model_args.backbone["name"]](
    #     **utils.pop_element(model_args.backbone, "name"))
    # head = mlib.heads[model_args.head.name](
    #     **utils.pop_element(model_args.head, "name"))
    # backbone.load_state_dict(torch.load(model_args.))

    device = torch.device("cuda:0")
    model = {
        "backbone": mlib.model_dict[model_args.backbone["name"]](
            **utils.pop_element(model_args.backbone, "name")),
        "head": mlib.heads[model_args.head.name](
            **utils.pop_element(model_args.head, "name"))
    }
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    model["backbone"].load_state_dict(checkpoint["backbone"])
    model["head"].load_state_dict(checkpoint["uncertain"])

    model["backbone"] = model["backbone"].eval().to(device)
    model["head"] = model["head"].eval().to(device)

    mu, sigma_sq = extract_features(
        model,
        tester.image_paths,
        args.batch_size,
        proc_func=proc_func,
        verbose=True,
        device=device,)

    features = np.concatenate([mu, sigma_sq], axis=1)

    print('---- Average pooling')
    aggregate_templates(tester.verification_templates, features, 'mean')
    TARs, std, FARs = tester.test_verification(force_compare(pair_euc_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))

    print('---- Uncertainty pooling')
    aggregate_templates(tester.verification_templates, features, 'PFE_fuse')
    TARs, std, FARs = tester.test_verification(force_compare(pair_euc_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))


    print('---- MLS comparison')
    aggregate_templates(tester.verification_templates, features, 'PFE_fuse_match')
    TARs, std, FARs = tester.test_verification(force_compare(pair_MLS_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", help="The path to the pre-trained model directory",
                        type=str, default=None)
    parser.add_argument("--protocol", help="The dataset to test",
                        type=str, default='ijba')
    parser.add_argument("--dataset_path", help="The path to the IJB-A dataset directory",
                        type=str, default='data/ijba_mtcnncaffe_aligned')
    parser.add_argument("--protocol_path", help="The path to the IJB-A protocol directory",
                        type=str, default='proto/IJB-A')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=64)
    parser.add_argument("--config_path", help="The path to config .yaml file",
                        type=str, default=None)
    args = parser.parse_args()
    main(args)
