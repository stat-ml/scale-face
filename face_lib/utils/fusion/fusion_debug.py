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
from face_lib.utils.face_metrics import accuracy_lfw_6000_pairs

import face_lib.utils.fusion.metrics as metrics


# path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
# sys.path.insert(0, path)



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
    x1, x2 = np.array(x1)[:, :512], np.array(x2)[:, :512]
    dist = np.sum(np.square(x1 - x2), axis=1)
    return -dist

def pair_cosine_score(x1, x2):
    x1, x2 = np.array(x1)[:, :512], np.array(x2)[:, :512]
    print("Shapes : ", x1.shape, x2.shape)
    x1, x2 = l2_normalize(x1, axis=1), l2_normalize(x2, axis=1)
    dist = np.sum(x1 * x2, axis=1)
    return dist

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

# def aggregate_templates(templates, features, method):
#     sum_fuse_len = 0
#     for i,t in enumerate(templates):
#         if len(t.indices) > 0:
#             if method == 'mean':
#                 t.feature = l2_normalize(np.mean(features[t.indices], axis=0))
#             if method == 'PFE_fuse':
#                 t.mu, t.sigma_sq = aggregate_PFE(features[t.indices], normalize=True, concatenate=False)
#                 t.feature = t.mu
#             if method == 'PFE_fuse_match':
#                 if not hasattr(t, 'mu'):
#                     t.mu, t.sigma_sq = aggregate_PFE(features[t.indices], normalize=True, concatenate=False)
#                 t.feature = np.concatenate([t.mu, t.sigma_sq])
#         else:
#             t.feature = None
#         if i % 1000 == 0:
#             sys.stdout.write('Fusing templates {}/{}...\t\r'.format(i, len(templates)))
#         sum_fuse_len += len(t.indices)
#     print('')
#     print("Fusion aggregation : ")
#     print("\tMean aggregated size : ", sum_fuse_len / len(templates))

def aggregate_templates_debug(start_indices, features, method):
    gallery_1 = []
    gallery_2 = []

    for idx, (left, right) in enumerate(zip(start_indices[:-1], start_indices[1:])):
        mid = (left + right) // 2
        if (right - left) > 2:
            if method == "mean":
                gallery_1.append(l2_normalize(np.mean(features[np.arange(left, mid)], axis=0)))
                gallery_2.append(l2_normalize(np.mean(features[np.arange(mid, right)], axis=0)))

        if idx % 1000 == 0:
            sys.stdout.write('Fusing templates {}/{}...\t\r'.format(idx, len(start_indices)))

    return gallery_1, gallery_2


    # for i,t in enumerate(templates):
    #     if len(t.indices) > 0:
    #         if method == 'mean':
    #             t.feature = l2_normalize(np.mean(features[t.indices], axis=0))
    #         if method == 'PFE_fuse':
    #             t.mu, t.sigma_sq = aggregate_PFE(features[t.indices], normalize=True, concatenate=False)
    #             t.feature = t.mu
    #         if method == 'PFE_fuse_match':
    #             if not hasattr(t, 'mu'):
    #                 t.mu, t.sigma_sq = aggregate_PFE(features[t.indices], normalize=True, concatenate=False)
    #             t.feature = np.concatenate([t.mu, t.sigma_sq])
    #     else:
    #         t.feature = None
    #     if i % 1000 == 0:
    #         sys.stdout.write('Fusing templates {}/{}...\t\r'.format(i, len(templates)))
    #     sum_fuse_len += len(t.indices)
    # print('')
    # print("Fusion aggregation : ")
    # print("\tMean aggregated size : ", sum_fuse_len / len(templates))

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

    # print("===Image path===")
    # for idx, pth in enumerate(tester.image_paths):
    #     print("\t[", idx, "]",  pth)

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


    sub_id = -1
    identities_starts = []
    for idx, pth in enumerate(tester.image_paths):
        # print(pth.split("/"))
        cur_sub_id = int(pth.split("/")[6])
        if sub_id != cur_sub_id:
            identities_starts.append(idx)
        sub_id = cur_sub_id

    identities_starts.append(len(tester.image_paths))


    mu, sigma_sq = extract_features(
        model,
        tester.image_paths,
        args.batch_size,
        proc_func=proc_func,
        verbose=True,
        device=device,)

    features = np.concatenate([mu, sigma_sq], axis=1)
    gallery_1, gallery_2 = aggregate_templates_debug(identities_starts, features, "mean")
    gallery_1 = np.array(gallery_1)
    gallery_2 = np.array(gallery_2)

    print(np.random.permutation(len(gallery_1)))
    indices_1 = np.random.permutation(len(gallery_1)).astype(int)
    indices_2 = np.random.permutation(len(gallery_2)).astype(int)

    # print(indices_1.shape, gallery_1.shape)

    gallery_1_perm = gallery_1[indices_1]
    gallery_2_perm = gallery_2[indices_2]
    gallery_1 = np.concatenate([gallery_1, gallery_1_perm], axis=0)
    gallery_2 = np.concatenate([gallery_2, gallery_2_perm], axis=0)
    labels = np.concatenate([np.ones(len(gallery_1_perm)), np.zeros(len(gallery_2_perm))], axis=0).astype(np.bool)


    print('---- Average pooling euc')
    # aggregate_templates(tester.verification_templates, features, 'mean')
    TARs, std, FARs = test_verification_debug(gallery_1, gallery_2, labels, force_compare(pair_euc_score), [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5])
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))

    print('---- Average pooling cos')
    # aggregate_templates(tester.verification_templates, features, 'mean')
    TARs, std, FARs = test_verification_debug(gallery_1, gallery_2, labels, force_compare(pair_cosine_score), [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5])
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))


    # print('---- Uncertainty pooling')
    #
    #
    # print('---- MLS comparison')
    # # TARs, std, FARs = tester.test_verification(force_compare(pair_MLS_score))
    # for i in range(len(TARs)):
    #     print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))

    # 0 30 164 174 186

    # same_1 = np.concatenate((mu[0:10], mu[30:40], mu[164:169], mu[174:179]), axis=0)
    # same_2 = np.concatenate((mu[10:20], mu[154:164], mu[169:174], mu[179:184]), axis=0)
    #
    # diff_1 = np.concatenate((mu[0:10], mu[30:40], mu[164:169], mu[176:186]), axis=0)
    # diff_2 = np.concatenate((mu[174:184], mu[10:20], mu[75:80], mu[164:174]), axis=0)
    #
    # compare_func = force_compare(pair_cosine_score)
    #
    # same_scores = compare_func(same_1, same_2)
    # diff_scores = compare_func(diff_1, diff_2)
    #
    # print("Same : ", same_scores.mean())
    # print(same_scores)
    # print("Diff : ", diff_scores.mean())
    # print(diff_scores)

    # def mean_cosine_dist(l, r):


    # names = (
    #     tester.image_paths[15:25],
    #     tester.image_paths[65:75],
    #     tester.image_paths[165:173],
    #     tester.image_paths[174:185])
    #
    # simp_process = lambda x : extract_features()
    # embeddings = map()




    # lfw_path = "/gpfs/gpfs0/r.karimov/lfw/data_aligned_112_112"
    # lfw_pairs_txt_path = "/gpfs/gpfs0/r.karimov/lfw/pairs_val_6000.txt"
    # lfw_res = accuracy_lfw_6000_pairs(
    #     model["backbone"],
    #     model["head"],
    #     lfw_path,
    #     lfw_pairs_txt_path,
    #     N=6000,
    #     n_folds=10,
    #     device=torch.device("cuda:0"))
    #
    # print("LFW res : ", lfw_res)
    # raise RuntimeError("==DEBUG==")

    # mu = np.random.randn(len(tester.image_paths), 512)
    # sigma_sq = np.random.randn(len(tester.image_paths), 512)

    # features = np.concatenate([mu, sigma_sq], axis=1)
    #
    # print('---- Average pooling')
    # aggregate_templates(tester.verification_templates, features, 'mean')
    # TARs, std, FARs = tester.test_verification(force_compare(pair_cosine_score))
    # for i in range(len(TARs)):
    #     print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))
    #
    # print('---- Uncertainty pooling')
    # aggregate_templates(tester.verification_templates, features, 'PFE_fuse')
    # TARs, std, FARs = tester.test_verification(force_compare(pair_cosine_score))
    # for i in range(len(TARs)):
    #     print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))
    #
    # print('---- MLS comparison')
    # aggregate_templates(tester.verification_templates, features, 'PFE_fuse_match')
    # TARs, std, FARs = tester.test_verification(force_compare(pair_MLS_score))
    # for i in range(len(TARs)):
    #     print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))

def test_verification_debug(features_1, features_2, label_vec, compare_func, FARs=None):

    FARs = [1e-5, 1e-4, 1e-3, 1e-2] if FARs is None else FARs

    # templates1 = self.verification_G1_templates
    # templates2 = self.verification_G2_templates

    # print("Temp1 : ", len(templates1), "temp2 : ", len(templates2))
    # nans1 = sum(temp.feature is None for temp in templates1)
    # nans2 = sum(temp.feature is None for temp in templates2)
    # print(f"Nans :")
    # print(f"1 : {nans1} / {len(templates1)}")
    # print(f"2 : {nans2} / {len(templates2)}")

    # not_nan_1 = np.array([template.feature is not None for template in templates1])
    # not_nan_2 = np.array([template.feature is not None for template in templates2])
    # not_nan = not_nan_1 & not_nan_2

    # print(f"Ignored {not_nan.shape[0] - not_nan.sum()} / {not_nan.shape[0]}")
    # print("Length before : ", len(templates1), len(templates2))
    # templates1 = templates1[not_nan]
    # templates2 = templates2[not_nan]
    # print("Length after : ", len(templates1), len(templates2))

    # print("Features1 : ", templates1.shape, templates1[0].feature)
    # print("Features2 : ", templates2.shape, templates2[0].feature)

    # features1 = [t.feature for t in templates1]
    # features2 = [t.feature for t in templates2]
    # labels1 = np.array([t.label for t in templates1])
    # labels2 = np.array([t.label for t in templates2])

    score_vec = compare_func(features_1, features_2)
    # label_vec = labels1 == labels2

    # idx = [0, 12, 1222, 12222, 122222, 10000000, 15000000, 10000001]
    # idx = [0, 1, 2, 3, 4]
    idx = [0, 1, int(len(score_vec) * 0.2), int(len(score_vec) * 0.5), int(len(score_vec) * 0.6), int(len(score_vec)) - 1, int(len(score_vec)) - 2]
    print(f"True labels : {sum(label_vec)} / {len(label_vec)} len_score == len_label : ({len(score_vec) == len(label_vec)})")
    print("Scores vec : ", score_vec.shape, score_vec[idx])
    print("Label_vec : ", label_vec.shape, label_vec[idx])

    tars, fars, thresholds = metrics.ROC(score_vec, label_vec, FARs=FARs)

    print("Tars : ", tars.shape, tars)
    print("Fars : ", fars.shape, fars)
    print("Thresholds : ", thresholds.shape, thresholds)

    # There is no std for IJB-C
    std = [0. for t in tars]

    return tars, std, fars

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
