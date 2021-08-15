import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import argparse
import cv2
import numpy as np
import torch
from skimage import transform as trans
from sklearn.metrics import roc_curve, auc

from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
from tqdm import tqdm
from face_lib.models import SphereNet20


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="IJBC verification protocol")
parser.add_argument(
    "--model-prefix",
    type=str,
    default="/gpfs/gpfs0/r.karimov/final_ijb/IJB/validation/arcface_torch/sphere20a_20171020.pth",
)

parser.add_argument(
    "--metadata-prefix",
    type=str,
    default="/gpfs/gpfs0/r.karimov/ijbc_11_metadata/",
)

parser.add_argument(
    "--loose-crop-prefix",
    type=str,
    default="/gpfs/gpfs0/r.karimov/ijbc_11_verification_data",
)

parser.add_argument(
    "--is-casia",
    type=str2bool,
    nargs="?",
    const=True,
    default=True,
)

parser.add_argument("--image-path", default="", type=str, help="")
parser.add_argument("--result-dir", default=".", type=str, help="")
parser.add_argument("--batch-size", default=128, type=int, help="")
parser.add_argument("--network", default="r50", type=str, help="")
parser.add_argument("--job", default="insightface", type=str, help="job name")
parser.add_argument(
    "--target", default="IJBC", type=str, help="target, set to IJBC or IJBB"
)
args = parser.parse_args()

target = args.target
model_path = args.model_prefix
image_path = args.image_path
result_dir = args.result_dir
gpu_id = None
use_norm_score = True
use_detector_score = True
use_flip_test = True
job = args.job
batch_size = args.batch_size


class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        self.image_size = (112, 96)
        src = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        if not args.is_case:
            src[:, 0] += 8.0
            self.image_size = (112, 112)

        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.net = SphereNet20()
        self.net.load_state_dict(torch.load(args.model_prefix))
        self.net.eval()
        self.net.feature = True

    def get(self, rimg, landmark):
        landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        try:
            img = cv2.warpAffine(rimg, M, (96, 112), borderValue=0.0)
        except:
            img = np.zeros((96, 112, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros(
            (2, 3, self.image_size[0], self.image_size[1]), dtype=np.uint8
        )
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        imgs = torch.Tensor(batch_data)
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.net(imgs)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()


def read_template_media_list(path):
    ijb_meta = pd.read_csv(path, sep=" ", header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=" ", header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


def read_image_feature(path):
    with open(path, "rb") as fid:
        img_feats = pickle.load(fid)
    return img_feats


def get_image_feature(img_path, files, model_path, epoch, gpu_id):
    batch_size = args.batch_size
    data_shape = (3, 112, 112)

    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0

    batch_data = np.empty((2 * batch_size, 3, 112, 96))
    embedding = Embedding(model_path, data_shape, batch_size)
    for img_index, each_line in tqdm(enumerate(files[: len(files) - rare_size])):
        name_lmk_score = each_line.strip().split(" ")
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread("/gpfs/gpfs0/r.karimov/final_ijb/IJB/edit" + img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)

        batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
        batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
        if (img_index + 1) % batch_size == 0:
            emb = embedding.forward_db(batch_data)
            img_feats[batch * batch_size : batch * batch_size + batch_size][:] = emb
            batch += 1
        faceness_scores.append(name_lmk_score[-1])

    batch_data = np.empty((2 * rare_size, 3, 112, 96))
    embedding = Embedding(model_path, data_shape, rare_size)
    for img_index_, each_line in tqdm(enumerate(files[len(files) - rare_size :])):
        name_lmk_score = each_line.strip().split(" ")
        img_name = os.path.join(img_path, name_lmk_score[0])

        img = cv2.imread(os.path.join(args.loose_crop_prefix, img_name))
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))

        input_blob = embedding.get(img, lmk)
        batch_data[2 * img_index_][:] = input_blob[0]
        batch_data[2 * img_index_ + 1][:] = input_blob[1]
        if (img_index_ + 1) % rare_size == 0:
            emb = embedding.forward_db(batch_data)
            img_feats[len(files) -
                      rare_size:][:] = emb
            batch += 1
        faceness_scores.append(name_lmk_score[-1])
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    return img_feats, faceness_scores


def image2template_feature(img_feats=None, templates=None, medias=None):
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print("Finish Calculating {} template features.".format(count_template))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i : i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print("Finish {}/{} pairs.".format(c, total_sublists))
    return score


def read_score(path):
    with open(path, "rb") as fid:
        img_feats = pickle.load(fid)
    return img_feats


templates, medias = read_template_media_list(
    os.path.join(args.metadata_prefix, "ijbc_face_tid_mid.txt")
)

pairs_path = os.path.join(args.metadata_prefix, "ijbc_template_pair_label.txt")
p1, p2, label = read_template_pair_list(pairs_path)

img_path = "%s/loose_crop" % image_path
img_list_path = os.path.join(args.metadata_prefix, "ijbc_name_5pts_score.txt")
img_list = open(img_list_path)
files = img_list.readlines()
files_list = files


img_feats, faceness_scores = get_image_feature(
    img_path, files_list, model_path, 0, gpu_id
)

if use_flip_test:
    img_input_feats = (
        img_feats[:, 0 : img_feats.shape[1] // 2]
        + img_feats[:, img_feats.shape[1] // 2 :]
    )
else:
    img_input_feats = img_feats[:, 0 : img_feats.shape[1] // 2]

if use_norm_score:
    img_input_feats = img_input_feats
else:
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats ** 2, -1, keepdims=True)
    )

if use_detector_score:
    print(img_input_feats.shape, faceness_scores.shape)
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(
    img_input_feats, templates, medias
)


score = verification(template_norm_feats, unique_templates, p1, p2)
save_path = os.path.join(result_dir, args.job)

if not os.path.exists(save_path):
    os.makedirs(save_path)

score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
np.save(score_save_file, score)


files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(zip(methods, sample_colours_from_colourmap(methods.shape[0], "Set2")))
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
tpr_fpr_table = PrettyTable(["Methods"] + [str(x) for x in x_labels])
fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(
        fpr,
        tpr,
        color=colours[method],
        lw=1,
        label=("[%s (AUC = %0.4f %%)]" % (method.split("-")[-1], roc_auc * 100)),
    )
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append("%.2f" % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle="--", linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale("log")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC on IJB")
plt.legend(loc="lower right")
fig.savefig(os.path.join(save_path, "%s.pdf" % target.lower()))
print(tpr_fpr_table)
