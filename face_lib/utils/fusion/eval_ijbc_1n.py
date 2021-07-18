import os
import numpy as np
import timeit
import sklearn
import cv2
import sys
import argparse
import glob
import numpy.matlib
import heapq
import math
from datetime import datetime as dt
import torch
from skimage import transform as trans
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# import mxnet as mx

# path = str(Path(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute()).parent.absolute())
# print(path)
# sys.path.insert(0, path)

from sklearn import preprocessing
sys.path.append('./recognition')
#from embedding import Embedding
# from menpo.visualize import print_progress
# from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap

import net_sphere
from backbones import get_model
from tqdm import tqdm
from model import iresnet50
# from face_lib import models


class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        # image_size = (112, 96)
        image_size = (112, 112)
        self.image_size = image_size
        resnet = get_model(args.network, dropout=0, fp16=False)
        #weight = torch.load(prefix)
        #resnet.load_state_dict(weight)
        model = torch.nn.DataParallel(resnet)
        self.model = model
        self.model.eval()


        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        #src[:, 0] += 8.0

        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

        # self.net = getattr(net_sphself.netere, "sphere20a")()
        # load_path = os.path.join(args.meta_path, "sphere20a_20171020.pth")
        # self.net.load_state_dict(torch.load(load_path))
        # self.net.cuda()
        # self.net.eval()
        # self.net.feature = True

        self.net = iresnet50()
        checkpoint = torch.load(args.model_prefix)
        self.net.load_state_dict(checkpoint["backbone"])
        self.net.cuda()
        self.net.eval()


    def get(self, rimg, landmark):

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        try:
            # img = cv2.warpAffine(rimg, M, (96, 112), borderValue=0.0)
            img = cv2.warpAffine(rimg, M, (112, 112), borderValue=0.0)
        except:
            # print("bad bad")
            # img = np.zeros((96, 112, 3), dtype=np.float32)
            img = np.zeros((112, 112, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[0], self.image_size[1]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        # print("Imgs : ", imgs.shape)
        feat, _ = self.net(imgs)
        # print("Feat : ", feat.shape)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        # print("Feat : ", feat.shape)
        return feat.cpu().numpy()


def read_template_subject_id_list(path):
    ijb_meta = np.loadtxt(path, dtype=str, skiprows=1, delimiter=',')
    templates = ijb_meta[:, 0].astype(np.int)
    subject_ids = ijb_meta[:, 1].astype(np.int)
    return templates, subject_ids


def read_template_media_list(path):
    ijb_meta = np.loadtxt(path, dtype=str)
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    pairs = np.loadtxt(path, dtype=str)
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


#def get_image_feature(feature_path, faceness_path):
#    img_feats = np.loadtxt(feature_path)
#    faceness_scores = np.loadtxt(faceness_path)
#    return img_feats, faceness_scores


def get_image_feature(img_path, files_list, model_path, epoch, gpu_id):
    batch_size = args.batch_size
    data_shape = (3, 112, 112)

    files = files_list
    print('files:', len(files))
    # files = files[:1000] # This is solution for debug. Comment this line pls


    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0
    img_feats = np.empty((len(files), 1024), dtype=np.float32)

    import os
    path = "/gpfs/gpfs0/r.karimov/final_ijb/IJB/edit/loose_crop"
    fin_out_path = os.path.join(args.meta_path, "final_out")
    fp = np.memmap(fin_out_path, dtype='float32', mode='r', shape=(len(os.listdir(path)), 11))
    #fp2 = np.memmap("final_out2", dtype='float32', mode='r', shape=(len(files), 512))
    #img_feats = np.array(fp2)

    # batch_data = np.empty((2 * batch_size, 3, 112, 96))
    batch_data = np.empty((2 * batch_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, batch_size)
    for img_index, each_line in tqdm(enumerate(files[:len(files) - rare_size])):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread("/gpfs/gpfs0/r.karimov/final_ijb/IJB/edit" + img_name)
        lmk = fp[img_index][:10]
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)

        batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
        batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
        if (img_index + 1) % batch_size == 0:
            res = embedding.forward_db(batch_data)
            # print("Res : ", res.shape)
            img_feats[batch * batch_size:batch * batch_size +
                                         batch_size][:] = embedding.forward_db(batch_data)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])

    batch_data = np.empty((2 * rare_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, rare_size)
    for img_index_, each_line in tqdm(enumerate(files[len(files) - rare_size:])):
        img_index = img_index_ + len(files) - rare_size
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread("/gpfs/gpfs0/r.karimov/final_ijb/IJB/edit" + img_name)

        lmk = fp[img_index][:10]
        lmk = lmk.reshape((5, 2))

        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)
        batch_data[2 * img_index_][:] = input_blob[0]
        batch_data[2 * img_index_ + 1][:] = input_blob[1]
        if (img_index_ + 1) % rare_size == 0:
            img_feats[len(files) -
                      rare_size:][:] = embedding.forward_db(batch_data)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])
    # fp2[:] = img_feats
    # fp2.flush()
    # exit()
    #faceness_scores = np.array(faceness_scores).astype(np.float32)
    faceness_scores = np.ones((469375,), dtype=np.float32)
    # faceness_scores = np.ones((1000,), dtype=np.float32)
    return img_feats, faceness_scores


def image2template_feature(img_feats=None, templates=None, medias=None, choose_templates=None, choose_ids=None):
    if choose_templates is not None:  # 1N
        print("1N")
        unique_templates, indices = np.unique(choose_templates, return_index=True)
        unique_subjectids = choose_ids[indices]
    else:  # 11
        print("11")
        unique_templates = np.unique(templates)
        unique_subjectids = None

    # template_feats = np.zeros((len(unique_templates), img_feats.shape[1]), dtype=img_feats.dtype)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    print("Unique_templates : ", unique_templates.shape)
    print("Unique_templates : ", max(unique_templates))

    for count_template, uqt in tqdm(enumerate(unique_templates), "Extract template feature", total=len(unique_templates)):
        (ind_t,) = np.where(templates == uqt)
        # if max(ind_t) < 1000: # <------------------------------------remove
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    return template_norm_feats, unique_templates, unique_subjectids

def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = cPickle.load(fid)
    return img_feats


def evaluation(query_feats, gallery_feats, mask):
    Fars = [0.01, 0.1, 1, 10]
    print(query_feats.shape)
    print(gallery_feats.shape)

    query_num = query_feats.shape[0]
    gallery_num = gallery_feats.shape[0]

    similarity = np.dot(query_feats, gallery_feats.T)
    print('similarity shape', similarity.shape)
    top_inds = np.argsort(-similarity)
    print(top_inds.shape)

    # calculate top1
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0]
        if j == mask[i]:
            correct_num += 1
    print("top1 = {}".format(correct_num / query_num))
    # calculate top5
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:5]
        if mask[i] in j:
            correct_num += 1
    print("top5 = {}".format(correct_num / query_num))

    neg_pair_num = query_num * gallery_num - query_num
    print(neg_pair_num)
    required_topk = [math.ceil(query_num * x) for x in Fars]
    top_sims = similarity
    # calculate fars and tprs
    pos_sims = []
    for i in range(query_num):
        gt = mask[i]
        pos_sims.append(top_sims[i, gt])
        top_sims[i, gt] = -2.0

    pos_sims = np.array(pos_sims)
    print(pos_sims.shape)
    neg_sims = top_sims[np.where(top_sims > -2.0)]
    print("neg_sims num = {}".format(len(neg_sims)))
    neg_sims = heapq.nlargest(max(required_topk), neg_sims)  # heap sort
    print("after sorting , neg_sims num = {}".format(len(neg_sims)))
    for far, pos in zip(Fars, required_topk):
        th = neg_sims[pos - 1]
        recall = np.sum(pos_sims > th) / query_num
        print("far = {:.10f} pr = {:.10f} th = {:.10f}".format(
            far, recall, th))


def gen_mask(query_ids, reg_ids):
    mask = []
    for query_id in query_ids:
        pos = [i for i, x in enumerate(reg_ids) if query_id == x]
        if len(pos) != 1:
            raise RuntimeError(
                "RegIdsError with id = {}， duplicate = {} ".format(
                    query_id, len(pos)))
        mask.append(pos[0])
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='do ijb 1n test')
    # general
    parser.add_argument('--model_prefix',
                        default='',
                        help='path to load model.')
    parser.add_argument('--model_epoch', default=1, type=int, help='')
    parser.add_argument('--gpu', default=7, type=int, help='gpu id')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--network', default='r50', type=str, help='')
    parser.add_argument('--image_path', default='', type=str, help='')
    parser.add_argument('--meta_path', default='', type=str, help='')
    parser.add_argument('--job',
                        default='insightface',
                        type=str,
                        help='job name')
    parser.add_argument('--target',
                        default='IJBC',
                        type=str,
                        help='target, set to IJBC or IJBB')
    args = parser.parse_args()
    target = args.target
    model_path = args.model_prefix
    gpu_id = args.gpu
    epoch = args.model_epoch
    meta_dir = "/gpfs/gpfs0/r.karimov/trash/ijba/meta"
    if target == 'IJBC':
        gallery_s1_record = "%s_1N_gallery_G1.csv" % (args.target.lower())
        gallery_s2_record = "%s_1N_gallery_G2.csv" % (args.target.lower())
    else:
        gallery_s1_record = "%s_1N_gallery_S1.csv" % (args.target.lower())
        gallery_s2_record = "%s_1N_gallery_S2.csv" % (args.target.lower())
    gallery_s1_templates, gallery_s1_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s1_record))
    print(gallery_s1_templates.shape, gallery_s1_subject_ids.shape)

    gallery_s2_templates, gallery_s2_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s2_record))
    print(gallery_s2_templates.shape, gallery_s2_templates.shape)

    gallery_templates = np.concatenate(
        [gallery_s1_templates, gallery_s2_templates])
    gallery_subject_ids = np.concatenate(
        [gallery_s1_subject_ids, gallery_s2_subject_ids])
    print(gallery_templates.shape, gallery_subject_ids.shape)

    media_record = "%s_face_tid_mid.txt" % args.target.lower()
    total_templates, total_medias = read_template_media_list(
        os.path.join(meta_dir, media_record))
    print("total_templates", total_templates.shape, total_medias.shape)
    #load image features
    start = timeit.default_timer()
    feature_path = ''  #feature path
    face_path = ''  #face path
    image_path = args.image_path
    img_path = '%s/loose_crop' % image_path
    img_list_path = os.path.join(args.meta_path, "ijbc_name_5pts_score.txt")

    img_list = open(img_list_path)
    files = img_list.readlines()
# files_list = divideIntoNstrand(files, rank_size)
    files_list = files

    img_feats, faceness_scores = get_image_feature(img_path, files_list,
                                                   model_path, epoch, gpu_id)
    print('img_feats', img_feats.shape)
    print('faceness_scores', faceness_scores.shape)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                              img_feats.shape[1]))

    # compute template features from image features.
    start = timeit.default_timer()
    # ==========================================================
    # Norm feature before aggregation into template feature?
    # Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
    # ==========================================================
    use_norm_score = True  # if True, TestMode(N1)
    use_detector_score = True  # if True, TestMode(D1)
    use_flip_test = True  # if True, TestMode(F1)

    if use_flip_test:
        # concat --- F1
        #img_input_feats = img_feats
        # add --- F2
        img_input_feats = img_feats[:, 0:int(
            img_feats.shape[1] / 2)] + img_feats[:,
                                                 int(img_feats.shape[1] / 2):]
    else:
        img_input_feats = img_feats[:, 0:int(img_feats.shape[1] / 2)]

    if use_norm_score:
        img_input_feats = img_input_feats
    else:
        # normalise features to remove norm information
        img_input_feats = img_input_feats / np.sqrt(
            np.sum(img_input_feats**2, -1, keepdims=True))

    if use_detector_score:
        img_input_feats = img_input_feats * np.matlib.repmat(
            faceness_scores[:, np.newaxis], 1, img_input_feats.shape[1])
    else:
        img_input_feats = img_input_feats
    print("input features shape", img_input_feats.shape)

    #load gallery feature
    gallery_templates_feature, gallery_unique_templates, gallery_unique_subject_ids = image2template_feature(
        img_input_feats, total_templates, total_medias, gallery_templates,
        gallery_subject_ids)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print("gallery_templates_feature", gallery_templates_feature.shape)
    print("gallery_unique_subject_ids", gallery_unique_subject_ids.shape)
    #np.savetxt("gallery_templates_feature.txt", gallery_templates_feature)
    #np.savetxt("gallery_unique_subject_ids.txt", gallery_unique_subject_ids)

    #load prope feature
    probe_mixed_record = "%s_1N_probe_mixed.csv" % target.lower()
    probe_mixed_templates, probe_mixed_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, probe_mixed_record))
    print(probe_mixed_templates.shape, probe_mixed_subject_ids.shape)
    probe_mixed_templates_feature, probe_mixed_unique_templates, probe_mixed_unique_subject_ids = image2template_feature(
        img_input_feats, total_templates, total_medias, probe_mixed_templates,
        probe_mixed_subject_ids)
    print("probe_mixed_templates_feature", probe_mixed_templates_feature.shape)
    print("probe_mixed_unique_subject_ids",
          probe_mixed_unique_subject_ids.shape)

    gallery_ids = gallery_unique_subject_ids
    gallery_feats = gallery_templates_feature
    probe_ids = probe_mixed_unique_subject_ids
    probe_feats = probe_mixed_templates_feature

    mask = gen_mask(probe_ids, gallery_ids)

    print("{}: start evaluation".format(dt.now()))
    evaluation(probe_feats, gallery_feats, mask)
    print("{}: end evaluation".format(dt.now()))