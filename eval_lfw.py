import os
import torch
from torch.autograd import Variable

torch.backends.cudnn.bencmark = True
import numpy as np
from utils import dataset


def KFold(n=10, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i * n // n_folds : (i + 1) * n // n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


lfw_prefix = "/gpfs/gpfs0/r.karimov/lfw/data_"

predicts = []
import model

net = model.SphereNet20(feature=True)
#net.load_state_dict(torch.load("checkpoints/sphere20a_20171020.pth"))
net.cuda()
net.eval()

from utils.imageprocessing import preprocess

proc_func = lambda images: preprocess(images, [112, 96], is_training=False)
lfw_set = dataset.Dataset("/gpfs/gpfs0/r.karimov/lfw/data_", preprocess_func=proc_func)

with open("/gpfs/gpfs0/r.karimov/lfw/pairs_val_6000.txt") as f:
    pairs_lines = f.readlines()[1:]

from tqdm import tqdm
cx = 0
for i in tqdm(range(6000)):
    p = pairs_lines[i].replace("\n", "").split("\t")

    if 3 == len(p):
        sameflag = 1
        name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
        name2 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[2]))
    if 4 == len(p):
        sameflag = 0
        name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
        name2 = p[2] + "/" + p[2] + "_" + "{:04}.jpg".format(int(p[3]))

    try:
        img1 = lfw_set.get_item_by_the_path(name1)
        img2 = lfw_set.get_item_by_the_path(name2)
    except:
        # FIXME: mtcncaffe and spherenet alignments are not the same
        continue

    img_batch = torch.from_numpy(np.concatenate((img1[None], img1[None]), axis=0)).permute(0, 3, 1, 2).cuda()
    output = net(img_batch)
    f1, f2 = output
    cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
    predicts.append("{}\t{}\t{}\t{}\n".format(name1, name2, cosdistance, sameflag))


accuracy = []
thd = []
folds = KFold(n=10, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)

predicts = np.array(list(map(lambda line: line.strip("\n").split(), predicts)))
for idx, (train, test) in enumerate(folds):
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print(
    "LFWACC={:.4f} std={:.4f} thd={:.4f}".format(
        np.mean(accuracy), np.std(accuracy), np.mean(thd)
    )
)