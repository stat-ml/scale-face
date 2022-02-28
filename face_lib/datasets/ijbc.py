import os
import numpy as np

from collections import namedtuple
from pathlib import Path
import pandas as pd

import face_lib.utils.metrics as metrics

VerificationFold = namedtuple(
    "VerificationFold",
    ["train_indices", "test_indices", "train_templates", "templates1", "templates2"],
)


class Template:
    """
    keeps the data about single template
    """
    def __init__(self, template_id, subject_id, image_paths, features, sigmas):
        self.template_id = template_id
        self.subject_id = subject_id
        self.image_paths = image_paths
        self.features = features
        self.sigmas = sigmas
        self.feature = None
        self.sigma_sq = None

# class Template:
#     def __init__(self, template_id, label, indices, medias):
#         self.template_id = template_id
#         self.label = label
#         self.indices = np.array(indices)
#         self.medias = np.array(medias)


# def build_subject_dict(image_list):
#     subject_dict = {}
#     for i, line in enumerate(image_list):
#         subject_id, image = tuple(line.split("/")[-2:])
#         if subject_id == "NaN":
#             continue
#         subject_id = int(subject_id)
#         image, _ = os.path.splitext(image)
#         image = image.replace("_", "/", 1)  # Recover filenames
#         if not subject_id in subject_dict:
#             subject_dict[subject_id] = {}
#         subject_dict[subject_id][image] = i
#     return subject_dict


# def build_templates(subject_dict, meta_file):
#     with open(meta_file, "r") as f:
#         meta_list = f.readlines()
#         meta_list = [x.split("\n")[0] for x in meta_list]
#         meta_list = meta_list[1:]
#
#     templates = []
#     template_id = None
#     template_label = None
#     template_indices = None
#     template_medias = None
#     count = 0
#     for line in meta_list:
#         temp_id, subject_id, image, media = tuple(line.split(",")[0:4])
#         temp_id = int(temp_id)
#         subject_id = int(subject_id)
#         image, _ = os.path.splitext(image)
#         if subject_id in subject_dict and image in subject_dict[subject_id]:
#             index = subject_dict[subject_id][image]
#             count += 1
#         else:
#             index = None
#
#         if temp_id != template_id:
#             if template_id is not None:
#                 templates.append(
#                     Template(
#                         template_id, template_label, template_indices, template_medias
#                     )
#                 )
#             template_id = temp_id
#             template_label = subject_id
#             template_indices = []
#             template_medias = []
#
#         if index is not None:
#             template_indices.append(index)
#             template_medias.append(media)
#
#     # last template
#     templates.append(
#         Template(template_id, template_label, template_indices, template_medias)
#     )
#     return templates


# def read_pairs(pair_file):
#     with open(pair_file, "r") as f:
#         pairs = f.readlines()
#         pairs = [x.split("\n")[0] for x in pairs]
#         pairs = [pair.split(",") for pair in pairs]
#         pairs = [(int(pair[0]), int(pair[1])) for pair in pairs]
#     return pairs



class IJBCTest:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.subject_dict = build_subject_dict(image_paths)
        self.verification_folds = None
        self.verification_templates = None
        self.verification_G1_templates = None
        self.verification_G2_templates = None

        print("Number of identities : ", len(self.subject_dict))

    def init_verification_proto(self, protofolder):
        self.verification_folds = []
        self.verification_templates = []

        meta_gallery1 = os.path.join(protofolder, "ijbc_1N_gallery_G1.csv")
        meta_gallery2 = os.path.join(protofolder, "ijbc_1N_gallery_G2.csv")
        meta_probe = os.path.join(protofolder, "ijbc_1N_probe_mixed.csv")
        pair_file = os.path.join(protofolder, "ijbc_11_G1_G2_matches.csv")

        gallery_templates = build_templates(self.subject_dict, meta_gallery1)
        gallery_templates.extend(build_templates(self.subject_dict, meta_gallery2))
        gallery_templates.extend(build_templates(self.subject_dict, meta_probe))

        # Build pairs
        template_dict = {}
        for t in gallery_templates:
            template_dict[t.template_id] = t
        pairs = read_pairs(pair_file)
        self.verification_G1_templates = []
        self.verification_G2_templates = []
        for p in pairs:
            self.verification_G1_templates.append(template_dict[p[0]])
            self.verification_G2_templates.append(template_dict[p[1]])

        self.verification_G1_templates = np.array(
            self.verification_G1_templates, dtype=np.object
        )
        self.verification_G2_templates = np.array(
            self.verification_G2_templates, dtype=np.object
        )

        self.verification_templates = np.concatenate(
            [self.verification_G1_templates, self.verification_G2_templates]
        )
        print("{} templates are initialized.".format(len(self.verification_templates)))

    def init_proto(self, protofolder):
        self.init_verification_proto(protofolder)

    def test_verification(self, compare_func, FARs=None, verbose=True):

        FARs = [1e-5, 1e-4, 1e-3, 1e-2] if FARs is None else FARs

        # templates1 = self.verification_G1_templates
        # templates2 = self.verification_G2_templates
        #
        # not_nan_1 = np.array([template.feature is not None for template in templates1])
        # not_nan_2 = np.array([template.feature is not None for template in templates2])
        # not_nan = not_nan_1 & not_nan_2
        #
        # print(
        #     f"Ignored {not_nan.shape[0] - not_nan.sum()} / {not_nan.shape[0]} # bad templates"
        # )
        # templates1 = templates1[not_nan]
        # templates2 = templates2[not_nan]
        #
        # features1 = [t.feature for t in templates1]
        # features2 = [t.feature for t in templates2]
        # sigmas_sq1 = [t.sigma_sq for t in templates1]
        # sigmas_sq2 = [t.sigma_sq for t in templates2]
        # labels1 = np.array([t.label for t in templates1])
        # labels2 = np.array([t.label for t in templates2])
        #
        # label_vec = labels1 == labels2

        features1, features2, sigmas_sq1, sigmas_sq2, label_vec = \
            self.get_features_uncertainties_labels(verbose=verbose)

        score_vec = compare_func(features1, features2, sigmas_sq1, sigmas_sq2)

        if verbose:
            print(f"Positive labels : {sum(label_vec)} / {len(label_vec)}")

        tars, fars, thresholds = metrics.ROC(score_vec, label_vec, FARs=FARs)

        # There is no std for IJB-C
        std = [0.0 for t in tars]

        return tars, std, fars

    def get_features_uncertainties_labels(self, verbose=False):
        templates1 = self.verification_G1_templates
        templates2 = self.verification_G2_templates

        not_nan_1 = np.array([template.feature is not None for template in templates1])
        not_nan_2 = np.array([template.feature is not None for template in templates2])
        not_nan = not_nan_1 & not_nan_2

        if verbose:
            print(f"Ignored {not_nan.shape[0] - not_nan.sum()} / {not_nan.shape[0]} # bad templates")

        templates1 = templates1[not_nan]
        templates2 = templates2[not_nan]

        features1 = [t.feature for t in templates1]
        features2 = [t.feature for t in templates2]
        sigmas_sq1 = [t.sigma_sq for t in templates1]
        sigmas_sq2 = [t.sigma_sq for t in templates2]
        labels1 = np.array([t.label for t in templates1])
        labels2 = np.array([t.label for t in templates2])

        label_vec = labels1 == labels2

        return features1, features2, sigmas_sq1, sigmas_sq2, label_vec



#verification templates list. What is this exactly?
#features/uncertainties/labels list. But we already had it, no?
def to_short_name(subject_id, img):
    name = img.replace('/', '_')
    return f"{subject_id}/{name}"


def build_templates(csv_path, feature_dict, uncertainty_dict):
    df = pd.read_csv(csv_path)
    templates = {}
    key_set = set(feature_dict.keys())

    def build_template(group):
        template_id = group.iloc[0]['TEMPLATE_ID']
        subject_id = group.iloc[0]['SUBJECT_ID']
        image_paths = [to_short_name(subject_id, n) for n in list(group.FILENAME)]
        present = list(key_set.intersection(set(image_paths)))
        if len(present) > 0:
            features = [feature_dict[img] for img in present]
            sigmas = [uncertainty_dict[img][0] for img in present]
            image_paths = present
            template = Template(template_id, subject_id, image_paths, features, sigmas)
            templates[template_id] = template

    df.groupby('TEMPLATE_ID').apply(build_template)

    return templates


class IJBCTemplates:
    def __init__(self, image_paths, feature_dict, uncertainty_dict):
        """
        image_paths - short path with subject_id and image_name
        """
        self.templates_dict = {}
        self.image_paths = image_paths
        self.feature_dict = feature_dict
        self.uncertainty_dict = uncertainty_dict

    def verification_templates(self):
        return self.templates_dict.values()

    def init_proto(self, proto_folder):
        """
        generates the list of verification templates
        and also a list of pairs
        """
        self.proto_folder = Path(proto_folder)
        enroll_path = self.proto_folder / 'enroll_templates.csv'
        verif_path = self.proto_folder / 'verif_templates.csv'
        self.templates_dict.update(
            build_templates(enroll_path, self.feature_dict, self.uncertainty_dict)
        )
        self.templates_dict.update(
            build_templates(verif_path, self.feature_dict, self.uncertainty_dict)
        )
        import ipdb; ipdb.set_trace()
        # enroll_templates = self.proto_folder

    def get_features_uncertainties_labels(self, verbose=False):
        features1, features2, sigmas_sq1, sigmas_sq2, label_vec = [], [], [], [], []
        return features1, features2, sigmas_sq1, sigmas_sq2, label_vec


# class IJBCTemplates:
#     def __init__(self, image_paths):
#         self.image_paths = image_paths
#         self.subject_dict = build_subject_dict(image_paths)
#         self.verification_folds = None
#         self.verification_templates = None
#         self.verification_G1_templates = None
#         self.verification_G2_templates = None
#
#         print("Number of identities : ", len(self.subject_dict))
#
#     def init_proto(self, protofolder):
#         self.verification_folds = []
#         self.verification_templates = []
#
#         meta_gallery1 = os.path.join(protofolder, "ijbc_1N_gallery_G1.csv")
#         meta_gallery2 = os.path.join(protofolder, "ijbc_1N_gallery_G2.csv")
#         meta_probe = os.path.join(protofolder, "ijbc_1N_probe_mixed.csv")
#         pair_file = os.path.join(protofolder, "ijbc_11_G1_G2_matches.csv")
#
#         gallery_templates = build_templates(self.subject_dict, meta_gallery1)
#         gallery_templates.extend(build_templates(self.subject_dict, meta_gallery2))
#         gallery_templates.extend(build_templates(self.subject_dict, meta_probe))
#
#         # Build pairs
#         template_dict = {}
#         for t in gallery_templates:
#             template_dict[t.template_id] = t
#         pairs = read_pairs(pair_file)
#         self.verification_G1_templates = []
#         self.verification_G2_templates = []
#         for p in pairs:
#             self.verification_G1_templates.append(template_dict[p[0]])
#             self.verification_G2_templates.append(template_dict[p[1]])
#
#         import ipdb; ipdb.set_trace()
#
#         self.verification_G1_templates = np.array(
#             self.verification_G1_templates, dtype=np.object
#         )
#         self.verification_G2_templates = np.array(
#             self.verification_G2_templates, dtype=np.object
#         )
#
#         self.verification_templates = np.concatenate(
#             [self.verification_G1_templates, self.verification_G2_templates]
#         )
#         print("{} templates are initialized.".format(len(self.verification_templates)))
#
#     def get_features_uncertainties_labels(self, verbose=False):
#         templates1 = self.verification_G1_templates
#         templates2 = self.verification_G2_templates
#
#         not_nan_1 = np.array([template.feature is not None for template in templates1])
#         not_nan_2 = np.array([template.feature is not None for template in templates2])
#         not_nan = not_nan_1 & not_nan_2
#
#         if verbose:
#             print(f"Ignored {not_nan.shape[0] - not_nan.sum()} / {not_nan.shape[0]} # bad templates")
#
#         templates1 = templates1[not_nan]
#         templates2 = templates2[not_nan]
#
#         features1 = [t.feature for t in templates1]
#         features2 = [t.feature for t in templates2]
#         sigmas_sq1 = [t.sigma_sq for t in templates1]
#         sigmas_sq2 = [t.sigma_sq for t in templates2]
#         labels1 = np.array([t.label for t in templates1])
#         labels2 = np.array([t.label for t in templates2])
#
#         label_vec = labels1 == labels2
#
#         return features1, features2, sigmas_sq1, sigmas_sq2, label_vec