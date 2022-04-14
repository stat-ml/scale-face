import os
import numpy as np

from collections import namedtuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm

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
        self.mu = None
        self.sigma_sq = None


def to_short_name(subject_id, img):
    """
    the path from dataset root to image
    """
    name = img.replace('/', '_')
    return f"{subject_id}/{name}"


def build_templates(csv_path, feature_dict, uncertainty_dict):
    """
    Generate template objects by csv list like enroll_templates or verif_templates
    feature_dict and uncertainty_dict - the dictionary of precomputed embeddings/ue
    keys are
    """
    df = pd.read_csv(csv_path)
    templates = {}
    key_set = set(feature_dict.keys())

    def build_template(group):
        template_id = group.iloc[0]['TEMPLATE_ID']
        subject_id = group.iloc[0]['SUBJECT_ID']
        image_paths = [to_short_name(subject_id, n) for n in list(group.FILENAME)]
        present = sorted(list(key_set.intersection(set(image_paths))))
        if len(present) > 0:
            features = np.array([feature_dict[img] for img in present])
            sigmas = np.array([uncertainty_dict[img] for img in present])
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

    def all_templates(self):
        return self.templates_dict.values()

    def enroll_templates(self):
        return self._enroll_templates

    def verification_templates(self):
        return self._verification_templates

    def init_proto(self, proto_folder):
        """
        generates the list of verification templates
        and also a list of pairs
        """
        self.proto_folder = Path(proto_folder)
        enroll_path = self.proto_folder / 'enroll_templates.csv'
        verif_path = self.proto_folder / 'verif_templates.csv'
        self._pairs = pd.read_csv(self.proto_folder / 'cropped_matches.csv', header=None).to_numpy()
        # self._pairs = pd.read_csv(self.proto_folder / 'short_matches.csv', header=None).to_numpy()

        enroll_dict = build_templates(enroll_path, self.feature_dict, self.uncertainty_dict)
        self.templates_dict.update(enroll_dict)
        self._enroll_templates = enroll_dict.values()

        verify_dict = build_templates(verif_path, self.feature_dict, self.uncertainty_dict)
        self.templates_dict.update(verify_dict)
        self._verification_templates = verify_dict.values()

        self._pairs = self._clean_pairs(self._pairs, self.templates_dict)

    def _clean_pairs(self, pairs, templates_dict):
        print('len before', len(pairs))
        pairs = np.array([p for p in pairs if (p[0] in self.templates_dict and p[1] in templates_dict)])
        print('len after', len(pairs))
        return pairs

    def get_features_uncertainties_labels(self):
        """
        returns features, uncertainties and labels for pairs
        if verify_only_ue flag set, ignores the uncertainty from enroll templates
        """
        features1 = np.array([
            self.templates_dict[t].mu for t in tqdm(self._pairs[:, 0], desc='Features')
        ])
        features2 = np.array([
            self.templates_dict[t].mu for t in tqdm(self._pairs[:, 1])
        ])
        sigmas_sq1 = np.array([
            self.templates_dict[t].sigma_sq for t in tqdm(self._pairs[:, 0], desc='Sigmas')
        ])
        sigmas_sq2 = np.array([
            self.templates_dict[t].sigma_sq for t in self._pairs[:, 1]
        ])
        labels1 = np.array([
            self.templates_dict[t].subject_id for t in tqdm(self._pairs[:, 0], desc='Labels')
        ])
        labels2 = np.array([
            self.templates_dict[t].subject_id for t in self._pairs[:, 1]
        ])
        label_vec = (labels1 == labels2)

        return features1, features2, sigmas_sq1, sigmas_sq2, label_vec
