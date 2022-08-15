from .casia import CASIAWebFace
from .cats import CatsDataset
from .ms1m import MXFaceDataset, MXFaceDatasetDistorted, MXFaceDatasetGauss, SyntheticDataset
from .ms1m_pfe import MS1MDatasetPFE, ms1m_collate_fn
from .pairs_datasets import MS1MDatasetRandomPairs
from .dataloader import DataLoaderX
from .ijb import IJBDataset
from .ijba import IJBATest
from .ijbc import IJBCTest
from .ijbc_templates import IJBCTemplates
from .stanford_products import ProductsDataset