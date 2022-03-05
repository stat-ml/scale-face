"""
Build the cleaned_templates.csv for build_the_list.py
Get rid of all records that misses the images after face detection and alignment
Source templates are taken from IJB-C test1 enroll and verify lists
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np


image_folder = Path("/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big")
protocol_folder = Path('/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/test1')


def image_exists(row):
    name = row.FILENAME.replace('/', '_')
    path = image_folder / str(row.SUBJECT_ID) / name
    return os.path.exists(path)


def main():
    df_0 = pd.read_csv(protocol_folder / 'enroll_templates.csv')
    df_1 = pd.read_csv(protocol_folder / 'verif_templates.csv')
    df = pd.concat((df_0, df_1))
    existing = df.apply(image_exists, axis=1)
    df_cleaned = df[existing]
    df_cleaned.to_csv(protocol_folder / 'cleaned_templates.csv')
    print(df.shape)
    print(df_cleaned.shape)


if __name__ == '__main__':
    main()
