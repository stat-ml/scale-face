import numpy as np
import pandas as pd


def parse_pairs(data_dir):
    pairs_files = data_dir / 'pairs_CPLFW.txt'

    with open(pairs_files, 'r') as f:
        lines = f.readlines()
    lines = [l.split() for l in lines]
    print(*(lines[:10] + lines[-10:]), sep='\n')

    pairs = []
    for i in range(len(lines) // 2):
        pair = lines[2*i][0], lines[2*i+1][0], int(lines[2*i][1])
        pairs.append(pair)

    as_dict = {
        'photo_1': [p[0] for p in pairs],
        'photo_2': [p[1] for p in pairs],
        'label': [p[2] for p in pairs]
    }
    df = pd.DataFrame(as_dict)
    df.to_csv(data_dir / 'pairs.csv', index=False)


def generate_val_test_split(dataframe, cplfw_dir):
    lst = np.unique(dataframe.photo_1.to_list() + dataframe.photo_2.to_list())

    def cut_name(name):
        return '_'.join(name.split('_')[:-1])

    names = np.sort(np.unique([cut_name(name) for name in lst]))
    np.random.seed(42)
    np.random.shuffle(names)

    test_identities = names[:2000]
    val_identities = names[2000:]
    print(test_identities)

    def suitable(row, identities):
        return cut_name(row.photo_1) in identities and cut_name(row.photo_2) in identities

    test_df = dataframe[dataframe.apply(lambda row: suitable(row, test_identities), axis=1)]
    val_df = dataframe[dataframe.apply(lambda row: suitable(row, val_identities), axis=1)]

    test_df.to_csv(cplfw_dir / 'pairs_test.csv', index=False)
    val_df.to_csv(cplfw_dir / 'pairs_val.csv', index=False)


