from pathlib import Path
import pandas as pd


FILE = '/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_10000_prob_0.1.csv'
PROTOCOL_DIR = Path('/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/test1/')
FILE_2 = PROTOCOL_DIR / 'short_matches.csv'
FILE_3 = PROTOCOL_DIR / 'enroll_templates.csv'
FILE_4 = PROTOCOL_DIR / 'verif_templates.csv'

# what do i need to do?
# make a fake enroll/verif templates files
#


def main():
    df = pd.read_csv(FILE, names=['enroll_img', 'verif_img', 'label'])
    print(df)
    res = sorted(df.enroll_img.unique().tolist() + df.verif_img.unique().tolist())
    # print(res)
    print(len(res))

    template_ids = {}
    for i, img_path in enumerate(res):
        template_ids[img_path] = i+1

    def parse(img_path):
        subject, path = img_path.split("/")
        path = path.replace('_', '/')
        return subject, path

    enroll_df = pd.DataFrame({
        'TEMPLATE_ID': [0], 'SUBJECT_ID': [0], 'FILENAME': ['a']
    })

    for img in df.enroll_img.unique():
        template = template_ids[img]
        subject, path = parse(img)
        new_row = pd.DataFrame({
            'TEMPLATE_ID': [template], 'SUBJECT_ID': [subject], 'FILENAME': [path]
        })
        enroll_df = pd.concat((enroll_df, new_row))
    enroll_df = enroll_df.iloc[1:]
    enroll_df.to_csv(PROTOCOL_DIR / 'enroll_templates2.csv', index=False)
    print(enroll_df)
    enroll_df = pd.read_csv(PROTOCOL_DIR / 'enroll_templates2.csv')
    print(enroll_df)


    verif_df = pd.DataFrame({
        'TEMPLATE_ID': [0], 'SUBJECT_ID': [0], 'FILENAME': ['a']
    })

    for img in df.verif_img.unique():
        template = template_ids[img]
        subject, path = parse(img)
        new_row = pd.DataFrame({
            'TEMPLATE_ID': [template], 'SUBJECT_ID': [subject], 'FILENAME': [path]
        })
        verif_df = pd.concat((verif_df, new_row))
    verif_df = verif_df.iloc[1:]
    verif_df.to_csv(PROTOCOL_DIR / 'verif_templates2.csv', index=False)
    print(verif_df)
    verif_df = pd.read_csv(PROTOCOL_DIR / 'verif_templates2.csv')
    print(verif_df)

    df['enroll_template'] = df.enroll_img.map(template_ids)
    df['verif_template'] = df.verif_img.map(template_ids)
    df.drop(['enroll_img', 'verif_img', 'label'], axis=1, inplace=True)
    df.to_csv(PROTOCOL_DIR / 'pair_matches.csv', index=False, header=None)


    # Sanity check
    enroll_df = pd.read_csv(PROTOCOL_DIR / 'enroll_templates2.csv')
    verif_df = pd.read_csv(PROTOCOL_DIR / 'verif_templates2.csv')
    templates = pd.concat((enroll_df, verif_df))
    temp_dir = {}
    def to_img(row):
        path = str(row.SUBJECT_ID) + '/' + row.FILENAME.replace('/', '_')
        temp_dir[row.TEMPLATE_ID] = path
    templates.apply(to_img, axis=1)

    df5 = pd.read_csv(PROTOCOL_DIR / 'pair_matches.csv', names=['temp1', 'temp2'])
    df5['path1'] = df5['temp1'].map(temp_dir)
    df5['path2'] = df5['temp2'].map(temp_dir)
    print(df5)


if __name__ == '__main__':
    main()