from pathlib import Path
import pandas as pd
from tqdm import tqdm

# DIRECTORY = Path('/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/pseudo_archive')
DIRECTORY = Path("/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/test1")


# 1000 false
# 100 true
POS_NUM = 19000
NEG_NUM = 1000000

def main():
    cleaned = "cleaned_templates.csv"
    template_df = pd.read_csv(DIRECTORY/cleaned)
    matches = pd.read_csv(DIRECTORY / 'match.csv')
    subject_by_template = template_df[["TEMPLATE_ID", "SUBJECT_ID"]].groupby("TEMPLATE_ID").min()

    pos_counter = 0
    neg_counter = 0

    idx = []

    for index, row in tqdm(matches.iterrows(), total=len(matches)):
        try:
            subject_1 = subject_by_template.loc[row['ENROLL_TEMPLATE_ID'], 'SUBJECT_ID']
            subject_2 = subject_by_template.loc[row['VERIF_TEMPLATE_ID'], "SUBJECT_ID"]
        except:
            continue

        if subject_1 == subject_2:
            if pos_counter < POS_NUM:
                pos_counter += 1
                idx.append(row.name)
        else:
            if neg_counter < NEG_NUM:
                neg_counter += 1
                idx.append(row.name)

        if pos_counter == POS_NUM and neg_counter == NEG_NUM:
            break

    print(matches.loc[idx])
    matches.loc[idx].to_csv(DIRECTORY / 'cropped_matches.csv', index=False, header=False)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()