from pathlib import Path
import pandas as pd

DIRECTORY = Path("/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/test1")


def main():
    pass
    def file_exists():
        pass
    enroll_df = pd.read_csv()
    # cleaned = "cleaned_templates.csv"
    # template_df = pd.read_csv(DIRECTORY/cleaned)
    # # matches = pd.read_csv(DIRECTORY / 'match.csv')
    # # subject_by_template = template_df[["TEMPLATE_ID", "SUBJECT_ID"]].groupby("TEMPLATE_ID").min()
    #
    # print(template_df)
    #
    # image_counts = template_df[['TEMPLATE_ID']].groupby('TEMPLATE_ID').count()
    #
    # print(image_counts)


if __name__ == '__main__':
    main()
