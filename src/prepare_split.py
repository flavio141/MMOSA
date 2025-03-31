import os
import shutil
import argparse
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

dataHum = pd.read_csv('dataset/MultiomicsFinal.csv', index_col='ID')
dataHum["patient_id"] = dataHum.index.astype(str).str.replace(r"_dp$", "", regex=True)

parser = argparse.ArgumentParser(description='Segmentation and Patching')
parser.add_argument('--column', type=str, default='Outcome at last FU', help='The prediction column of dead or alive')
parser.add_argument('--path', type=str, default='splits', help='Path for the split')
parser.add_argument('--split', type=int, default=5, help='How many splits we desire')


def check_proportion(args):
    for split in os.listdir(args.path):
        df = pd.read_csv(f'{args.path}/{split}')
        print(f'Split {split} has {len(df["train"]) + len(df["val"].dropna())} patients')
        print(f'Train: {len(df["train"])} --> {round(len(df["train"]) / (len(df["train"]) + len(df["val"].dropna())), 2)}')
        print(f'Val: {len(df["val"].dropna())} --> {round(len(df["val"].dropna()) / (len(df["train"]) + len(df["val"].dropna())), 2)}')
        print('\n')


def create_group_stratified(args):
    skf = StratifiedGroupKFold(n_splits=args.split, shuffle=True, random_state=42)

    if os.path.exists(args.path):
        shutil.rmtree(args.path)
    os.makedirs(args.path)

    labels = dataHum[args.column].values
    groups = dataHum["patient_id"].values

    for group, (train_index, test_index) in enumerate(skf.split(dataHum, labels, groups)):

        train_patients = list(dataHum.iloc[train_index].index)
        test_patients = list(dataHum.iloc[test_index].index)

        max_length = max(len(train_patients), len(test_patients))

        train_patients += [''] * (max_length - len(train_patients))
        test_patients += [''] * (max_length - len(test_patients))

        df = pd.DataFrame({'train': train_patients, 'val': test_patients})
        df.to_csv(f'{args.path}/{group}.csv', index=False)

        if os.path.exists(f'{args.path}/{group}.csv'):
            print(f'File number {group + 1} created!')


if __name__ == '__main__':
    args = parser.parse_args()

    create_group_stratified(args)
    check_proportion(args)

