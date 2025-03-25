import os
import re
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import argparse

dataHum = pd.read_csv('dataset/MultiomicsFinal.csv', index_col='ID')

parser = argparse.ArgumentParser(description='Segmentation and Patching')
parser.add_argument('--column', type=str, default='Outcome at last FU', help='The prediction column of dead or alive')
parser.add_argument('--path', type=str, default='splits', help='Path for the split')
parser.add_argument('--split', type=int, default=5, help='How many splits we desire')
parser.add_argument('--without_yy', type=bool, default=False, help='How many splits we desire')
parser.add_argument('--groups', type=bool, default=True, help='If we want to use groups for the splits')
parser.add_argument('--features', type=str, default='features_dino', help='Features directory')


def generate_mapping(args):
    mapping = {}

    for id in dataHum.index:
        slides_features = os.listdir(args.features)
        
        if "_dp" in id:
            id = id[:-3]
        
        mapping[id] = []

        for slide in slides_features:
            if (os.path.join(args.features, slide).endswith('.pt')) and (id == slide.split('_')[0]):
                mapping[id].append(slide)

    return {k: v for k, v in mapping.items() if len(v) != 0}


def check_proportion(args):
    for split in os.listdir(args.path):
        df = pd.read_csv(f'{args.path}/{split}')
        print(f'Split {split} has {len(df["train"]) + len(df["val"].dropna())} patients')
        print(f'Train: {len(df["train"])} --> {round(len(df["train"]) / (len(df["train"]) + len(df["val"].dropna())), 2)}')
        print(f'Val: {len(df["val"].dropna())} --> {round(len(df["val"].dropna()) / (len(df["train"]) + len(df["val"].dropna())), 2)}')
        print('\n')


def prepare_dataset_new(args):
    mapping = generate_mapping(args)
    list_of_indexes = list(mapping.keys())

    for idx in dataHum.index:
        if '_dp' in idx and idx.replace('_dp', '') in list_of_indexes:
            list_of_indexes.append(idx)

    dataNew = dataHum.loc[dataHum.index.isin(list_of_indexes)].groupby(level=0).first()
    
    dataNew = dataNew[dataNew[args.column].notna()]
    X = dataNew.drop(columns=[args.column])
    y = dataNew[args.column]

    return X, y



def prepare_dataset(args):
    train_prepared = []
    duplicates = [index.split('_')[0] for index in dataHum.index if '_dp' in index]
    ids_map = {file.split('_')[0]: file.split('_')[2] for file in os.listdir(args.features) if 'YY-ART' in file}

    for id, heir in ids_map.items():
        if id in list(dataHum.index) and id not in train_prepared:
            train_prepared.append(id)

        if heir in list(dataHum.index) and heir not in train_prepared:
            train_prepared.append(heir)

    for id in duplicates:
        if id not in train_prepared:
            train_prepared.append(id)

        if id + '_dp' not in train_prepared:
            train_prepared.append(id + '_dp')

    dataHum = dataHum[dataHum[args.column].notna()]
    X = dataHum.drop(columns=[args.column])
    y = dataHum[args.column]

    X.drop(train_prepared, axis=0, inplace=True)
    y.drop(train_prepared, axis=0, inplace=True)

    return X, y, train_prepared


def prepare_groups(args):
    files = os.listdir('dataset')
    groups = {}
    group_counter = 0

    for filename in files:
        if not filename.endswith(('.tiff', '.ndpi', '.svs', '.tif')):
            continue

        splitted = filename.split('_')
        id = splitted[0]
        if 't0' in splitted and 'MGG' in splitted:
            id_derivato = splitted[0]
        else:
            pattern = r'.+-.+'

            for pos, number in enumerate(splitted):
                if re.match(pattern, number):
                    if '-' in splitted[pos + 1]:
                        id_derivato = splitted[0]
                    elif 'HE' in splitted[pos + 1] or 'MGG' in splitted[pos + 1]:
                        id_derivato = splitted[0]
                    else:
                        id_derivato = splitted[pos + 1]
                    break

        if 'YY-ART' in filename and id_derivato in list(dataHum.index):
            if id not in list(groups.keys()) and id_derivato not in list(groups.keys()):
                groups[id] = group_counter
                groups[id_derivato] = group_counter
                group_counter += 1
            elif id in list(groups.keys()) and id_derivato not in list(groups.keys()):
                groups[id_derivato] = groups[id]
            elif id not in list(groups.keys()) and id_derivato in list(groups.keys()):
                groups[id] = groups[id_derivato]
            else:
                pass
        else:
            if id not in list(groups.keys()):
                groups[id] = group_counter
                group_counter += 1
            else:
                pass

    dataHum['group'] = None

    for idx in list(groups.keys()):
        if idx in list(dataHum.index):
            dataHum.loc[idx, 'group'] = groups[idx]

    for idx in list(dataHum.index):
        if '_dp' in idx:
            base_idx = idx.replace('_dp', '')
            if base_idx in list(groups.keys()):
                dataHum.loc[idx, 'group'] = groups[base_idx]

    dataHum.dropna(subset=['group'], inplace=True)

    X = dataHum.drop(columns=[args.column])
    y = dataHum[args.column]

    return X, y, list(X['group'])


def create_group_stratified(args, X, y, group=None):

    skf = StratifiedKFold(n_splits=args.split, shuffle=True, random_state=42) #StratifiedGroupKFold(n_splits=args.split, shuffle=True, random_state=42)

    if os.path.exists(args.path):
        shutil.rmtree(args.path)
    
    os.makedirs(args.path)

    for group, (train_index, test_index) in enumerate(skf.split(X, y)):#, groups=group)):

        train_patients = list(X.iloc[train_index].index)
        test_patients = list(X.iloc[test_index].index)

        max_length = max(len(train_patients), len(test_patients))

        train_patients += [''] * (max_length - len(train_patients))
        test_patients += [''] * (max_length - len(test_patients))

        df = pd.DataFrame({'train': train_patients, 'val': test_patients})
        df.to_csv(f'{args.path}/{group}.csv', index=False)

        if os.path.exists(f'{args.path}/{group}.csv'):
            print(f'File number {group + 1} created!')


if __name__ == '__main__':
    args = parser.parse_args()
    #X, y, groups = prepare_groups(args)
    X, y = prepare_dataset_new(args)
    print('Init creation of Splits by using StratifiedGroupKFold')
    create_group_stratified(args, X, y)#, groups)


