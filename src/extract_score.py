import os
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Evaluate mean for the trials specified")
parser.add_argument("--dir", type=str, default='old_he_mgg/results_multiomics/4', help="Directory with the .csv files from the folds")
parser.add_argument("--optuna", type=bool, default=False, help="Directory with the .csv files from the folds")
parser.add_argument("--model", type=str, default='ModelAtt', help="Model to extract the information from")
parser.add_argument("--trial", type=str, default='4', help='Trial number to extract the information from')
parser.add_argument("--last_epochs", type=int, default=5, help='Trial number to extract the information from')
#16 for Dino


def create_plots(df, trial):
    for c_index in ['Val_C-Index', 'Train_C-Index']:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df[c_index], marker='o', linestyle='-', color='b')
        plt.title(f'{c_index} vs Epoche')
        plt.xlabel('Epoche')
        plt.ylabel(c_index)
        plt.grid(True)

        plt.show()
        plt.savefig(f'plots/{trial}_{c_index.lower().replace("_", "-")}_vs_epoche.png')


def box_plots_last(c_values, folds):
    cmap = plt.get_cmap('tab10', len(folds))

    plt.figure(figsize=(10, 6))
    box = plt.boxplot(c_values, patch_artist=True)

    for patch, color in zip(box['boxes'], cmap.colors):
        patch.set_facecolor(color)

    plt.xticks(range(1, len(folds) + 1), folds)
    plt.xlabel('Fold')
    plt.ylabel('C-Index')
    plt.title('Boxplot of for each Fold')
    plt.show()
    plt.savefig(f'plots/boxplot.png')


def main(args, directory):
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' doesn't exists")
        raise
    
    val_c_index_totals = []
    train_c_index_totals = []

    val_c_indexes_boxplot = {}
    train_c_indexes_boxplot = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            
            if 'Val_C-Index' in df.columns and 'Train_C-Index' in df.columns:
                val_c_index_totals.append(df['Val_C-Index'][-args.last_epochs:].mean())
                train_c_index_totals.append(df['Train_C-Index'][-args.last_epochs:].mean())


                val_c_indexes_boxplot[filename.split('_')[-1].split('.')[0]] = list(df['Val_C-Index'][-args.last_epochs:])
                train_c_indexes_boxplot[filename.split('_')[-1].split('.')[0]] = list(df['Train_C-Index'][-args.last_epochs:])
            else:
                print(f"The file '{filename}' doesn't have the corrected columns")
    
    if val_c_index_totals and train_c_index_totals:
        final_val_c_index_mean = np.mean(val_c_index_totals)
        final_train_c_index_mean = np.mean(train_c_index_totals)

        print(f"Mean Training C-Index over 5Fold CV: {round(final_train_c_index_mean, 2)} \u00B1 {round(np.std(train_c_index_totals), 2)} {np.quantile(train_c_index_totals, q=[0.025, 0.975])}")
        print(f"Mean Validation C-Index over 5Fold CV: {round(final_val_c_index_mean, 2)} \u00B1 {round(np.std(val_c_index_totals), 2)} {np.quantile(val_c_index_totals, q=[0.025, 0.975])}")


    box_plots_last(list(val_c_indexes_boxplot.values()), list(val_c_indexes_boxplot.keys()))



def optuna_validation(directory, model, trial):
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' doesn't exists")
        raise
    
    val_c_index_totals = []
    train_c_index_totals = []

    val_c_indexes_boxplot = {}
    train_c_indexes_boxplot = {}

    if os.path.exists('plots'):
        shutil.rmtree('plots')
    
    os.makedirs('plots')

    for sub_dir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, sub_dir)) and sub_dir == model:
            for filename in os.listdir(os.path.join(directory, sub_dir)):
                if filename.endswith(".csv") and filename.split('_')[0] == f'trial{trial}':
                    filepath = os.path.join(directory, sub_dir, filename)
                    df = pd.read_csv(filepath)
                    
                    if 'Val_C-Index' in df.columns and 'Train_C-Index' in df.columns:
                        val_c_index_totals.append(df['Val_C-Index'][-1:].mean())
                        train_c_index_totals.append(df['Train_C-Index'][-1:].mean())

                        val_c_indexes_boxplot[filename.split('_')[-1].split('.')[0]] = list(df['Val_C-Index'][-50:])
                        train_c_indexes_boxplot[filename.split('_')[-1].split('.')[0]] = list(df['Train_C-Index'][-50:])
                    else:
                        print(f"The file '{filename}' doesn't have the corrected columns")

                    create_plots(df, filename.split('_')[-1].split('.')[0])
                else:
                    continue

                box_plots_last(list(val_c_indexes_boxplot.values()), list(val_c_indexes_boxplot.keys()))


    if val_c_index_totals and train_c_index_totals:
        final_val_c_index_mean = np.mean(val_c_index_totals)
        final_train_c_index_mean = np.mean(train_c_index_totals)
        
        print(f"Mean Validation C-Index over 5Fold CV: {round(final_val_c_index_mean, 2)} \u00B1 {round(np.std(val_c_index_totals), 2)} {np.quantile(val_c_index_totals, q=[0.025, 0.975])}")
        print(f"Mean Training C-Index over 5Fold CV: {round(final_train_c_index_mean, 2)} \u00B1 {round(np.std(train_c_index_totals), 2)} {np.quantile(train_c_index_totals, q=[0.025, 0.975])}")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.optuna:
        optuna_validation('optuna_best', args.model, args.trial)
    else:
        main(args, args.dir)
