# Multi-MOdal Survival Attention (MMOSA) deep learning algorithm

This repository contains the code for an AI model designed to perform multiomic analysis on a leukemia datasets. The model is trained to predict survival outcomes based on multiomic data. 


- **Multiomic Analysis:** The model integrates various omics data types to provide a comprehensive analysis.
  
- **Survival Prediction:** Using deep learning learning techniques, the model predicts survival outcomes for leukemia patients, offering valuable insights for personalized medicine approaches.


## Prerequisites & Configuration
To use the model, follow these steps:

1. Make sure you have at least **Python 3.11.5** installed on your system (it was tested on python 3.8.10 too, but it is better to update some libreries that requires python 3.10);
2. Clone this repository with the command:
```bash
git clone https://github.com/flavio141/MMOSA.git
```
3. You need to create a python environment and to activate it in order to install all the packages;
4. Install openslide by selecting the correct distribution at the link: https://openslide.org/download/ ;
5. Install the dependencies listed in the `requirements.txt` file with the following command:

```bash
pip install -r requirements.txt
```

If it is possibile, **pytorch** may be installed without the command above because it is better to invoke each torch package in one command.

## Configuration

**Dataset**: 
- Place the dataset inside the folder named `dataset`. It is necessary to put all the slides images in any format: .tiff, .svs, .ndpi. This folder requires also the excel dataset **Multiomics_dataset_020823v_updt.xlsx**.

**Image processing configuration**:
- Modify the `config.yaml` file if you want different image processing configurations.

**TensorboardX**
- You can use tensorboard in order to look at the training/val results. You can open it with the command:

```bash
tensorboard --logdir=logs
```

Of course it is necessary before to create the folder log if it is not available. To close you can just press CTRL + C and in some cases you need to close the port 6006 with the command:

```bash
fuser -k 6006/tcp
```

Do not worry for other folders, they will be created at each step.

## 1. Extracting Patches
Now, for the first part, execute the following script by use this command:

```bash
python src/extract_patches.py
```

You can pass different parameters for changing the default name of the folders, it will be explained in a doc file. 

Once the execution is complete, the extracted patches will be available for further analysis.

## 2. Create Features
The second part creates all the features for the Survival. It is easy because it requires only the following command:

```bash
python src/extract_features.py
```
## 3. Create Splits
We are ready for the Survival Analysis. But before, we need to create the splits. In order to have a 80/20 stratified splits and a 5-fold Cross-Validation.
First of all we need to go inside the directory **preprocessing** and launch all the notebook preproc.ipynb in order to create the dataset .csv cleaned.

After this step, we have to launch this command:

```bash
python src/prepare_split.py
```

