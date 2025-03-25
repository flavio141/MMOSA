import os
import sys
sys.path.append('utils')
sys.path.append('models')

import yaml
import h5py
import torch
import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from models import ModelSurv
from surv_utils import create_bins, dataHum


def get_from_env(file_path):
    """
    Reads a YAML file from the given file path and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as error:
        print(f'There was a problem --> {error}')


def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()


def load_pkl(filename):
    loader = open(filename, 'rb')
    file = pickle.load(loader)
    loader.close()
    return file


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    """
    Save data to an HDF5 file.

    Args:
        output_path (str): The path to the output HDF5 file.
        asset_dict (dict): A dictionary containing the data to be saved. The keys represent the dataset names,
                           and the values represent the corresponding data arrays.
        attr_dict (dict, optional): A dictionary containing attributes to be added to the datasets. The keys
                                    represent the dataset names, and the values represent dictionaries of attribute
                                    names and values. Defaults to None.
        mode (str, optional): The file mode to open the HDF5 file. Defaults to 'a'.

    Returns:
        str: The path to the output HDF5 file.
    """
    
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def initialize_hdf5_bag(first_patch, save_coord=False):
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path = tuple(
        first_patch.values())
    file_path = os.path.join(save_path, name) + '.h5'
    file = h5py.File(file_path, "w")
    img_patch = np.array(img_patch)[np.newaxis, ...]
    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[1:]  # maximum dimensions up to which dataset maybe resized (None means unlimited)
    dset = file.create_dataset('imgs',
                               shape=img_shape, maxshape=maxshape, chunks=img_shape, dtype=dtype)

    dset[:] = img_patch
    dset.attrs['patch_level'] = patch_level
    dset.attrs['wsi_name'] = name
    dset.attrs['downsample'] = downsample
    dset.attrs['level_dim'] = level_dim
    dset.attrs['downsampled_level_dim'] = downsampled_level_dim

    if save_coord:
        coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
        coord_dset[:] = (x, y)

    file.close()
    return file_path


def sample_indices(scores, k, start=0.48, end=0.52, convert_to_percentile=False, seed=1):
    np.random.seed(seed)
    if convert_to_percentile:
        end_value = np.quantile(scores, end)
        start_value = np.quantile(scores, start)
    else:
        end_value = end
        start_value = start
    score_window = np.logical_and(scores >= start_value, scores <= end_value)
    indices = np.where(score_window)[0]
    if len(indices) < 1:
        return -1
    else:
        return np.random.choice(indices, min(k, len(indices)), replace=False)


def top_k(scores, k, invert=False):
    if invert:
        top_k_ids = scores.argsort()[:k]
    else:
        top_k_ids = scores.argsort()[::-1][:k]
    return top_k_ids


def to_percentiles(scores):
    scores = rankdata(scores, 'average') / len(scores) * 100
    return scores


def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords


def top_values_unique_columns(scores, top_k_row=15):
    flat_indices = np.argsort(scores, axis=None)[::-1]
    selected_rows = set()
    selected_cols = set()
    top_rows = []

    for flat_idx in flat_indices:
        row_idx, col_idx = np.unravel_index(flat_idx, scores.shape)
        
        if col_idx not in selected_cols:
            top_rows.append(row_idx)
            selected_rows.add(row_idx)
            selected_cols.add(col_idx)
        
        if len(top_rows) == top_k_row:
            break
    
    return top_rows


def sample_rois(scores, coords, k=5, mode='range_sample', seed=1, score_start=0.45, score_end=0.55, top_left=None,
                bot_right=None):

    #scores = to_percentiles(scores)
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    if mode == 'range_sample':
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False,
                                     seed=seed)
    elif mode == 'topk':
        sampled_ids = top_k(scores, k, invert=False)
    elif mode == 'reverse_topk':
        sampled_ids = top_k(scores, k, invert=True)
    elif mode == 'custom':
        sampled_ids = top_values_unique_columns(scores, top_k_row=k)
    else:
        raise NotImplementedError
    
    coords = coords[sampled_ids]
    scores = scores[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset


def initialize_df(slides, seg_params, filter_params, vis_params, patch_params):

    total = len(slides)
    if isinstance(slides, pd.DataFrame):
        slide_ids = slides.slide_id.values
    else:
        slide_ids = slides

    default_df_dict = {'slide_id': slide_ids,
                       'process': np.full((total), 1, dtype=np.uint8)}
    
    modified_data = create_bins(dataHum, label_col=None, n_bins=4, eps=1e-6)

    default_df_dict.update({
        'status': np.full((total), 'tbp'),
        # seg params
        'seg_level': np.full((total), int(seg_params['seg_level']), dtype=np.int8),
        'mthresh': np.full((total), int(seg_params['mthresh']), dtype=np.uint8),
        'kernel_size': np.full((total), int(seg_params['kernel_size']), dtype=np.uint32),
        'use_otsu': np.full((total), bool(seg_params['use_otsu']), dtype=bool),
        'keep_ids': np.full((total), seg_params['keep_ids']),
        'exclude_ids': np.full((total), seg_params['exclude_ids']),

        # filter params
        'a_t': np.full((total), int(filter_params['a_t']), dtype=np.float32),
        'a_h': np.full((total), int(filter_params['a_h']), dtype=np.float32),
        'max_n_holes': np.full((total), int(filter_params['max_n_holes']), dtype=np.uint32),

        # vis params
        'vis_level': np.full((total), int(vis_params['vis_level']), dtype=np.int8),
        'line_thickness': np.full((total), int(vis_params['line_thickness']), dtype=np.uint32),

        # patching params
        'use_padding': np.full((total), bool(patch_params['use_padding']), dtype=bool),
        'contour_fn': np.full((total), patch_params['contour_fn'])
    })


    if isinstance(slides, pd.DataFrame):
        temp_copy = pd.DataFrame(default_df_dict)
          
        for key in default_df_dict.keys(): 
            if key in slides.columns:
                mask = slides[key].isna()
                slides.loc[mask, key] = temp_copy.loc[mask, key]
            else:
                slides.insert(len(slides.columns), key, default_df_dict[key])
    else:
        slides = pd.DataFrame(default_df_dict)
    
    slides['patient'] = slides['slide_id'].apply(lambda x: x.split('_')[0])
    merged_df = slides.merge(modified_data['time_label'], left_on='patient', right_index=True, how='left')

    return merged_df


def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {'dropout': 0.4, 'input_size': args.input_size, 'surv_nodes': [1536, 64, 1]}
    
    model = ModelSurv(**model_dict)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.eval()
    return model