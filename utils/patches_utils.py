import os
import time
import numpy as np
import pandas as pd

import sys
sys.path.append('wsi_utils')

from file_utils import get_from_env
from WSI import WholeSlideImage, StitchCoords


env_variable = get_from_env('config/config.yaml')


def create_folders(args):
    '''
    This function create the folders where we are gonna save the processed images.

    Parameters
    ----------
    args : argparse
        Parameters in order to extract the saving directory

    Returns
    ----------
    directories : dict
        The dictionary to return
    '''

    try:
        patch_save_dir = os.path.join(args.save_dir, env_variable['folders']['patches'])
        mask_save_dir = os.path.join(args.save_dir, env_variable['folders']['masks'])
        stitch_save_dir = os.path.join(args.save_dir, env_variable['folders']['stitches'])

        directories = {'source': args.dataset,
                       'save_dir': args.save_dir,
                       'patch_save_dir': patch_save_dir,
                       'mask_save_dir': mask_save_dir,
                       'stitch_save_dir': stitch_save_dir}

        for key, val in directories.items():
            print(f'Creating-> {key} : {val}')
            os.makedirs(val, exist_ok=True)

            if os.path.exists(val):
                print(f'Folder {val} created!\n')
        
        return directories
    except Exception as error:
        print(f'Error --> {error}')
        raise
    

def initialize_params(args):
    '''
    This function initialize the parameters for the segmentation, filtering etc.

    Parameters
    ----------
    args : argparse
        Parameters in order to extract if we need to read a specific file or we can
        use the default parameters from config.yaml.

    Returns
    ----------
    parameters : dict
        The dictionary to return
    '''

    try:
        seg_params = env_variable['seg_params']
        filter_params = env_variable['filter_params']
        vis_params = env_variable['vis_params']
        patch_params = env_variable['patch_params']

        if args.preset:
            preset_df = pd.read_csv(os.path.join('config', args.preset))
            params = [seg_params, filter_params, vis_params, patch_params]

            for param_set in params:
                for key in param_set:
                    param_set[key] = preset_df.loc[0, key]

        parameters = {'seg_params': seg_params,
                      'filter_params': filter_params,
                      'patch_params': patch_params,
                      'vis_params': vis_params}

        return parameters
    except Exception as error:
        print(f'Error --> {error}')
        raise


def create_df(slides,
              seg_params, 
              filter_params, 
              vis_params, 
              patch_params,
              external_df = False):

    try:
        if external_df:
            slide_ids = slides.slide_id.values
        else:
            slide_ids = slides

        total = len(slides)
        default_df_dict = {'slide_id': slide_ids, 
                           'process': np.full((total), 1, dtype=np.uint8), 
                           'status': np.full((total), 'tbp')}

        default_df_dict.update({
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


        if external_df:
            for key, value in default_df_dict.items():
                if key in slides.columns:
                    slides[key].fillna(value[0], inplace=True)
                else:
                    slides[key] = value
        else:
            slides = pd.DataFrame(default_df_dict)

        return slides, total
    except Exception as error:
        print(f'There was an error --> {error}')
        raise


def set_level(param, WSI_object, level_key):
    '''
    Sets the level of the slide to be considered.

    Parameters:
    ----------
    param : dict
        A dictionary containing parameters, where the level to be set is stored.
    
    WSI_object : object
        An object representing a Whole Slide Image (WSI) from the OpenSlide library.
    
    level_key : str
    A string representing the key in the param dictionary that corresponds to the level to be set.

    Returns:
    ----------
        dict: The modified param dictionary with the updated level of the slide.
    '''
    downsample = 64
    if param[level_key] < 0:
        if len(WSI_object.level_dim) == 1:
            param[level_key] = 0
        else:
            wsi = WSI_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(downsample)
            while wsi.level_dimensions[best_level][0] < 1900:
                downsample = downsample / 2
                best_level = wsi.get_best_level_for_downsample(downsample)
            param[level_key] = best_level
    return param


def stitching(file_path, wsi_object, downscale=64):
    """
    Stitch coordinates from a file onto a whole slide image (WSI).

    Args:
        file_path (str): The path to the file containing the coordinates.
        wsi_object: The whole slide image object onto which the coordinates will be stitched.
        downscale (int, optional): The downscale factor for the coordinates. Defaults to 64.

    Returns:
        tuple: A tuple containing the stitched heatmap and the total time taken for stitching.
    """
    
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
    """
    Segment the Whole Slide Image (WSI) object.

    Args:
        WSI_object (object): The Whole Slide Image object to be segmented.
        seg_params (dict, optional): Parameters for the segmentation process. Defaults to None.
        filter_params (dict, optional): Parameters for filtering the segmented image. Defaults to None.
        mask_file (str, optional): Path to a mask file to be used for segmentation initialization. Defaults to None.

    Returns:
        tuple: A tuple containing the segmented WSI object and the time elapsed for segmentation.
    """

    start_time = time.time()

    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
    """
    Apply patching to the WSI_object.

    Args:
        WSI_object: The object representing the whole slide image.
        **kwargs: Additional keyword arguments to be passed to the WSI_object.process_contours method.

    Returns:
        Tuple: A tuple containing the file path where the patched image is saved and the time elapsed for patching.
    """

    start_time = time.time()

    file_path = WSI_object.process_contours(**kwargs)

    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(source, 
                  save_dir, 
                  patch_save_dir, 
                  mask_save_dir, 
                  stitch_save_dir,
                  patch_size = 224, 
                  step_size = 224,
                  seg_params = env_variable['seg_params'],
                  filter_params = env_variable['filter_params'],
                  vis_params = env_variable['vis_params'],
                  patch_params = env_variable['patch_params'],
                  patch_level = 0,
                  save_mask = True,
                  stitch = False,
                  patch = False, 
                  auto_skip = True, 
                  process_list = None):
    
    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    slides = [slide for slide in slides if slide.split(".")[-1] in ["tiff", "ndpi", "svs"]]
    
    if process_list is None:
        df, total = create_df(slides, seg_params, filter_params, vis_params, patch_params, external_df = False)
    else:
        df = pd.read_csv(process_list)
        df, total = create_df(df, seg_params, filter_params, vis_params, patch_params, external_df = True)
    
    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        idx = df.index[i]
        slide = df.loc[idx, 'slide_id']
        print("\n\nProgress: {:.2f}, {}/{}".format(i / total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id = slide.split('.')[0]

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        try:
            full_path = os.path.join(source, slide)
            WSI_object = WholeSlideImage(full_path)
        except Exception as e:
            print('{} does not work'.format(slide_id))
            df.loc[idx, 'status'] = 'not_working'
            continue

        current_vis_params = {key: df.loc[idx, key] for key in vis_params.keys()}
        current_filter_params = {key: df.loc[idx, key] for key in filter_params.keys()}
        current_seg_params = {key: df.loc[idx, key] for key in seg_params.keys()}
        current_patch_params = {key: df.loc[idx, key] for key in patch_params.keys()}

        for key in vis_params.keys():
            current_vis_params.update({key: df.loc[idx, key]})

        for key in filter_params.keys():
            current_filter_params.update({key: df.loc[idx, key]})

        for key in seg_params.keys():
            current_seg_params.update({key: df.loc[idx, key]})

        for key in patch_params.keys():
            current_patch_params.update({key: df.loc[idx, key]})


        current_vis_params = set_level(current_vis_params, WSI_object, 'vis_level')
        current_seg_params = set_level(current_seg_params, WSI_object, 'seg_level')
        
        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        w, h = WSI_object.level_dim[current_seg_params['seg_level']]
        if w * h > 1e8:
            print(f'Level Dimension {w} x {h} is too large for a successful segmentation, skipping this slide')
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        seg_time_elapsed = -1
        WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id + '.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1
        if patch:
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                                         'save_path': patch_save_dir})
            file_path, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params, )

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id + '.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id + '.jpg')
                heatmap.save(stitch_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times