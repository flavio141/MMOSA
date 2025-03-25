import argparse

import sys
sys.path.append('utils')

from patches_utils import *

parser = argparse.ArgumentParser(description='Segmentation and Patching')
parser.add_argument('--dataset', type=str, default='dataset_mgg', help='folder of the dataset')
parser.add_argument('--save_dir', type=str, default='result_dir_mgg', help='folder for the images processed')
parser.add_argument('--preset', type=str, default=None, help='preset of values for segmentation, filtering and patching')
parser.add_argument('--process', type=str, default=None, help='specific values for each slide. Insert the path')
parser.add_argument('--patch_size', type=int, default=224, help='the size of the patches')
parser.add_argument('--step_size', type=int, default=224, help='the size of the step in order to modify the patch')
parser.add_argument('--stitch', type=bool, default=True, help='boolean parameter in order to save the stitches')
parser.add_argument('--patch_level', type=int, default=1, help='level of the WSI')
parser.add_argument('--patch', type=bool, default=True, help='boolean parameter in order to save the patches')
parser.add_argument('--no_auto_skip', type=bool, default=True, help='tell if you like to skip pre-processed images. Default: true')


if __name__ == '__main__':
    args = parser.parse_args()

    directories = create_folders(args)
    parameters = initialize_params(args)

    seg_times, patch_times = seg_and_patch(**directories, 
                                           **parameters,
                                           patch_size=args.patch_size, 
                                           step_size=args.step_size,
                                           patch_level=args.patch_level, 
                                           save_mask=True,
                                           stitch=args.stitch,
                                           patch=args.patch, 
                                           auto_skip=args.no_auto_skip, 
                                           process_list=args.process)

