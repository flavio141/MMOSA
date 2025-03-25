import os
import torch
import shutil
import argparse
import openslide

from tqdm import tqdm

import sys
sys.path.append('utils')

from features_utils import get_dino_bloom, trasform_img

torch.set_num_threads(100)


parser = argparse.ArgumentParser(description='Extract Features')
parser.add_argument('--source', type=str, default='dataset', help='folder of the dataset')
parser.add_argument('--process', type=str, default='processed_images_trial', help='folder for the processed features')
parser.add_argument('--features_dir', type=str, default='features_complete', help='folder for the features')
parser.add_argument('--eliminate', type=bool, default=False, help='eliminate the processed images folder')


def create_features_dino(args, slide, wsi, model, trsforms, size=(224, 224), level=-1):    
    if os.path.exists(os.path.join(args.features_dir, slide.split('.')[0] + '.pt')):
        print(f'Slide {slide.split(".")[0]} already processed')
    else:
        downscale = 128
        while wsi.level_dimensions[level][0] < 4000:
            downscale = downscale / 2
            level = wsi.get_best_level_for_downsample(downscale)
        image = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert("RGB").resize(size)
        
        output_path = os.path.join(args.process, f"{slide.split('.')[0]}_resized.png")
        image.save(output_path)
        tensor_predict = trsforms(image)

        features = model.forward_features(tensor_predict.unsqueeze(0).cuda())
        torch.save(features['x_norm_patchtokens'].reshape(1 * 16 * 16, 1024).clone(), os.path.join(args.features_dir, f'{slide.split(".")[0]}.pt'))

if __name__ == '__main__':
    args = parser.parse_args()

    slides = sorted(os.listdir(args.source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(args.source, slide))]
    slides = [slide for slide in slides if slide.split(".")[-1] in ["tiff", "ndpi", "svs"]]

    print(f'Creating -> {args.process} and {args.features_dir} folder')
    if os.path.exists(args.process) and args.eliminate:
        shutil.rmtree(args.process)

    if os.path.exists(args.features_dir) and args.eliminate:    
        shutil.rmtree(args.features_dir)
    
    os.makedirs(args.process, exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)
    print(f'Folder {args.process} and {args.features_dir} created or already available!\n')

    model = get_dino_bloom()
    trsforms = trasform_img()

    if torch.cuda.is_available():
        model.cuda()
    
    slides_to_skip = ['5308132_21-1000_5308132_21-1000_HE_2.tiff', '5216613_21-6946_5216613_21-6946_HE 2.tiff', '572711_YY-ART_5326968_21-5465_HE 2.tiff']

    
    for slide in tqdm(slides, desc='Extracting Features by Resizing'):
        file_path = os.path.join(args.source, slide)
        if slide in slides_to_skip:
            continue

        wsi = openslide.open_slide(file_path)
        create_features_dino(args, slide, wsi, model, trsforms)

