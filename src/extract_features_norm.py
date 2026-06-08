import os
import timm
import h5py
import torch
import shutil
import argparse
import openslide

import warnings
warnings.filterwarnings('ignore')

#os.environ['NUMEXPR_MAX_THREADS'] = '15'
#os.sched_setaffinity(0, range(12))

from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2

from tiatoolbox import data
from tiatoolbox.tools import stainnorm
from tiatoolbox.wsicore import wsireader

import sys
sys.path.append('utils')

from features_utils import get_dino_bloom, trasform_img

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Extract Features')
# Folders
parser.add_argument('--source', type=str, default='dataset', help='folder of the dataset')
parser.add_argument('--process', type=str, default='processed_images', help='folder for the processed features')
parser.add_argument('--features_giga', type=str, default='features_giga', help='folder for the features')
parser.add_argument('--features_dino', type=str, default='features_dino_not_norm', help='folder for the features')
parser.add_argument('--h5', type=str, default='result_dir/patches', help='folder for the patches to read')

# Settings
parser.add_argument('--model', type=str, default='dino', help='select the model to extract features')
parser.add_argument('--eliminate', type=bool, default=False, help='eliminate the processed images folder')
parser.add_argument('--keep_feat', type=bool, default=True, help='keep features for each patch')
parser.add_argument('--normalize', type=str, default=False, help='select why server to run the code')
parser.add_argument('--use_normalized', type=bool, default=False, help='use normalized images for feature extraction')


os.environ['HF_TOKEN'] = ''


def create_features_dino(args, slide, model, trsforms):    
    with h5py.File(h5_path, "r") as f:
        iterator = tqdm(range(len(f['coords'])), desc='Extract Features with DinoBloom')
        
        if os.path.exists(os.path.join(args.features_dino, slide.split('.')[0])) and len(os.listdir(os.path.join(args.features_dino, slide.split('.')[0]))) == len(f['coords']):
                iterator.close()
                print(f'Slide {slide.split(".")[0]} already processed')
        else:
            for idx in iterator:
                if not os.path.exists(f"{os.path.join(args.process, slide.split('.')[0])}/{idx}.jpg"):
                    continue
                patch = Image.open(f"{os.path.join(args.process, slide.split('.')[0])}/{idx}.jpg")
                tensor_predict = trsforms(patch)

                features = model.forward_features(tensor_predict.unsqueeze(0).cuda())
                torch.save(features['x_norm_clstoken'].clone(), f'{args.features_dino}/{slide.split(".")[0]}/{f["coords"][idx][0]}_{f["coords"][idx][1]}.pt')
    
    try:
        file_list = [file for file in os.listdir(os.path.join(args.features_dino, slide.split('.')[0])) if file.endswith('.pt')]
        stacked_tensor = torch.cat([torch.load(os.path.join(f'{args.features_dino}/{slide.split(".")[0]}', file)) for file in file_list], dim=0)
    except FileNotFoundError as error:
        print(f'No files found in {os.path.join(args.features_dino, slide.split(".")[0])}: {error}')

    if not args.keep_feat:
        shutil.rmtree(os.path.join(args.features_dino, slide.split('.')[0]))

    torch.save(stacked_tensor.squeeze(dim=1), f'{args.features_dino}/{slide.split(".")[0]}.pt')


def create_features_dino_not_normalized(args, slide, wsi, model, trsforms, size=(224, 224), level=-1):    
    with h5py.File(h5_path, "r") as f:
        iterator = tqdm(range(len(f['coords'])), desc='Extract Features with DinoBloom')
        
        for idx in iterator:
            downscale = 128
            while wsi.level_dimensions[level][0] < 4000:
                downscale = downscale / 2
                level = wsi.get_best_level_for_downsample(downscale)
            image = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert("RGB").resize(size)
            tensor_predict = trsforms(image)
            
            features = model.forward_features(tensor_predict.unsqueeze(0).cuda())
            torch.save(features['x_norm_clstoken'].clone(), f'{args.features_dino}/{slide.split(".")[0]}/{f["coords"][idx][0]}_{f["coords"][idx][1]}.pt')
    
    try:
        file_list = [file for file in os.listdir(os.path.join(args.features_dino, slide.split('.')[0])) if file.endswith('.pt')]
        stacked_tensor = torch.cat([torch.load(os.path.join(f'{args.features_dino}/{slide.split(".")[0]}', file)) for file in file_list], dim=0)
    except FileNotFoundError as error:
        print(f'No files found in {os.path.join(args.features_dino, slide.split(".")[0])}: {error}')

    if not args.keep_feat:
        shutil.rmtree(os.path.join(args.features_dino, slide.split('.')[0]))

    torch.save(stacked_tensor.squeeze(dim=1), f'{args.features_dino}/{slide.split(".")[0]}.pt')


def create_features_giga(args, slide, model):    
    with h5py.File(h5_path, "r") as f:
        iterator = tqdm(range(len(f['coords'])), desc='Creating Patches and Extracting Features')
        if os.path.exists(os.path.join(args.features_giga, slide.split('.')[0])) and len(os.listdir(os.path.join(args.features_giga, slide.split('.')[0]))) == len(f['coords']):
                iterator.close()
                print(f'Slide {slide.split(".")[0]} already processed')
        else:
            for idx in iterator:
                patch = Image.open(f"{os.path.join(args.process, slide.split('.')[0])}/{idx}.jpg")
                model.eval()
                with torch.no_grad():
                    features = model(v2.functional.to_tensor(patch).unsqueeze(0).cuda('cuda:0'))
                torch.save(features.clone(), f'{args.features_dir}/{slide.split(".")[0]}/tensor_{idx}.pt')

    try:
        file_list = [file for file in os.listdir(os.path.join(args.features_dir, slide.split('.')[0])) if file.endswith('.pt')]
        stacked_tensor = torch.cat([torch.load(os.path.join(f'{args.features_dir}/{slide.split(".")[0]}', file)) for file in file_list], dim=0)
    except FileNotFoundError as error:
        print(f'No files found in {os.path.join(args.features_dir, slide.split(".")[0])}: {error}')

    if not args.keep_feat:
        shutil.rmtree(os.path.join(args.features_dir, slide.split('.')[0]))

    torch.save(stacked_tensor.squeeze(dim=1), f'{args.features_dir}/{slide.split(".")[0]}.pt')


def normalize(args, slide, stain_normalizer, level=-1, size=224):
    with h5py.File(h5_path, "r") as f:
        iterator = tqdm(range(len(f['coords'])), desc='Normalizing')
        
        for idx in iterator:
            if len(os.listdir(os.path.join(args.process, slide.split('.')[0]))) == len(f['coords']):
                iterator.close()
                print(f'Slide {slide.split(".")[0]} already normalized')
                break
            
            coord = f['coords'][idx]
            image = wsi.read_region(location=coord, level=level, size=[size, size])
            normed_sample = stain_normalizer.transform(image.copy())

            patch = Image.fromarray(normed_sample)
            patch.save(f"{os.path.join(args.process, slide.split('.')[0])}/{idx}.jpg")


if __name__ == '__main__':
    args = parser.parse_args()

    slides = sorted(os.listdir(args.source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(args.source, slide))]
    slides = [slide for slide in slides if slide.split(".")[-1] in ["tiff", "ndpi", "svs", "tif"]]

    print(f'Creating Folder -> {args.process}')
    
    os.makedirs(args.process, exist_ok=True)

    if args.model == 'dino':
        os.makedirs(args.features_dino, exist_ok=True)
        model = get_dino_bloom()
        trsforms = trasform_img()
    else:
        os.makedirs(args.features_giga, exist_ok=True)
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    if args.normalize:
        target_image = data.stain_norm_target()
        stain_normalizer = stainnorm.VahadaneNormalizer()
        stain_normalizer.fit(target_image)


    if torch.cuda.is_available():
        model.cuda('cuda:0')


    for i, slide in enumerate(slides):
        print("\n\nProgress: {:.2f}, {}/{}".format((i + 1) / len(slides), i + 1, len(slides)))
        file_path = os.path.join(args.source, slide)
        h5_path = os.path.join(args.h5, slide.split('.')[0] + '.h5')

        if not os.path.exists(h5_path):
            print(f'File not present: {h5_path}')
            continue

        if args.normalize:
            print(file_path)
            wsi = wsireader.OpenSlideWSIReader.open(file_path)
            os.makedirs(os.path.join(args.process, slide.split('.')[0]), exist_ok=True)
            normalize(args, slide, stain_normalizer, level=1)

        print(file_path)
        if args.model == 'dino' and args.use_normalized:
            if os.path.exists(os.path.join(args.features_dino, slide.split('.')[0] + '.pt')) or not os.path.exists(os.path.join(args.process, slide.split('.')[0])):
                continue
            
            os.makedirs(os.path.join(args.features_dino, slide.split('.')[0]), exist_ok=True)
            create_features_dino(args, slide, model, trsforms)
        elif args.model == 'dino' and not args.use_normalized:
            if os.path.exists(os.path.join(args.features_dino, slide.split('.')[0] + '.pt')) or not os.path.exists(h5_path):
                continue

            wsi = openslide.open_slide(file_path)
            os.makedirs(os.path.join(args.features_dino, slide.split('.')[0]), exist_ok=True)
            create_features_dino_not_normalized(args, slide, wsi, model, trsforms)
        else:
            if os.path.exists(os.path.join(args.features_giga, slide.split('.')[0] + '.pt')):
                continue

            os.makedirs(os.path.join(args.features_giga, slide.split('.')[0]), exist_ok=True)
            create_features_giga(args, slide, model)

