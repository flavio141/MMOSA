import os
import h5py
import timm
import torch
import shutil
import argparse
import openslide

from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import v2

import sys
sys.path.append('utils')

from features_utils import tensor_to_image, get_dino_bloom, trasform_img
from resnet_custom import resnet50_baseline
from color_normalization import normalizeStaining

#torch.set_num_threads(100)


parser = argparse.ArgumentParser(description='Extract Features')
parser.add_argument('--source', type=str, default='dataset_mgg', help='folder of the dataset')
parser.add_argument('--process', type=str, default='processed_images_mgg', help='folder for the processed features')
parser.add_argument('--features_dir', type=str, default='features_mgg', help='folder for the features')
parser.add_argument('--h5', type=str, default='result_dir_mgg/patches', help='folder for the patches to read')
parser.add_argument('--model', type=str, default='gigapath', help='folder where the models are saved')
parser.add_argument('--keep_feat', type=bool, default=False, help='keep features for each patch')
parser.add_argument('--eliminate', type=bool, default=False, help='eliminate the processed images folder')
parser.add_argument('--stained', type=bool, default=False, help='normalize the stained image')


os.environ['HF_TOKEN'] = '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'

#@jit(nopython=False)
def create_features(args, h5_path, slide, wsi, model, trsforms):
    with h5py.File(h5_path, "r") as f:
        iterator = tqdm(range(len(f['coords'])), desc='Creating Patches and Extracting Features')
        
        for idx in iterator:
            if len(os.listdir(os.path.join(args.features_dir, slide.split('.')[0]))) == len(f['coords']):
                iterator.close()
                print(f'Slide {slide.split(".")[0]} already processed')
                break
            elif (len(os.listdir(os.path.join(args.features_dir, slide.split('.')[0]))) != 0) and (os.path.exists(f'{args.features_dir}/{slide.split(".")[0]}/tensor_{idx}.pt')):
                continue
            else:
                coord = f['coords'][idx]
                img = wsi.read_region(coord, 1, (224, 224)).convert('RGB')
                image_tensor, tensor_predict = tensor_to_image(img, trsforms)
                if not os.path.exists(f'{args.process}/{slide.split(".")[0]}/{idx}.jpg'):
                    image_tensor.save(f"{os.path.join(args.process, slide.split('.')[0])}/{idx}.jpg")

                if args.stained == True:
                    image_tensor = normalizeStaining(image_tensor)

                if args.model == 'dinobloom':
                    features = model.forward_features(tensor_predict.unsqueeze(0).cuda())
                    torch.save(features['x_norm_clstoken'].clone(), f'{args.features_dir}/{slide.split(".")[0]}/tensor_{idx}.pt')
                elif args.model == 'gigapath':
                    model.eval()
                    with torch.no_grad():
                        if args.stained == True:
                            image_tensor = v2.functional.to_pil_image(image_tensor)
                        features = model(v2.functional.to_tensor(image_tensor).unsqueeze(0).cuda())
                    torch.save(features.clone(), f'{args.features_dir}/{slide.split(".")[0]}/tensor_{idx}.pt')
                else:
                    features = model(tensor_predict.unsqueeze(0).cuda())
                    torch.save(features.clone(), f'{args.features_dir}/{slide.split(".")[0]}/tensor_{idx}.pt')
    try:
        file_list = [file for file in os.listdir(os.path.join(args.features_dir, slide.split('.')[0])) if file.endswith('.pt')]
        stacked_tensor = torch.cat([torch.load(os.path.join(f'{args.features_dir}/{slide.split(".")[0]}', file)) for file in file_list], dim=0)
    except FileNotFoundError as error:
        print(f'No files found in {os.path.join(args.features_dir, slide.split(".")[0])}: {error}')

    if not args.keep_feat:
        shutil.rmtree(os.path.join(args.features_dir, slide.split('.')[0]))

    torch.save(stacked_tensor.squeeze(dim=1), f'{args.features_dir}/{slide.split(".")[0]}.pt')

if __name__ == '__main__':
    args = parser.parse_args()

    slides = sorted(os.listdir(args.source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(args.source, slide))]
    slides = [slide for slide in slides if slide.split(".")[-1] in ["tiff", "ndpi", "svs"]]

    print(f'Creating -> {args.process} folder')
    if os.path.exists(args.process) and args.eliminate:
        shutil.rmtree(args.process)
    os.makedirs(args.process, exist_ok=True)
    print(f'Folder {args.process} created or already available!\n')

    if args.model == 'dinobloom':
        model = get_dino_bloom()
        trsforms = trasform_img()
    elif args.model == 'gigapath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        trsforms = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    else:
        model = resnet50_baseline(pretrained=True)
        trsforms = trasform_img()

    if torch.cuda.is_available():
        model.cuda()

    
    slides_to_skip = ['5308132_21-1000_5308132_21-1000_HE_2.tiff', '5216613_21-6946_5216613_21-6946_HE 2.tiff', '572711_YY-ART_5326968_21-5465_HE 2.tiff', '5193573_5193573_19-37477_19-37477_HE.ndpi']

    
    for i, slide in enumerate(slides):
        print("\n\nProgress: {:.2f}, {}/{}".format((i + 1) / len(slides), i + 1, len(slides)))
        file_path = os.path.join(args.source, slide)
        h5_path = os.path.join(args.h5, slide.split('.')[0] + '.h5')
        if slide in slides_to_skip:
            continue
        elif os.path.exists(os.path.join(args.features_dir, slide.split('.')[0] + '.pt')):
            continue

        wsi = openslide.open_slide(file_path)

        os.makedirs(os.path.join(args.process, slide.split('.')[0]), exist_ok=True)
        os.makedirs(os.path.join(args.features_dir, slide.split('.')[0]), exist_ok=True)

        create_features(args, h5_path, slide, wsi, model, trsforms)

