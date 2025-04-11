import torch
import os
import h5py

import sys
sys.path.append('utils')
sys.path.append('wsi_utils')

from WSI import WholeSlideImage, Wsi_Region
from scipy.stats import percentileofscore
import math

from tqdm import tqdm
from file_utils import save_hdf5
from scipy.stats import percentileofscore

from torch.utils.data import DataLoader, sampler

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    percentile = percentileofscore(ref.reshape(-1,), score[0])
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        downscale = 64
        vis_level = wsi.get_best_level_for_downsample(downscale)
        while wsi.level_dimensions[vis_level][0] < 1900:
            downscale = downscale / 2
            vis_level = wsi.get_best_level_for_downsample(downscale)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap


def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        downscale = 64
        best_level = wsi_object.wsi.get_best_level_for_downsample(downscale)
        while wsi_object.wsi.level_dimensions[best_level][0] < 1900:
            downscale = downscale / 2
            best_level = wsi_object.wsi.get_best_level_for_downsample(downscale)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object


def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 


def compute_from_patches(wsi_object, features_path=None, clam_pred=None, model=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None):    
    
    roi_dataset = Wsi_Region(wsi_object)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', num_batches)
    mode = "w"
    for idx, (roi, coords) in enumerate(tqdm(roi_loader)):
        roi = roi.to(device)
        coords = coords.numpy()
        
        with torch.inference_mode():
            features = torch.load(features_path)

            if attn_save_path is not None:
                risk, A = model(features, torch.tensor(features.shape[0]).to(device), explainability=True)
           
                # if A.size(0) > 1:
                #     A = A[clam_pred]

                A = A.cpu().numpy()

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object
