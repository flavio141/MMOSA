import os
import cv2
import math
import h5py
import openslide
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append('utils')

from PIL import Image
from file_utils import load_pkl, save_hdf5, save_pkl, to_percentiles, screen_coords


class WholeSlideImage(object):
    def __init__(self, path):
        '''
        The class for the Whole Slide Images. It has all the important
        function to modify or access to the WSI.

        Parameters
        ----------
        path : str
            Path to the image.
        '''

        self.wsi = openslide.open_slide(path)
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.level_dim = self.wsi.level_dimensions
        self.level_downsamples = self._assertLevelDownsamples()
        self.hdf5_file = None

    def getOpenSlide(self):
        return self.wsi
    
    def initSegmentation(self, mask_file):
        asset_dict = load_pkl(mask_file)
        self.holes_tissue = asset_dict['holes']
        self.contours_tissue = asset_dict['tissue']
    
    def saveSegmentation(self, mask_file):
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        save_pkl(mask_file, asset_dict)

    def segmentTissue(self, 
                      seg_level=0,
                      mthresh=7, 
                      kernel_size = 0, 
                      use_otsu=True, 
                      filter_params={'a_t':100}, 
                      ref_patch_size=512, 
                      exclude_ids=[], 
                      keep_ids=[]):
        
        def _morphological_closing(image, kernel_size=5):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            return closed

        def _foreground(contours, hierarchy, filter_params):
            filtered = []

            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
            all_holes = []
            
            # loop through foreground contour indices
            for cont_idx in hierarchy_1:
                cont = contours[cont_idx]
                
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                a = cv2.contourArea(cont)
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                
                # actual area of foreground contour region
                a = a - np.array(hole_areas).sum()
                if a == 0: 
                    continue
                
                if tuple((filter_params['a_t'],)) < tuple((a,)): 
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []
                
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        def _otsu_threshold(image):
            _, thresholded = cv2.threshold(image, 0, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresholded
        
        img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        image = cv2.medianBlur(img_hsv[:,:,1], mthresh)

        # Apply Otsu's thresholding if required
        if use_otsu:
            image = _otsu_threshold(image)

        # Apply morphological closing if required
        if kernel_size > 0:
            image = _morphological_closing(image, kernel_size)

        # Filter contours
        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

        contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        foreground_contours, hole_contours = _foreground(contours, hierarchy, filter_params)                 

        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

        contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)
        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]


    def visWSI(self, vis_level=0, color = (0,255,0), annot_color=(255,0,0), 
                        line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1, view_slide_only=False,
                        number_contours=False, seg_display=True, annot_display=True):
        
        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]]
        
        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0,0)
            region_size = self.level_dim[vis_level]

        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        
        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale), 
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

                else: # add numbering to each contour
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # draw the contour and put text next to center
                        cv2.drawContours(img,  [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, "{}".format(idx), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)
        
        img = Image.fromarray(img)
    
        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img
    
    def process_contours(self, save_path, patch_level=0, patch_size=256, step_size=256, **kwargs):
        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        print("Creating patches for: ", self.name, "...",)
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True
        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))
            
            asset_dict, attr_dict = self.process_contour(cont, self.holes_tissue[idx], patch_level, save_path, patch_size, step_size, **kwargs)
            if len(asset_dict) > 0:
                if init:
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode='a')

        return self.hdf5_file


    def process_contour(self, cont, contour_holes, patch_level, save_path, patch_size = 256, step_size = 256,
        contour_fn='four_pt', use_padding=True, top_left=None, bot_right=None):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        
        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1]+1)
            stop_x = min(start_x+w, img_w-ref_patch_size[0]+1)
        
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                print("Adjusted Bounding Box:", start_x, start_y, w, h)
    
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        num_workers = mp.cpu_count()
        if num_workers > 4:
            num_workers = 4
        pool = mp.Pool(num_workers)

        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        results = np.array([result for result in results if result is not None])
        
        print('Extracted {} coordinates'.format(len(results)))

        if len(results)>1:
            asset_dict = {'coords' :          results}
            
            attr = {'patch_size' :            patch_size,
                    'patch_level' :           patch_level,
                    'downsample':             self.level_downsamples[patch_level],
                    'downsampled_level_dim' : tuple(np.array(self.level_dim[patch_level])),
                    'level_dim':              self.level_dim[patch_level],
                    'name':                   self.name,
                    'save_path':              save_path}

            attr_dict = { 'coords' : attr}
            return asset_dict, attr_dict

        else:
            return {}, {}
        
    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]
        
        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
        
        return level_downsamples

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1
        
        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]
    

    def get_seg_mask(self, region_size, scale, use_holes=False, offset=(0,0)):
        print('\ncomputing foreground tissue mask')
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
        contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
        offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

        contours_holes = self.scaleHolesDim(self.holes_tissue, scale)
        contours_tissue, contours_holes = zip(*sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
        for idx in range(len(contours_tissue)):
            cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1)

            if use_holes:
                cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1)
            # contours_holes = self._scaleContourDim(self.holes_tissue, scale, holes=True, area_thresh=area_thresh)
                
        tissue_mask = tissue_mask.astype(bool)
        print('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
        return tissue_mask
    

    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=224):
        print('\ncomputing blend')
        downsample = self.level_downsamples[vis_level]
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        print('using block size: {} x {}'.format(block_size_x, block_size_y))

        shift = top_left # amount shifted w.r.t. (0,0)
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
                #print(x_start, y_start)

                # 1. convert wsi coordinates to image coordinates via shift and scale
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))
                
                # 2. compute end points of blend tile, careful not to go over the edge of the image
                y_end_img = min(h, y_start_img+block_size_y)
                x_end_img = min(w, x_start_img+block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue
                #print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))
                
                # 3. fetch blend block and size
                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img] 
                blend_block_size = (x_end_img-x_start_img, y_end_img-y_start_img)
                
                if not blank_canvas:
                    # 4. read actual wsi block as canvas block
                    pt = (x_start, y_start)
                    canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))     
                else:
                    # 4. OR create blank canvas block
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255,255,255)))

                # 5. blend color block and canvas block
                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0, canvas)
        return img
    

    def visHeatmap(self, scores, coords, vis_level=-1, 
                   top_left=None, bot_right=None,
                   patch_size=(224, 224), 
                   blank_canvas=False, 
                   canvas_color=(220, 20, 50), 
                   alpha=0.4, 
                   blur=False, overlap=0.0, 
                   segment=True, use_holes=True,
                   convert_to_percentiles=False, 
                   binarize=False, thresh=0.5,
                   max_size=None,
                   custom_downsample = 1,
                   cmap='coolwarm'):

        if vis_level < 0:
            downscale = 64
            vis_level = self.wsi.get_best_level_for_downsample(downscale)
            while self.wsi.level_dimensions[vis_level][0] < 1900:
                downscale = downscale / 2
                vis_level = self.wsi.get_best_level_for_downsample(downscale)

        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]] # Scaling from 0 to desired level
                
        if len(scores.shape) == 2:
            scores = scores.flatten()

        if binarize:
            if thresh < 0:
                threshold = 1.0/len(scores)
                
            else:
                threshold =  thresh
        
        else:
            threshold = 0.0

        ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
        if top_left is not None and bot_right is not None:
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)

        else:
            region_size = self.level_dim[vis_level]
            top_left = (0,0)
            bot_right = self.level_dim[0]
            w, h = region_size

        patch_size  = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)
        
        print('\ncreating heatmap for: ')
        print('top_left: ', top_left, 'bot_right: ', bot_right)
        print('w: {}, h: {}'.format(w, h))
        print('scaled patch size: ', patch_size)

        ###### normalize filtered scores ######
        if convert_to_percentiles:
            scores = to_percentiles(scores) 

        scores /= 100
        
        ######## calculate the heatmap of raw attention scores (before colormap) 
        # by accumulating scores over overlapped regions ######
        
        # heatmap overlay: tracks attention score over each pixel of heatmap
        # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
        overlay = np.full(np.flip(region_size), 0).astype(float)
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)      
        count = 0
        for idx in range(len(coords)):
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                if binarize:
                    score=1.0
                    count+=1
            else:
                score=0.0
            # accumulate attention
            overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
            # accumulate counter
            counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1

        if binarize:
            print('\nbinarized tiles based on cutoff of {}'.format(threshold))
            print('identified {}/{} patches as positive'.format(count, len(coords)))
        
        # fetch attended region and average accumulated attention
        zero_mask = counter == 0

        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter 
        if blur:
            overlay = cv2.GaussianBlur(overlay,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

        if segment:
            tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))
            # return Image.fromarray(tissue_mask) # tissue mask
        
        if not blank_canvas:
            # downsample original image and use as canvas
            img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        else:
            # use blank canvas
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255,255,255))) 

        #return Image.fromarray(img) #raw image

        print('\ncomputing heatmap image')
        print('total of {} patches'.format(len(coords)))
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        
        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                print('progress: {}/{}'.format(idx, len(coords)))
            
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:

                # attention block
                raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
                
                # image block (either blank canvas or orig image)
                img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()

                # color block (cmap applied to attention block)
                color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

                if segment:
                    # tissue mask block
                    mask_block = tissue_mask[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] 
                    # copy over only tissue masked portion of color block
                    img_block[mask_block] = color_block[mask_block]
                else:
                    # copy over entire color block
                    img_block = color_block

                # rewrite image block
                img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()
        
        #return Image.fromarray(img) #overlay
        print('Done')
        del overlay

        if blur:
            img = cv2.GaussianBlur(img,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

        if alpha < 1.0:
            img = self.block_blending(img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=1536)
        
        img = Image.fromarray(img)
        w, h = img.size

        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img



class Contour_Checking_fn(object):
	# Defining __call__ method 
	def __call__(self, pt): 
		raise NotImplementedError

class isInContourV1(Contour_Checking_fn):
	def __init__(self, contour):
		self.cont = contour

	def __call__(self, pt): 
		return 1 if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False) >= 0 else 0

class isInContourV2(Contour_Checking_fn):
	def __init__(self, contour, patch_size):
		self.cont = contour
		self.patch_size = patch_size

	def __call__(self, pt): 
		pt = np.array((pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)).astype(float)
		return 1 if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False) >= 0 else 0

# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class isInContourV3_Easy(Contour_Checking_fn):
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, tuple(np.array(points).astype(float)), False) >= 0:
				return 1
		return 0

class isInContourV3_Hard(Contour_Checking_fn):
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, tuple(np.array(points).astype(float)), False) < 0:
				return 0
		return 1


def DrawGrid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord - thickness // 2)),
                  tuple(coord - thickness // 2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img


def DrawMapFromCoords(canvas, wsi_object, coords, patch_size, vis_level, indices=None, verbose=1, draw_grid=True):
    downsamples = wsi_object.wsi.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)

    patch_size = tuple(np.ceil((np.array(patch_size) / np.array(downsamples))).astype(np.int32))
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))

    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))

        patch_id = indices[idx]
        coord = coords[patch_id]
        patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = patch[:canvas_crop_shape[0],
                                                                                           :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)


def StitchCoords(hdf5_file_path, wsi_object, downscale=64, draw_grid=False, bg_color=(0, 0, 0), alpha=-1):
    wsi = wsi_object.getOpenSlide()
    vis_level = wsi.get_best_level_for_downsample(downscale)
    while wsi.level_dimensions[vis_level][0] < 1900:
        downscale = downscale / 2
        vis_level = wsi.get_best_level_for_downsample(downscale)
    
    file = h5py.File(hdf5_file_path, 'r')
    dset = file['coords']
    coords = dset[:]
    w, h = wsi.level_dimensions[0]

    print('start stitching {}'.format(dset.attrs['name']))
    print('original size: {} x {}'.format(w, h))

    w, h = wsi.level_dimensions[vis_level]

    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(coords)))

    patch_size = dset.attrs['patch_size']
    patch_level = dset.attrs['patch_level']
    print('patch size: {}x{} patch level: {}'.format(patch_size, patch_size, patch_level))
    patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
    print('ref patch size: {}x{}'.format(patch_size, patch_size))

    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)

    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w, h), mode="RGBA", color=bg_color + (int(255 * alpha),))

    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(heatmap, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=draw_grid)

    file.close()
    return heatmap


def default_transforms(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    t = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean = mean, std = std)])
    return t

def get_contour_check_fn(contour_fn='four_pt_hard', cont=None, ref_patch_size=None, center_shift=None):
    if contour_fn == 'four_pt_hard':
        cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size, center_shift=center_shift)
    elif contour_fn == 'four_pt_easy':
        cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size, center_shift=0.5)
    elif contour_fn == 'center':
        cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size)
    elif contour_fn == 'basic':
        cont_check_fn = isInContourV1(contour=cont)
    else:
        raise NotImplementedError
    return cont_check_fn


class Wsi_Region(Dataset):
    def __init__(self, wsi_object, top_left=None, bot_right=None, level=0, 
                 patch_size = (256, 256), step_size=(256, 256), 
                 contour_fn='four_pt_hard',
                 t=None, custom_downsample=1, use_center_shift=False):
        
        self.custom_downsample = custom_downsample

        # downscale factor in reference to level 0
        self.ref_downsample = wsi_object.level_downsamples[level]
        # patch size in reference to level 0
        self.ref_size = tuple((np.array(patch_size) * np.array(self.ref_downsample)).astype(int)) 
        
        if self.custom_downsample > 1:
            self.target_patch_size = patch_size
            patch_size = tuple((np.array(patch_size) * np.array(self.ref_downsample) * custom_downsample).astype(int))
            step_size = tuple((np.array(step_size) * custom_downsample).astype(int))
            self.ref_size = patch_size
        else:
            step_size = tuple((np.array(step_size)).astype(int))
            self.ref_size = tuple((np.array(patch_size) * np.array(self.ref_downsample)).astype(int)) 
        
        self.wsi = wsi_object.wsi
        self.level = level
        self.patch_size = patch_size
            
        if not use_center_shift:
            center_shift = 0.
        else:
            overlap = 1 - float(step_size[0] / patch_size[0])
            if overlap < 0.25:
                center_shift = 0.375
            elif overlap >= 0.25 and overlap < 0.75:
                center_shift = 0.5
            elif overlap >=0.75 and overlap < 0.95:
                center_shift = 0.5
            else:
                center_shift = 0.625
        
        filtered_coords = []
        for cont_idx, contour in enumerate(wsi_object.contours_tissue): 
            print('processing {}/{} contours'.format(cont_idx, len(wsi_object.contours_tissue)))
            cont_check_fn = get_contour_check_fn(contour_fn, contour, self.ref_size[0], center_shift)
            coord_results, _ = wsi_object.process_contour(contour, wsi_object.holes_tissue[cont_idx], level, '', 
                            patch_size = patch_size[0], step_size = step_size[0], contour_fn=cont_check_fn,
                            use_padding=True, top_left = top_left, bot_right = bot_right)
            if len(coord_results) > 0:
                filtered_coords.append(coord_results['coords'])
        
        coords=np.vstack(filtered_coords)

        self.coords = coords
        print('filtered a total of {} coordinates'.format(len(self.coords)))
        
        if t is None:
            self.transforms = default_transforms()
        else:
            self.transforms = t

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        coord = self.coords[idx]
        patch = self.wsi.read_region(tuple(coord), self.level, self.patch_size).convert('RGB')
        if self.custom_downsample > 1:
            patch = patch.resize(self.target_patch_size)
        patch = self.transforms(patch).unsqueeze(0)
        return patch, coord 