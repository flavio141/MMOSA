import numpy as np

import argparse
import sys
import timm
sys.path.append('utils')
sys.path.append('models')

import torch
import pdb
import os
import pandas as pd
from models import ModelSurv


import h5py
import yaml
from heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from file_utils import save_hdf5, sample_rois, initialize_df, initiate_model

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default='best_trial', help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="config_template.yaml")
parser.add_argument('--features', type=str, default="features_mgg")
parser.add_argument('--h5', type=str, default="result_dir_mgg/patches")

args_begin = parser.parse_args()


os.environ['HF_TOKEN'] = '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'


def infer_single_slide(model, features, k=1):
	model.to(device)
	features = features.to(device)
	with torch.no_grad():
		if isinstance(model, (ModelSurv)):
			risk, A = model(features, torch.tensor(features.shape[0]), explainability=True)
			#A = A.view(-1, 1).cpu().numpy()
			A = A.cpu().numpy()

		else:
			raise NotImplementedError
		
		probs, ids = torch.topk(risk, k, dim=0)
		probs = probs[-1].cpu().numpy()
		ids = ids[-1].cpu().numpy()

	return ids, probs, A

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def parse_config_dict(args, config_dict):
	if args.save_exp_code is not None:
		config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	return config_dict

if __name__ == '__main__':
	config_path = os.path.join('explainability', args_begin.config_file)
	config_dict = yaml.safe_load(open(config_path, 'r'))
	config_dict = parse_config_dict(args_begin, config_dict)

	for key, value in config_dict.items():
		if isinstance(value, dict):
			print('\n'+key)
			for value_key, value_value in value.items():
				print (value_key + " : " + str(value_value))
		else:
			print ('\n'+key + " : " + str(value))

	args = config_dict
	patch_args = argparse.Namespace(**args['patching_arguments'])
	data_args = argparse.Namespace(**args['data_arguments'])
	model_args = args['model_arguments']
	model_args.update({'n_classes': args['exp_arguments']['n_classes'], 'input_size': args['exp_arguments']['input_size']})
	model_args = argparse.Namespace(**model_args)
	exp_args = argparse.Namespace(**args['exp_arguments'])
	heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
	sample_args = argparse.Namespace(**args['sample_arguments'])

	patch_size = tuple([patch_args.patch_size for i in range(2)])
	step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

	
	preset = data_args.preset
	def_seg_params = {'seg_level': -1, 'mthresh': 7, 'kernel_size': 2, 'use_otsu': True, 
					  'keep_ids': 'none', 'exclude_ids':'none'}
	def_filter_params = {'a_t':1.0, 'a_h': 1.0, 'max_n_holes':10}
	def_vis_params = {'vis_level': -1, 'line_thickness': 50}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if preset is not None:
		preset_df = pd.read_csv(preset)
		for key in def_seg_params.keys():
			def_seg_params[key] = preset_df.loc[0, key]

		for key in def_filter_params.keys():
			def_filter_params[key] = preset_df.loc[0, key]

		for key in def_vis_params.keys():
			def_vis_params[key] = preset_df.loc[0, key]

		for key in def_patch_params.keys():
			def_patch_params[key] = preset_df.loc[0, key]


	if data_args.process_list is None:
		if isinstance(data_args.data_dir, list):
			slides = []
			for data_dir in data_args.data_dir:
				slides.extend(os.listdir(data_dir))
		else:
			slides = sorted(os.listdir(data_args.data_dir))
		slides = [slide for slide in slides if slide.endswith(eval(data_args.slide_ext))]
		df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params)
		
	else:
		df = pd.read_csv(os.path.join('explainability/process_lists', data_args.process_list))
		df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params)

	mask = df['process'] == 1
	process_stack = df[mask].reset_index(drop=True)
	total = len(process_stack)
	print('\nlist of slides to process: ')
	print(process_stack.head(len(process_stack)))

	print('\ninitializing model from checkpoint')
	ckpt_path = model_args.ckpt_path
	print('\nckpt path: {}'.format(ckpt_path))
	
	if model_args.initiate_fn == 'initiate_model':
		model =  initiate_model(model_args, ckpt_path)
	else:
		raise NotImplementedError


	# feature_extractor = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
	# feature_extractor.eval()
	device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Done!')

	reverse_label_dict =  data_args.label_dict
	#feature_extractor = feature_extractor.to(device)

	os.makedirs(exp_args.production_save_dir, exist_ok=True)
	os.makedirs(exp_args.raw_save_dir, exist_ok=True)
	blocky_wsi_kwargs = {
		'top_left': None, 
		'bot_right': None, 
		'patch_size': patch_size, 
		'step_size': patch_size, 
		'custom_downsample':patch_args.custom_downsample, 
		'level': patch_args.patch_level, 
		'use_center_shift': heatmap_args.use_center_shift
	}


	for i in range(len(process_stack)):
		slide_name = process_stack.loc[i, 'slide_id']
		if np.isnan(process_stack.loc[i, 'time_label']):
			continue
		print('\nprocessing: ', slide_name)	

		try:
			label = process_stack.loc[i, 'time_label']
		except KeyError:
			label = 'Unspecified'

		slide_id = slide_name.split('.')[0]

		if not isinstance(label, str):
			grouping = reverse_label_dict[label]
		else:
			grouping = label

		p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
		os.makedirs(p_slide_save_dir, exist_ok=True)

		r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
		os.makedirs(r_slide_save_dir, exist_ok=True)

		if heatmap_args.use_roi:
			x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
			y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
			top_left = (int(x1), int(y1))
			bot_right = (int(x2), int(y2))
		else:
			top_left = None
			bot_right = None
		
		print('slide id: ', slide_id)
		print('top left: ', top_left, ' bot right: ', bot_right)

		if isinstance(data_args.data_dir, str):
			slide_path = os.path.join(data_args.data_dir, slide_name)
		elif isinstance(data_args.data_dir, dict):
			data_dir_key = process_stack.loc[i, data_args.data_dir_key]
			slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
		else:
			raise NotImplementedError

		mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
		
		# Load segmentation and filter parameters
		seg_params = def_seg_params.copy()
		filter_params = def_filter_params.copy()
		vis_params = def_vis_params.copy()

		seg_params = load_params(process_stack.loc[i], seg_params)
		filter_params = load_params(process_stack.loc[i], filter_params)
		vis_params = load_params(process_stack.loc[i], vis_params)

		keep_ids = str(seg_params['keep_ids'])
		if len(keep_ids) > 0 and keep_ids != 'none':
			seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
		else:
			seg_params['keep_ids'] = []

		exclude_ids = str(seg_params['exclude_ids'])
		if len(exclude_ids) > 0 and exclude_ids != 'none':
			seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
		else:
			seg_params['exclude_ids'] = []

		for key, val in seg_params.items():
			print('{}: {}'.format(key, val))

		for key, val in filter_params.items():
			print('{}: {}'.format(key, val))

		for key, val in vis_params.items():
			print('{}: {}'.format(key, val))
		
		print('Initializing WSI object')
		wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
		print('Done!')

		wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

		# the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
		vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

		block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
		mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
		if vis_params['vis_level'] < 0:
			downscale = 64
			best_level = wsi_object.wsi.get_best_level_for_downsample(downscale)
			while wsi_object.wsi.level_dimensions[best_level][0] < 1900:
				downscale = downscale / 2
				best_level = wsi_object.wsi.get_best_level_for_downsample(downscale)
			
			vis_params['vis_level'] = best_level
		mask = wsi_object.visWSI(**vis_params, number_contours=True)
		mask.save(mask_path)
		
		features_path = os.path.join(args_begin.features, slide_id+'.pt')
		h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')

		h5_path_coords = os.path.join(args_begin.h5, slide_id+'.h5')
	

		##### check if h5_features_file exists ######
		if not os.path.isfile(h5_path) :
			_, _, wsi_object = compute_from_patches(wsi_object=wsi_object,
										   features_path=features_path,
										   model=model, 
										   attn_save_path=None, 
										   feat_save_path=h5_path, 
										   ref_scores=None)				
		
		# ##### check if pt_features_file exists ######
		# if not os.path.isfile(features_path):
		# 	file = h5py.File(h5_path, "r")
		# 	features = torch.tensor(file['features'][:])
		# 	torch.save(features, features_path)
		# 	file.close()

		# load features 
		if not os.path.exists(features_path):
			continue
		features = torch.load(features_path)
		process_stack.loc[i, 'bag_size'] = len(features)
		
		wsi_object.saveSegmentation(mask_file)
		Y_hats, Y_probs, A = infer_single_slide(model, features, exp_args.n_classes)
		del features
		
		#if not os.path.isfile(block_map_save_path):
		if not os.path.isfile(h5_path_coords):
			continue
		file = h5py.File(h5_path_coords, "r")
		coords = file['coords'][:]
		file.close()
		asset_dict = {'attention_scores': A, 'coords': coords}
		block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
		
		# save top 3 predictions
		process_stack.loc[i, 'risk'] = Y_probs

		os.makedirs('heatmaps/results/', exist_ok=True)
		if data_args.process_list is not None:
			process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
		else:
			process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)
		
		file = h5py.File(block_map_save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		samples = sample_args.samples
		for sample in samples:
			if sample['sample']:
				sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', sample['name'])
				os.makedirs(sample_save_dir, exist_ok=True)
				print('sampling {}'.format(sample['name']))
				sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
				for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
					patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
					patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a.png'.format(idx, slide_id, s_coord[0], s_coord[1])))

		wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
		'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

		heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
		if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
			pass
		else:
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
							thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
		
			heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
			del heatmap

		save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

		if heatmap_args.use_ref_scores:
			ref_scores = scores
		else:
			ref_scores = None
		
		if heatmap_args.calc_heatmap:
			_, _, wsi_object = compute_from_patches(wsi_object=wsi_object,
										   features_path=features_path,
										   model=model, 
										   attn_save_path=save_path, 
										   feat_save_path=h5_path, 
										   ref_scores=ref_scores)	

		if not os.path.isfile(save_path):
			print('heatmap {} not found'.format(save_path))
			if heatmap_args.use_roi:
				save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
				print('found heatmap for whole slide')
				save_path = save_path_full
			else:
				continue

		file = h5py.File(save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
		if heatmap_args.use_ref_scores:
			heatmap_vis_args['convert_to_percentiles'] = False

		heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
																						int(heatmap_args.blur), 
																						int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
																						float(heatmap_args.alpha), int(heatmap_args.vis_level), 
																						int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


		if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
			pass
		
		else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
						          cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
						          binarize=heatmap_args.binarize, 
						  		  blank_canvas=heatmap_args.blank_canvas,
						  		  thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
						  		  overlap=patch_args.overlap, 
						  		  top_left=top_left, bot_right = bot_right)
			if heatmap_args.save_ext == 'jpg':
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
			else:
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
		
		if heatmap_args.save_orig:
			if heatmap_args.vis_level >= 0:
				vis_level = heatmap_args.vis_level
			else:
				vis_level = vis_params['vis_level']
			heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
			if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
				pass
			else:
				heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
				if heatmap_args.save_ext == 'jpg':
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
				else:
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

	with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
		yaml.dump(config_dict, outfile, default_flow_style=False)


