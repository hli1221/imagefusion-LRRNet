# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: hui_li_jnu@163.com
# @File : testing_fusion_lrr.py
# @Time : 2020/6/30 15:00

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from torch.autograd import Variable
from net_lista import LRR_NET
from args import Args as args
import utils


def load_model(path, num_block, fusion_type='cat'):
	model_or = LRR_NET(args.s, args.n, args.channel, args.stride, num_block, fusion_type)
	# Parallel for block >= 4
	if num_block <= 4:
		model = model_or
	else:
		model = torch.nn.DataParallel(model_or, list(range(torch.cuda.device_count())))
	model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
	
	total = sum([param.nelement() for param in model.parameters()])
	print('Number	of	parameter: {:4f}M'.format(total / 1e6))
	
	model.eval()
	model.cuda()

	return model


def run(model, infrared_path, visible_path, output_path, img_flag, img_name, lam2_str, wir_str):
	"""
	run the LRR-RFN-NET
	:param model: network model
	:param infrared_path:  nfrared image path
	:param visible_path: visible image path
	:param output_path: output path
	:param output_path_fea: feature output path
	:param img_flag: gray or RGB
	:param img_name: tag of output image
	"""
	img_ir = utils.get_train_images(infrared_path, height=None, width=None, flag=img_flag)
	img_vi = utils.get_train_images(visible_path, height=None, width=None, flag=img_flag)

	img_ir = Variable(img_ir, requires_grad=False)
	img_vi = Variable(img_vi, requires_grad=False)
	if args.cuda:
		img_ir = img_ir.cuda()
		img_vi = img_vi.cuda()

	img_ir = utils.normalize_tensor(img_ir)
	img_vi = utils.normalize_tensor(img_vi)
	output = model(img_ir, img_vi)

	out = output['fuse']
	
	path_out = output_path + 'result_lrrnet_' + img_name + '.png'
	utils.save_image(out, path_out)

	print(img_name + '_lam2_' + lam2_str + '_wir_' + wir_str +' Done......')


def main():
	# True - RGB, False - gray
	if args.channel == 1:
		img_flag = False
	else:
		img_flag = True

	# test_path = "images/21_pairs_tno/ir/"
	test_path = "images/40_pairs_tno_vot_new/ir/"
	imgs_paths_ir, names = utils.list_images(test_path)
	num = len(imgs_paths_ir)

	# lam2_list = ['0.1', '0.5', '1.0', '1.5', '2.0', '2.5']  # '1.5',
	# wir_list = ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0']
	# lam3_gram_list = ['0', '100', '500', '1000', '1500', '2000', '2500']
	num_block = 4

	wvi_str = '0.5'
	lam2_str = '1.5'
	wir_str = '3.0'
	lam3_gram = '2000'

	model_path = './model/final_lrr_net_lam2_' + lam2_str + '_wir_' + wir_str + \
					'_lam3_gram_' + lam3_gram + '_epoch_4_block_4.model'

	output_path_root = 'outputs/40_pairs_tno_vot_new/'  # 21_pairs_tno_new, 40_pairs_tno_vot_new

	if os.path.exists(output_path_root) is False:
		os.mkdir(output_path_root)

	output_path = output_path_root + '/Fused_21_pairs_lam2_' + lam2_str + '_wvi_' + wvi_str + '_wir_' + wir_str + \
					'_lam3_gram_' + lam3_gram + '_blocks_' + str(num_block) + '/'
	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	with torch.no_grad():
		model = load_model(model_path, num_block)
		for i in range(num):
			img_name = names[i]
			infrared_path = imgs_paths_ir[i]
			visible_path = infrared_path.replace('ir/', 'vis/')
			if visible_path.__contains__('IR'):
				visible_path = visible_path.replace('IR', 'VIS')
			else:
				visible_path = visible_path.replace('i.', 'v.')
			run(model, infrared_path, visible_path, output_path, img_flag, img_name, lam2_str, wir_str)


if __name__ == '__main__':
	main()
