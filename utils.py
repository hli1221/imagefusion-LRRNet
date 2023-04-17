# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: hui_li_jnu@163.com
# @File : utils.py
# @Time : 2020/7/11 15:59

import random
import numpy as np
import torch
from args import Args as args
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import join
import cv2

EPSILON = 1e-5


def list_images(directory):
	images = []
	names = []
	dir = listdir(directory)
	dir.sort()
	for file in dir:
		# name = file.lower()
		name = file
		if name.endswith('.png'):
			images.append(join(directory, file))
		elif name.endswith('.jpg'):
			images.append(join(directory, file))
		elif name.endswith('.jpeg'):
			images.append(join(directory, file))
		elif name.endswith('.bmp'):
			images.append(join(directory, file))
		elif name.endswith('.tif'):
			images.append(join(directory, file))
		name1 = name.split('.')
		names.append(name1[0])
	return images, names


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
	if num_imgs is None:
		num_imgs = len(image_path)
	original_imgs_path = image_path[:num_imgs]
	# random
	random.shuffle(original_imgs_path)
	mod = num_imgs % BATCH_SIZE
	print('BATCH SIZE %d.' % BATCH_SIZE)
	print('Train images number %d.' % num_imgs)
	print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

	if mod > 0:
		print('Train set has been trimmed %d samples...\n' % mod)
		original_imgs_path = original_imgs_path[:-mod]
	batches = int(len(original_imgs_path) // BATCH_SIZE)
	return original_imgs_path, batches


def save_mat(out, path):
	if args.cuda:
		out = out.cpu().data[0].numpy()
	else:
		out = out.data[0].numpy()
	out = np.squeeze(out)
	out = out.transpose((2, 1, 0))
	sio.savemat(path, {'img': out})


def get_image(path, height=256, width=256, flag=False):
	if flag is True:
		mode = cv2.IMREAD_COLOR
	else:
		mode = cv2.IMREAD_GRAYSCALE
	# image = Image.open(path).convert(mode)
	image = cv2.imread(path, mode)
	if height is not None and width is not None:
		# image = image.resize((height, width), Image.ANTIALIAS)
		# image = image.resize((height, width))
		image = cv2.resize(image,(height, width))
	return image


def get_train_images(paths, height=256, width=256, flag=False):
	if isinstance(paths, str):
		paths = [paths]
	images = []
	for path in paths:
		image = get_image(path, height, width, flag)
		if flag is True:
			image = np.transpose(image, (2, 0, 1))
		else:
			image = np.reshape(image, [1, image.shape[0], image.shape[1]])
		images.append(image)

	images = np.stack(images, axis=0)
	images = torch.from_numpy(images).float()
	return images


def save_image(img_fusion, output_path):
	img_fusion = img_fusion.float()
	if args.cuda:
		img_fusion = img_fusion.cpu().data[0].numpy()
	else:
		img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

	img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion) + EPSILON)
	img_fusion = img_fusion * 255
	img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
	if img_fusion.shape[2] == 1:
		img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
	cv2.imwrite(output_path, img_fusion)


def show_heatmap(feature, output_path):
	sns.set()
	feature = feature.float()
	if args.cuda:
		feature = feature.cpu().data[0].numpy()
	else:
		feature = feature.clamp(0, 255).data[0].numpy()

	feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + EPSILON)
	feature = feature * 255
	feature = feature.transpose(1, 2, 0).astype('uint8')
	if feature.shape[2] == 1:
		feature = feature.reshape([feature.shape[0], feature.shape[1]])

	fig = plt.figure()
	# sns.heatmap(feature, cmap='YlGnBu', xticklabels=50, yticklabels=50)
	sns.heatmap(feature, xticklabels=50, yticklabels=50)
	fig.savefig(output_path, bbox_inches='tight')
	# plt.show()


def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram


def normalize_tensor(tensor):
	(b, ch, h, w) = tensor.size()

	tensor_v = tensor.view(b, -1)
	t_min = torch.min(tensor_v, 1)[0]
	t_max = torch.max(tensor_v, 1)[0]

	t_min = t_min.view(b, 1, 1, 1)
	t_min = t_min.repeat(1, ch, h, w)
	t_max = t_max.view(b, 1, 1, 1)
	t_max = t_max.repeat(1, ch, h, w)
	tensor = (tensor - t_min) / (t_max - t_min + EPSILON)
	return tensor


# initial VGG16 network
def init_vgg16(vgg, model_dir):
	vgg_load = torch.load(model_dir)
	count = 0
	for name, param in vgg_load.items():
		if count >= 20:
			break
		if count == 0:
			vgg.conv1_1.weight.data = param
		if count == 1:
			vgg.conv1_1.bias.data = param
		if count == 2:
			vgg.conv1_2.weight.data = param
		if count == 3:
			vgg.conv1_2.bias.data = param

		if count == 4:
			vgg.conv2_1.weight.data = param
		if count == 5:
			vgg.conv2_1.bias.data = param
		if count == 6:
			vgg.conv2_2.weight.data = param
		if count == 7:
			vgg.conv2_2.bias.data = param

		if count == 8:
			vgg.conv3_1.weight.data = param
		if count == 9:
			vgg.conv3_1.bias.data = param
		if count == 10:
			vgg.conv3_2.weight.data = param
		if count == 11:
			vgg.conv3_2.bias.data = param
		if count == 12:
			vgg.conv3_3.weight.data = param
		if count == 13:
			vgg.conv3_3.bias.data = param

		if count == 14:
			vgg.conv4_1.weight.data = param
		if count == 15:
			vgg.conv4_1.bias.data = param
		if count == 16:
			vgg.conv4_2.weight.data = param
		if count == 17:
			vgg.conv4_2.bias.data = param
		if count == 18:
			vgg.conv4_3.weight.data = param
		if count == 19:
			vgg.conv4_3.bias.data = param
		count = count + 1