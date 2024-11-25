import subprocess
import math
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage.measure.simple_metrics import compare_psnr

def weights_init_kaiming(lyr):

	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm') != -1:
		lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
			clamp_(-0.025, 0.025)
		nn.init.constant(lyr.bias.data, 0.0)

def batch_psnr(img, imclean, data_range):

	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
					   data_range=data_range)
	return psnr/img_cpu.shape[0]

def data_augmentation(image, mode):

	out = np.transpose(image, (1, 2, 0))
	if mode == 0:
		# original
		out = out
	elif mode == 1:
		# flip up and down
		out = np.flipud(out)
	elif mode == 2:
		# rotate counterwise 90 degree
		out = np.rot90(out)
	elif mode == 3:
		# rotate 90 degree and flip up and down
		out = np.rot90(out)
		out = np.flipud(out)
	elif mode == 4:
		# rotate 180 degree
		out = np.rot90(out, k=2)
	elif mode == 5:
		# rotate 180 degree and flip
		out = np.rot90(out, k=2)
		out = np.flipud(out)
	elif mode == 6:
		# rotate 270 degree
		out = np.rot90(out, k=3)
	elif mode == 7:
		# rotate 270 degree and flip
		out = np.rot90(out, k=3)
		out = np.flipud(out)
	else:
		raise Exception('Invalid choice of image transformation')
	return np.transpose(out, (2, 0, 1))

def variable_to_cv2_image(varim):

	nchannels = varim.size()[1]
	if nchannels == 1:
		res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
	elif nchannels == 3:
		res = varim.data.cpu().numpy()[0]
		res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
		res = (res*255.).clip(0, 255).astype(np.uint8)
	else:
		raise Exception('Number of color channels not supported')
	return res

def get_git_revision_short_hash():

	return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def init_logger(argdict):

	from os.path import join

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(argdict.log_dir, 'log.txt'), mode='a')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	try:
		logger.info("Commit: {}".format(get_git_revision_short_hash()))
	except Exception as e:
		logger.error("Couldn't get commit number: {}".format(e))
	logger.info("Arguments: ")
	for k in argdict.__dict__:
		logger.info("\t{}: {}".format(k, argdict.__dict__[k]))

	return logger

def init_logger_ipol():

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler('out.txt', mode='w')
	formatter = logging.Formatter('%(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def init_logger_test(result_dir):

	from os.path import join

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='a')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def normalize(data):

	return np.float32(data/255.)

def svd_orthogonalization(lyr):

	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		weights = lyr.weight.data.clone()
		c_out, c_in, f1, f2 = weights.size()
		dtype = lyr.weight.data.type()

		# Reshape filters to columns
		# From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
		weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)

		# Convert filter matrix to numpy array
		weights = weights.cpu().numpy()

		# SVD decomposition and orthogonalization
		mat_u, _, mat_vh = np.linalg.svd(weights, full_matrices=False)
		weights = np.dot(mat_u, mat_vh)

		# As full_matrices=False we don't need to set s[:] = 1 and do mat_u*s
		lyr.weight.data = torch.Tensor(weights).view(f1, f2, c_in, c_out).\
			permute(3, 2, 0, 1).type(dtype)
	else:
		pass

def remove_dataparallel_wrapper(state_dict):

	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict

def is_rgb(im_path):

	from skimage.io import imread
	rgb = False
	im = imread(im_path)
	if (len(im.shape) == 3):
		if not(np.allclose(im[...,0], im[...,1]) and np.allclose(im[...,2], im[...,1])):
			rgb = True
	print("rgb: {}".format(rgb))
	print("im shape: {}".format(im.shape))
	return rgb
