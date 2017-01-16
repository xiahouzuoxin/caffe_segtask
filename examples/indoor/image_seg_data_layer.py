
import sys, os
sys.path.append('../../python/')
import caffe
import numpy as np
from PIL import Image
import cv2
import random

class ImageSegDataLayer(caffe.Layer):
	"""
	Load Data for image segmentation task
	"""

	def setup(self, bottom, top):
		# config
		params = eval(self.param_str)
		self.root_dir = params['root_dir'] 
		self.src_list = params['src_list']
		self.split = params['split']
		self.mean = np.array(params['mean']) 
		self.random = params.get('randomize', False)
		self.seed = params.get('seed', None)
		self.crop = 473

        # two tops: data and label 
		if len(top) != 2: 
			raise Exception("Need to define two tops: data and label.") # data layers have no bottoms 
		if len(bottom) != 0: 
			raise Exception("Do not define a bottom.")
		
		# File Format: Deeplab list format
		self.indices = open(self.src_list, 'r').readlines()
		self.idx  = 0

        # make eval deterministic 
		if 'train' not in self.split: 
			self.random = False

        # randomization: seed and pick
		if self.random:
			random.seed(self.seed)
			self.idx = random.randint(0, len(self.indices)-1)

	def reshape(self, bottom, top):
		# load image + label image dir
		self.data, self.label = self.load_data(self.idx)
		# reshape tops to fit
		top[0].reshape(1, *self.data.shape)
		top[1].reshape(1, *self.label.shape)

	def forward(self, bottom, top):
		# assign output
		top[0].data[...] = self.data
		top[1].data[...] = self.label
		
		# pick next input
		if self.random:
			self.idx = random.randint(0, len(self.indices)-1)
		else:
			self.idx += 1
			if self.idx == len(self.indices):
				self.idx = 0

	def backward(self, top, propagate_down, bottom):
		pass

	def load_data(self, idx):
		"""
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
		"""
		im_file, gt_file = self.indices[idx].strip().split()
		im = cv2.imread(os.path.join(self.root_dir, im_file))
		gt = cv2.imread(os.path.join(self.root_dir, gt_file))

		im = np.array(im, dtype=np.float32)
		gt = np.array(gt, dtype=np.uint8)
		assert(im.shape[0]==gt.shape[0] and im.shape[1]==gt.shape[1])

		# crop
		if self.crop > 0:
			ori_shape = (gt.shape[0], gt.shape[1])
			min_edge = min(ori_shape[0], ori_shape[1])
			if min_edge < self.crop:
				scale = float(self.crop) / min_edge
				new_shape = tuple([int(np.ceil(scale * ori_shape[0])), int(np.ceil(scale * ori_shape[1]))])
				im = cv2.resize(im, new_shape, interpolation=cv2.INTER_LINEAR)
				gt = cv2.resize(gt, new_shape, interpolation=cv2.INTER_NEAREST)

			offset_h = np.random.randint(0, im.shape[0]-self.crop+1)
			offset_w = np.random.randint(0, im.shape[1]-self.crop+1)
			im = im[offset_h:offset_h+self.crop, offset_w:offset_w+self.crop, :]
			gt = gt[offset_h:offset_h+self.crop, offset_w:offset_w+self.crop, :]

		# 
		if (im.ndim == 2):
			im = np.repeat(im[:,:,None], 3, axis=2)
		im = im[:,:,::-1]
		im -= self.mean
		im = im.transpose((2,0,1))

		if (gt.ndim == 3):
			gt = gt[:,:,0]
		gt -= 1   # Round 0->255 & others-1
		gt = gt[np.newaxis,...]

		# print gt.shape, im.shape

		return im, gt

