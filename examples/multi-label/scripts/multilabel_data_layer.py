
import sys, os
sys.path.append('../../python/')
import caffe
import numpy as np
from PIL import Image
import cv2
import random

class MultilabelDataLayer(caffe.Layer):
	""" Load Data for multi-label trainning """
	def setup(self, bottom, top):
		params = eval(self.param_str)
		self.root_dir   = params['root_dir'] 
		self.src_list   = params['src_list']
		self.split      = params['split']
		self.mean       = np.array(params['mean'])  # BGRD order
		self.random     = params.get('randomize', False)
		self.seed       = params.get('seed', None)
		self.crop       = 224
		self.batch_size = int(params.get('batch_size',1))
		self.label_len  = 5  # multi-label length
		self.gray_input = False
		self.is_regression = True

		print self.batch_size, self.mean, self.seed

		self.indices = open(self.src_list, 'r').readlines()
		self.idx_tbl = range(len(self.indices))

        # make eval deterministic 
		if 'train' not in self.split: 
			self.random = False

        # randomization: seed and pick
		if self.random:
			random.seed(self.seed)
			self.tbl = np.random.permutation(self.idx_tbl)

		self.idx  = 0

	def reshape(self, bottom, top):
		top[0].reshape(self.batch_size, 3, self.crop, self.crop)
		if self.is_regression:
			top[1].reshape(self.batch_size, self.label_len)
		else:
			top[1].reshape(self.batch_size, 1, self.label_len, 1)

	def forward(self, bottom, top):
		for it in range(self.batch_size):
			top[0].data[it,...], top[1].data[it,...] = self.load_data(self.idx_tbl[self.idx])
			self.idx += 1
			if self.idx == len(self.indices):
				self.idx = 0

	def backward(self, top, propagate_down, bottom):
		pass

	def load_data(self, idx):
		line_split = self.indices[idx].strip().split()
		im_file = line_split[0]
		label   = line_split[1:]
		assert(len(label) == self.label_len)
		for k in range(len(label)):
			label[k] = float(label[k])  # Float support both regression and classification
		if self.gray_input:
			im = cv2.imread(os.path.join(self.root_dir, im_file), cv2.CV_LOAD_IMAGE_GRAY)
		else:
			im = cv2.imread(os.path.join(self.root_dir, im_file), cv2.CV_LOAD_IMAGE_COLOR)
		im = np.array(im, dtype=np.float32)
		im = self.data_argument(im)

		return im, label

	def data_argument(self, im):
		im = cv2.resize(im, (self.crop,self.crop), interpolation=cv2.INTER_LINEAR)
		im -= self.mean[0:3]
		im = im.transpose((2,0,1))
		return im
