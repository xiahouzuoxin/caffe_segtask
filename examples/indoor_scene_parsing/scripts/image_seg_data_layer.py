
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
		self.root_dir   = params['root_dir'] 
		self.src_list   = params['src_list']
		self.split      = params['split']
		self.mean       = np.array(params['mean'])  # BGRD order
		self.scale      = np.array([1.0/255,1.0/255,1.0/255,1.0/8])  # (x - mean) * scale
		self.random     = params.get('randomize', False)
		self.seed       = params.get('seed', None)
		self.crop       = 473
		self.batch_size = int(params.get('batch_size',1))  # real batch_size = self.batch_size * self.argu_n
		self.argu       = DataArgument()
		# self.argu_types = [self.argu.imresize, self.argu.random_crop]
		self.argu_types = [self.argu.imresize]
		self.argu_n     = len(self.argu_types)

		self.has_depth  = True

		# check size
		if self.has_depth:
			if len(top) != 3:
				raise Exception("Need to define three tops: data label and depth.") 
			if len(self.mean) != 4:
				raise Exception("mean_value: [B, G, R, D].")
		else:
			# two tops: data and label 
			if len(top) != 2:
				raise Exception("Need to define two tops: data and label.") 
			if len(self.mean) != 3:
				raise Exception("mean_value: [B, G, R].")
		if len(bottom) != 0: 
			raise Exception("Do not define a bottom.")  # data layers have no bottoms 

		# File Format: Deeplab's list format
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
		# reshape tops to fit
		real_batch_size = self.batch_size * self.argu_n
		top[0].reshape(real_batch_size, 3, self.crop, self.crop)  # image
		top[1].reshape(real_batch_size, 1, self.crop, self.crop)  # groundtruth
		if self.has_depth:
			top[2].reshape(real_batch_size, 1, self.crop, self.crop)  # depth

	def forward(self, bottom, top):
		for it in range(self.batch_size):
			if self.has_depth:
				im, gt, dp = self.load_data(self.idx_tbl[self.idx])
			else:
				im, gt = self.load_data(self.idx_tbl[self.idx])

			for k in range(self.argu_n):
				# assign output
				if self.has_depth:
					top[0].data[it*self.argu_n+k, ...], top[1].data[it*self.argu_n+k, ...], \
							top[2].data[it*self.argu_n+k, ...] = \
							self.data_argument(im, gt, self.argu_types[k], dp)
				else:
					top[0].data[it*self.argu_n+k, ...], top[1].data[it*self.argu_n+k, ...] = \
							self.data_argument(im, gt, self.argu_types[k])

			# pick next input
			self.idx += 1
			if self.idx == len(self.indices):
				self.idx = 0

	def backward(self, top, propagate_down, bottom):
		pass

	def load_data(self, idx):
		""" 
		Load image and groundtruth. Using @cv2.imread, return image is BGR order.
		"""
		if self.has_depth:
			im_file, gt_file, dp_file = self.indices[idx].strip().split()
		else:
			im_file, gt_file = self.indices[idx].strip().split()[0:2]

		im = cv2.imread(os.path.join(self.root_dir, im_file), cv2.CV_LOAD_IMAGE_COLOR)
		gt = cv2.imread(os.path.join(self.root_dir, gt_file), cv2.CV_LOAD_IMAGE_UNCHANGED)
		if (gt.ndim == 3):
			gt = gt[:,:,0]

		im = np.array(im, dtype=np.float32)
		gt = np.array(gt, dtype=np.uint8)
		assert(im.shape[0]==gt.shape[0] and im.shape[1]==gt.shape[1])
		if self.has_depth:
			dp = cv2.imread(os.path.join(self.root_dir, dp_file), cv2.CV_LOAD_IMAGE_UNCHANGED)
			dp = self.parse_depth_sunrgbd(dp)
			assert(im.shape[0]==dp.shape[0] and im.shape[1]==dp.shape[1])
			return im, gt, dp
		else:
			return im, gt

	def data_argument(self, im, gt, argu_func, depth=None):
		"""
        Preprocess for Caffe:
		- [x] Argument
        - [x] cast to float
        - [ ] switch channels RGB -> BGR
        - [x] subtract mean
        - [x] transpose to channel x height x width order
		- [x] label 0 indicate <UNK>, round 0 to 255 (Please set ignore_label: 255 in .prototxt)
		"""
		if self.has_depth:
			im, gt, depth = argu_func(im, gt, self.crop, depth)
			assert(im.shape[0]==depth.shape[0] and im.shape[1]==depth.shape[1])
		else:
			im, gt = argu_func(im, gt, self.crop)

		assert(im.shape[0]==gt.shape[0] and im.shape[1]==gt.shape[1] and 
			   im.shape[0]==self.crop and im.shape[1]==self.crop)

		if (im.ndim == 2):
			im = np.repeat(im[:,:,None], 3, axis=2)
		# im = im[:,:,::-1]   # Already BGR order using opencv's imread
		im -= self.mean[0:3]
		im *= self.scale[0:3]
		im = im.transpose((2,0,1))

		gt = np.array(gt, dtype=np.uint8)
		gt -= 1   # 0 is <UNK> class, round to 255
		gt = gt[np.newaxis,...]

		if self.has_depth:
			depth -= self.mean[3]
			depth *= self.scale[3]
			depth = depth[np.newaxis,...]
			return im, gt, depth
		else:
			return im, gt

	def parse_depth_sunrgbd(self, dp):
		dp = np.bitwise_or( np.right_shift(dp,3), np.left_shift(dp,16-3) )
		dp = np.array(dp, dtype=np.float32) / 1000
		dp[dp>8.0] = 8.0
		return dp


class DataArgument(object):
	"""
	DataArgument Functions: function template `def func(im, gt, crop_size, ...)` return `im, gt`
	imresize           : resize image and groundtruth to any size
	random_crop        : crop with random offset (If image too small, resize mininum edge to crop_size)
	random_resize      : resize to any size then crop to output size
	random_shrink_zoom : shrink/zoom to any size then crop to ouput size
	flip_hori          : generate mirror image and groundtruth
	random_blur        : blur with random kernelsize
	light_adjust       : 
	"""
	def __init__(self):
		pass

	def __resize(self, im, gt, out_dim, depth=None):
		new_im = cv2.resize(im, out_dim, interpolation=cv2.INTER_LINEAR)
		if gt is not None:
			new_gt = cv2.resize(gt, out_dim, interpolation=cv2.INTER_NEAREST)
		if depth is not None:
			new_depth = cv2.resize(depth, out_dim, interpolation=cv2.INTER_LINEAR)
			return new_im, new_gt, new_depth
		else:
			return new_im, new_gt

	def imresize(self, im, gt, crop_size, depth=None):
		return self.__resize(im, gt, (crop_size, crop_size), depth)

	def random_crop(self, im, gt, crop_size, depth=None):
		ori_shape = (gt.shape[0], gt.shape[1])
		# print im.shape
		min_edge = min(ori_shape[0], ori_shape[1])
		if min_edge < crop_size:
			scale = float(crop_size) / min_edge
			# shape to opencv is [w,h], while numpy is [h,w]
			new_shape = tuple([int(np.ceil(scale * ori_shape[1])), int(np.ceil(scale * ori_shape[0]))])
			if depth is not None:
				im, gt, depth = self.__resize(im, gt, new_shape, depth)
			else:
				im, gt= self.__resize(im, gt, new_shape)
			new_shape = new_shape[::-1]
		else:
			new_shape = ori_shape
		# print im.shape

		offset_h = np.random.randint(0, new_shape[0]-crop_size+1)
		offset_w = np.random.randint(0, new_shape[1]-crop_size+1)
		# print new_shape, offset_h, offset_w

		im = im[offset_h:(offset_h+crop_size), offset_w:(offset_w+crop_size), :]
		gt = gt[offset_h:(offset_h+crop_size), offset_w:(offset_w+crop_size)]
		if depth is not None:
			depth = depth[offset_h:(offset_h+crop_size), offset_w:(offset_w+crop_size)]
			return im, gt, depth
		else:
			return im, gt

	def random_resize(self, im, gt, crop_size, minsize=300, maxsize=500, depth=None):
		s = np.random.randint(minsize,maxsize,size=(1,2))
		if depth is not None:
			im, gt, depth = self.__resize(im, gt, s, depth)
		else:
			im, gt = self.__resize(im, gt, s)
		return random_crop(im, gt, crop_size, depth)

	def random_shrink_zoom(self, im, gt, crop_size, depth=None):
		ratio = 0.5 + np.random.rand()   # [0.5,1.0]
		new_shape = (int(im.shape[0]*ratio), int(im.shape[1]*ratio))
		if depth is not None:
			im, gt, depth = self.__resize(im, gt, new_shape, depth)
		else:
			im, gt = self.__resize(im, gt, new_shape)
		return random_crop(im, gt, crop_size)

	def flip_hori(self, im, gt, crop_size, depth=None):
		return cv2.flip(im,1), cv2.flip(gt,1), depth

	def random_blur(self, im, gt, crop_size, depth=None):
		ksize = np.random.randint(1,5)
		return cv2.blur(im, (2*ksize+1,2*ksize+1)), gt, depth

	def light_adjust(self, im, gt, crop_size, pix_ratio=1.0, mean_ratio=0.0, depth=None):
		cvt_im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
		mean = np.mean(cvt_im[:,:,1])
		# cvt_im[:,:,1] = cvt_im[:,:,1] - mean
		cvt_im[:,:,1] = cvt_im[:,:,1] * pix_ratio + mean * mean_ratio
		re_im = cv2.cvtColor(cvt_im, cv2.COLOR_YUV2BGR)
		return re_im, gt, depth
