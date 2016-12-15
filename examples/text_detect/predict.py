#!/usr/bin/env python

import os, sys
import numpy as np
import cv2
import glob
import argparse
sys.path.append("../../python/")
import caffe
import matplotlib.pyplot as plt

def parse_input(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument("img", help="Input image")
	parser.add_argument("--model", default="./test.prototxt", help="Caffe prototxt file")
	parser.add_argument("--weights", 
			default="./snapshots/textdet_iter_10000.caffemodel", 
			help="Caffe pre-trained weight file")
	parser.add_argument("--gpu", default=True, help="Use gpu")
	args = parser.parse_args()
	return args

def predict(args):
	if args.gpu:
		caffe.set_mode_gpu()
		caffe.set_device(1)
	else:
		caffe.set_mode_cpu()

	img = cv2.imread(args.img)
	img = cv2.resize(img, (321, 321))

	in_ = img.copy()[:,:,::-1].astype(float)
	in_ -= np.array([104.008,116.669,122.675])
	in_ = in_.transpose((2,0,1))

	net = caffe.Net(args.model, args.weights, caffe.TEST)
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_
	net.forward()
	out = net.blobs['prob'].data[0].argmax(axis=0)
	
	# cv2.imshow("img", img)
	# cv2.imshow("pred", out.astype('uint8'))
	# cv2.waitKey()

	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
	ax1.imshow(img)
	ax1.set_title('img')
	ax2.imshow(out, extent=[0,img.shape[1],0,img.shape[0]], aspect=1)
	ax2.set_title('out')
	plt.tight_layout()
	plt.show()

if __name__=='__main__':
	args = parse_input(sys.argv)
	predict(args)
