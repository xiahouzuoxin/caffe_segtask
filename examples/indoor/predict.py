#!/usr/bin/env python

import os, sys
import numpy as np
import cv2
import glob
import argparse
sys.path.append("../../python/")
import caffe
import matplotlib.pyplot as plt
import time

def parse_input(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument("img", help="Input image")
	parser.add_argument("--model", default="./test.prototxt", help="Caffe prototxt file")
	parser.add_argument("--weights", 
			default="./snapshots/MSC-LargeFOV-voc12_iter_20000.caffemodel",
			help="Caffe pre-trained weight file")
	parser.add_argument("--gpu", default=True, help="Use gpu")
	args = parser.parse_args()
	return args

def predict(args):
	if args.gpu:
		caffe.set_mode_gpu()
		caffe.set_device(0)
	else:
		caffe.set_mode_cpu()

	img = cv2.imread(args.img)
	img = cv2.resize(img, (473, 473))

	in_ = img.copy()[:,:,::-1].astype(float)
	in_ -= np.array([104.008,116.669,122.675])
	in_ = in_.transpose((2,0,1))

	net = caffe.Net(args.model, args.weights, caffe.TEST)
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_

	t0 = time.time()
	net.forward()
	t1 = time.time()
	print("Forward Time(s) = %s\n"%(str(t1-t0)))

	out = net.blobs['conv6_interp'].data[0].argmax(axis=0)
	
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
