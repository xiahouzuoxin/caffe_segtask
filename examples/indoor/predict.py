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

from viz_seg import VizSeg

def parse_input(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument("img", help="Input image")
	parser.add_argument("--outdir", default="", help="Out Dir")
	parser.add_argument("--model", default="./test.prototxt", help="Caffe prototxt file")
	parser.add_argument("--clstxt", default="./ADE20K/classes_64.txt", help="class file(1 class 1 line)")
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
	net = caffe.Net(args.model, args.weights, caffe.TEST)

	if os.path.isdir(args.img):
		imgs = glob.glob(os.path.join(args.img, '*.JPG'))
	else:
		imgs = [args.img]

	seg_disp = VizSeg()
	# seg_disp.load_clstbl('NYUv2_40cls/classes.txt')
	# seg_disp.load_clstbl('ADE20K/classes_150.txt')
	# seg_disp.load_clstbl('ADE20K/classes_70.txt')
	seg_disp.load_clstbl(args.clstxt)

	for img_path in imgs:
		img = cv2.imread(img_path)
		img = cv2.resize(img, (473, 473))

		in_ = img.copy()[:,:,::-1].astype(float)
		in_ -= np.array([104.008,116.669,122.675])
		in_ = in_.transpose((2,0,1))

		batch_size = 1
		net.blobs['data'].reshape(batch_size, *in_.shape)
		for data_idx in range(batch_size):
			net.blobs['data'].data[data_idx,...] = in_

		t0 = time.time()
		net.forward()
		t1 = time.time()
		print("Forward Time(s) = %s\n"%(str(t1-t0)))

		out = net.blobs['prob'].data[0].argmax(axis=0)
		
		prefix = os.path.splitext(os.path.basename(img_path))[0]
		print prefix
		if len(args.outdir) > 0:
			seg_disp.show(img, out, save_fname=os.path.join(args.outdir, prefix+'.png'))
		else:
			seg_disp.show(img, out)

if __name__=='__main__':
	args = parse_input(sys.argv)
	predict(args)
