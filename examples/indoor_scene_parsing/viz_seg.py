
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

"""
clsname_tbl: class name lists, such as ['wall', 'floor', 'ceiling', ...]
"""
class VizSeg(object):

	def __init__(self, clsname_tbl=None):
		self.clsname_tbl =clsname_tbl
		self.viz_colors = list(six.iteritems(colors.cnames))
		# Add the single letter colors.
		for name, rgb in six.iteritems(colors.ColorConverter.colors):
			hex_ = colors.rgb2hex(rgb)
			self.viz_colors.append((name, hex_))
		# Transform to hex color values.
		hex_ = [color[1] for color in self.viz_colors]
		# Get the rgb equivalent.
		self.rgb = [colors.hex2color(color) for color in hex_]
		self.bgr = [_rgb[::-1] for _rgb in self.rgb]

	def load_clstbl(self, clsfile):
		"""
		clsfile format: clsname line one by one
		"""
		clsname_tbl = []
		with open(clsfile, 'r') as rf:
			for line in rf.readlines():
				# txt = line.strip().split(u',')
				# if len(txt) >= 1 and len(txt[0]) > 0:
				# 	clsname_tbl.append(txt[0])
				clsname_tbl.append(line.strip())

		self.clsname_tbl = clsname_tbl


	def show(self, im, pred, gt=None, save_fname=None):
		fig, ((ax1,ax2),(ax4,ax3)) = plt.subplots(nrows=2, ncols=2)
		# fig.subplots_adjust(hspace=0)

		if im is not None:
			ax1.imshow(im)
			ax1.set_title('im')

		# Calc objnames
		if gt is not None:
			classes_idx = np.unique([pred[:], gt[:]])
		else:
			classes_idx = np.unique(pred[:])

		obj_names = []
		pred_rgb = np.empty((im.shape[0],im.shape[1],3),dtype=np.float32)
		pred_rgb[:,:,:] = pred[:,:,np.newaxis]
		if gt is not None:
			gt_rgb = np.empty((im.shape[0],im.shape[1],3),dtype=np.float32)
			gt_rgb[:,:,:] = gt[:,:,np.newaxis]
		for k in classes_idx:
			if self.clsname_tbl is None:
				obj_names.append(str(k))
			else:
				obj_names.append(self.clsname_tbl[k])
			pred_rgb[pred==k, :] = self.rgb[k]
			if gt is not None:
				gt_rgb[gt==k, :] = self.rgb[k]
	
		if pred is not None:
			ax2.imshow(pred_rgb, extent=[0,pred_rgb.shape[1],0,pred_rgb.shape[0]], aspect=1)
			ax2.set_title('predict')

		if gt is not None:
			ax4.imshow(gt_rgb)
			ax4.set_title('groundtruth')

		# Disp Colormap
		X, Y = fig.get_dpi() * fig.get_size_inches()
		max_col = 3
		max_row = 10
		while max_row*max_col < len(obj_names):
			max_row += 1
		w = X / max_col
		h = Y / (max_row + 1)

		for k, name in enumerate(obj_names):
			col = k % max_col;
			row = int(k / max_col);

			yi = Y - row * ( h+h/(2*max_row) )
			xi_line = w * (col + 0.00)
			xf_line = w * (col + 0.80)
			xi_text = w * (col + 0.40)

			ax3.text(xi_text, yi, name, fontsize=12, 
					horizontalalignment='center',
					verticalalignment='center')
			ax3.hlines(yi, xi_line, xf_line, color=self.viz_colors[classes_idx[k]], linewidth=(h * 0.7))

		plt.axis('off')
		plt.tight_layout()
		if save_fname is not None:
			plt.savefig(save_fname)
		else:
			plt.show()

