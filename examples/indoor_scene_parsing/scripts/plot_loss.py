
import os, sys
import matplotlib.pyplot as plt
import re

def accumulate(log, reftxt):
	loss = []
	with open(log) as rf:
		for line in rf.readlines():
			if re.search(reftxt, line):
				_, valid_text = line.strip().split('=')
				loss.append(float(valid_text))
	return loss

if __name__=="__main__":
	loss = accumulate(sys.argv[1], 'loss = ')
	acc = accumulate(sys.argv[1], '#0: accuracy = ')
	recall = accumulate(sys.argv[1], '#1: accuracy = ')
	iou = accumulate(sys.argv[1], '#2: accuracy = ')

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	ax1.plot(loss, 'r-', label = 'loss')
	ax1.set_title('loss')
	ax1.grid(True)
	ax2.plot(acc, 'ro-', label = 'acc')
	ax2.set_title('Accuracy')
	ax2.grid(True)
	ax3.plot(recall, 'bo-', label = 'recall')
	ax3.set_title('Recall')
	ax3.grid(True)
	ax4.plot(iou, 'go-', label = 'iou')
	ax4.set_title('IoU')
	ax4.grid(True)
	fig.tight_layout()
	plt.show()

