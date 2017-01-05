
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
	acc = accumulate(sys.argv[1], '#2: accuracy = ')

	fig, (ax1, ax2) = plt.subplots(2, 1)
	ax1.plot(loss, label = 'loss')
	ax1.set_title('loss')
	ax1.grid(True)
	ax2.plot(acc, label = 'acc')
	ax2.set_title('acc')
	ax2.grid(True)
	fig.tight_layout()
	plt.show()

