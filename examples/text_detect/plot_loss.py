
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
	acc = accumulate(sys.argv[1], '#11: accuracy_res1 = ')
	plt.plot(loss, label = 'loss')
	plt.plot(acc, label = 'acc')
	plt.legend()
	plt.show()

