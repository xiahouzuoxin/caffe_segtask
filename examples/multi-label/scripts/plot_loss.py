
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
	loss = accumulate(sys.argv[1], ', loss = ')

	plt.plot(loss, 'r-', label = 'loss')
	plt.title('loss')
	plt.grid(True)
	plt.show()

