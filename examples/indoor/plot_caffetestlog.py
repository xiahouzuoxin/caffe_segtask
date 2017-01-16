
#!/usr/bin/python

import os, sys
import matplotlib.pyplot as plt
import re

def accumulate(log, reftxt):
	values = []
	with open(log,'r') as rf:
		for line in rf.readlines():
			if re.search(reftxt, line):
				_, valid_text = line.strip().split('=')
				values.append(float(valid_text))
	return values

if __name__=='__main__':
	acc = accumulate(sys.argv[1], 'accuracy = ')
	acc = acc[0:-1:3]

	n, bins, patches = plt.hist(acc, 100, facecolor='green', alpha=0.75)

	# plt.plot(acc, label = 'acc')
	# plt.title('acc')
	plt.grid(True)
	plt.show()

