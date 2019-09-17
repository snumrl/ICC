from contextlib import contextmanager
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

def Plot(y_list, title,num_fig=1,path=None):
	plt.figure(num_fig, clear=True, figsize=(5.5, 4))
	plt.title(title)

	i = 0
	for y in y_list:
		plt.plot(y[0], label=y[1])
		i+= 1

	plt.legend(loc=2)
	plt.show()
	plt.pause(0.001)
	if path is not None:
		plt.savefig(path, format="png")


class TimeChecker():
	def __init__(self):
		self._timeList = {}

	@contextmanager
	def check(self, key):
		if key not in self._timeList:
			self._timeList[key] = 0
		_st = time.time()
		yield
		self._timeList[key] += time.time() - _st


	def printAll(self):
		for k, v in self._timeList.items():
			print("{} : {}s".format(k, v))

	def clear(self):
		self._timeList = {}