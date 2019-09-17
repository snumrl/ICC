import numpy as np

class AdaptiveSampler:
	def __init__(self, size):
		self._size = size

		self._rewardWeights = np.zeros(self._size)
		self._counts = np.zeros(self._size)


	def updateWeights(self, index, rew, end_of_trajectory):
		# update counts
		self._counts[index] += 1

		# update reward weights
		rew = float(rew)
		if end_of_trajectory:
			rew *= float(self._size) / (self._size - index)
		delta = rew - self._rewardWeights[index]
		self._rewardWeights[index] += delta * (1.0/(self._counts[index]+1))

	def selectTime(self):
		e = (self._rewardWeights-self._counts.min())
		e = np.clip(e, a_min=None, a_max=50)
		e = np.exp(-e*0.2)

		# uniform until all pieces are selected 10 times.
		if any(self._counts<10):
			e = np.ones(e.shape)
		tot = e.sum()
		e = e/tot

		index = self._size-1
		cur = 0
		ran = np.random.uniform(0.0,1.0)
		for i in range(self._size):
			cur += e[i]
			if ran < cur:
				index = i
				break

		time = (index + np.random.uniform(0.0, 1.0))/self._size

		return index, time

	def reset(self):
		self._rewardWeights = np.zeros(self._size)
		self._counts = np.zeros(self._size)

	def save(self, path, save_count):
		return