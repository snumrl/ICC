import collections
from collections import namedtuple
from collections import deque
import random
import numpy as np

Transition = namedtuple('Transition',('s', 'a', 'r', 'value', 'neglogprob', 'TD', 'GAE'))
Transition.__new__.__defaults__ = (None, )*len(Transition._fields)

# tuple for adaptive initial states
# t = {initial states, states mod, rew_discounted_sum, prev_log_prob} 

class Episode(object):
	def __init__(self):
		self._data = []

	def push(self, *args):
		self._data.append(Transition(*args))

	@property
	def data(self):
		return self._data

	def getTotalReward(self, gamma = 1.0):
		data = self.data
		rew = 0
		for i in reversed(range(len(data))):
			rew = data[i].r + rew*gamma
		return rew
	

class ReplayBuffer(object):
	def __init__(self, buff_size = None):
		super(ReplayBuffer, self).__init__()
		self._buffer = deque(maxlen=buff_size)

	@property
	def buffer(self):
		return self._buffer
	

	def size(self):
		return len(self._buffer)

	def push(self,*args):
		self._buffer.append(Transition(*args))

	def clear(self):
		self._buffer.clear()