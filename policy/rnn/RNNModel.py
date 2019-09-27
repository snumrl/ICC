import tensorflow as tf
import numpy as np
from copy import deepcopy

from rnn.RNNConfig import RNNConfig

from IPython import embed

class RNNModel(object):
	def __init__(self):
		input_size = RNNConfig.instance().xDimension + RNNConfig.instance().yDimension
		output_size = RNNConfig.instance().yDimension

		cells = [tf.keras.layers.LSTMCell(units=RNNConfig.instance().lstmLayerSize, dropout=0.1) for _ in range(RNNConfig.instance().lstmLayerNumber)]
		self.stacked_cells = tf.keras.layers.StackedRNNCells(cells, input_shape=(None, input_size), dtype=tf.float32)
		self.dense = tf.keras.layers.Dense(output_size, dtype=tf.float32)

		self.stacked_cells.build(input_shape=(None, input_size))
		self.dense.build(input_shape=(None, RNNConfig.instance().lstmLayerSize))

		self._trainable_variables = self.stacked_cells.trainable_variables + self.dense.trainable_variables

		self._ckpt = tf.train.Checkpoint(
			stacked_cells=self.stacked_cells,
			dense=self.dense
		)

	def resetState(self, batch_size):
		self.state = self.stacked_cells.get_initial_state(batch_size=batch_size, dtype=tf.float32)

	def saveState(self):
		self.savedState = self.state

	def loadState(self):
		self.state = self.savedState

	def forwardOneStep(self, controls, pose, training=True):
		inputs = tf.concat([controls, pose], 1)
		m, self.state = self.stacked_cells(inputs, self.state, training=training)
		outputs = self.dense(m)

		return outputs

	def forwardMultiple(self, controls, initial_pose):
		self.resetState(controls.shape[0])
		controls = tf.transpose(controls, perm=[1,0,2])
		outputList = []
		pose = initial_pose
		for i in range(len(controls)):
			pose = self.forwardOneStep(controls[i], pose, True)
			outputList.append(pose)

		outputList = tf.convert_to_tensor(outputList)
		outputList = tf.transpose(outputList, perm=[1,0,2])
		return outputList

	@property
	def trainable_variables(self):
		return self._trainable_variables

	def save(self, path):
		self._ckpt.write(path)

	def restore(self, path):
		self._ckpt.restore(path)