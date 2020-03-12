import tensorflow as tf
import numpy as np
from copy import deepcopy

from rnn.RNNConfig import RNNConfig

from IPython import embed

class ForgetBiasInitializer(tf.keras.initializers.Initializer):
	def __init__(self, value=0):
		if not (np.isscalar(value) or isinstance(value, (list, tuple, np.ndarray))):
			raise TypeError(
				"Invalid type for initial value: %s (expected Python scalar, list or "
				"tuple of values, or numpy.ndarray)." % type(value))
		self.value = value

	def __call__(self, shape, dtype):
		if dtype is not None:
			dtype = tf.dtypes.as_dtype(dtype)

		ret = np.zeros(shape)
		ret[int(shape[0]/4):int(shape[0]/2)].fill(self.value)
		return tf.convert_to_tensor(ret, dtype=dtype)

class RNNModel(object):
	def __init__(self):
		input_size = RNNConfig.instance().xDimension + RNNConfig.instance().yDimension
		if RNNConfig.instance().useControlPrediction:
			output_size = RNNConfig.instance().yDimension + RNNConfig.instance().xDimension
		else:
			output_size = RNNConfig.instance().yDimension

		self.regularizer = tf.keras.regularizers.l2(0.00001)

		cells = [tf.keras.layers.LSTMCell(
				units=RNNConfig.instance().lstmLayerSize, 
				unit_forget_bias=False, 
				bias_initializer=ForgetBiasInitializer(0.8), 
				dropout=0.0,
				kernel_regularizer=self.regularizer,
				bias_regularizer=self.regularizer,
				recurrent_regularizer=self.regularizer
			) for _ in range(RNNConfig.instance().lstmLayerNumber)]

		self.stacked_cells = tf.keras.layers.StackedRNNCells(cells, input_shape=(None, input_size), dtype=tf.float32)
		self.dense = tf.keras.layers.Dense(output_size,
			kernel_regularizer=self.regularizer,
			bias_regularizer=self.regularizer,
			dtype=tf.float32)

		self.stacked_cells.build(input_shape=(None, input_size))
		self.dropout = tf.keras.layers.Dropout(0.0)
		self.dense.build(input_shape=(None, RNNConfig.instance().lstmLayerSize))


		self._trainable_variables = self.stacked_cells.trainable_variables + self.dense.trainable_variables
		self._losses = tf.reduce_sum(tf.convert_to_tensor(self.stacked_cells.losses + self.dense.losses))

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

	@tf.function
	def forward(self, controls, pose, state, training=True):
		inputs = tf.concat([controls, pose], 1)
		m, new_state = self.stacked_cells(inputs, state, training=training)
		m = self.dropout(m, training=training)
		outputs = self.dense(m)

		return outputs, new_state

	def forwardOneStep(self, controls, pose, training=True):
		outputs, self.state = self.forward(controls, pose, self.state, training)
		return outputs

	@property
	def trainable_variables(self):
		return self._trainable_variables

	@property
	def losses(self):
		return self._losses
	

	def save(self, path):
		self._ckpt.write(path)

	def restore(self, path):
		self._ckpt.restore(path)