import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

from IPython import embed

a = tf.Variable(3.0, trainable=True)

m = tf.keras.Sequential([tf.keras.layers.Dense(10),tf.keras.layers.Dense(1)])

x = tf.Variable([[3.0]])
with tf.GradientTape() as tape:
	b = m(x)+2*a

aa = tape.gradient(b, m.trainable_variables + [a])
# print(m(np.array([[10],[10],[10]], dtype=np.float32)))


embed()
exit()