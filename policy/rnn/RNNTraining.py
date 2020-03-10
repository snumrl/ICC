import tensorflow as tf
import numpy as np
import time

from util.Util import Plot

from util.Pose2d import Pose2d
from util.dataLoader import loadData
from rnn.RNNConfig import RNNConfig
from rnn.RNNModel import RNNModel
from IPython import embed


class MotionData(object):
	def __init__(self):
		motion = RNNConfig.instance().motion

		self.x_data = loadData("../motions/%s/data/0_xData.dat"%(motion))
		self.y_data = loadData("../motions/%s/data/0_yData.dat"%(motion))

		self.xDimension = RNNConfig.instance().xDimension
		self.yDimension = RNNConfig.instance().yDimension
		self.footSlideWeight = RNNConfig.instance().footSlideWeight

		self.useControlPrediction = RNNConfig.instance().useControlPrediction

		self.yNormal = RNNConfig.instance().yNormal

		self.rootDimension = RNNConfig.instance().rootDimension
		self.rootAngleIdx = RNNConfig.instance().rootAngleIdx

		self.stepSize = RNNConfig.instance().stepSize
		self.batchSize = RNNConfig.instance().batchSize
		self.numEpoch = RNNConfig.instance().numEpoch

	def resetBatch(self):
		self.idxList = np.random.randint(0, len(self.y_data)-self.stepSize*self.numEpoch, size=self.batchSize)
		self.batchIdx = 0

	def getNextBatch(self):
		batch_x = []
		batch_y = []
		for idx in self.idxList:
			batch_x.append(self.x_data[idx+self.batchIdx*self.stepSize:idx+(self.batchIdx+1)*self.stepSize+1])
			batch_y.append(self.y_data[idx+self.batchIdx*self.stepSize:idx+(self.batchIdx+1)*self.stepSize+1])
		batch_x = np.array(batch_x, dtype=np.float32)
		batch_y = np.array(batch_y, dtype=np.float32)

		self.batchIdx += 1
		return batch_x, batch_y


	@tf.function
	def loss(self, y, x, generated):
		motion_y = y

		if self.useControlPrediction:
			motion_g = generated[:,:,:-(self.xDimension)]
		else:
			motion_g = generated

		
		loss_root, loss_pose = self.motion_mse_loss(motion_y[:,1:], motion_g)
		loss_root = loss_root
		loss_pose = loss_pose
		
				
		motion_g = self.yNormal.de_normalize(motion_g)
		motion_y_0 = self.yNormal.de_normalize(motion_y[:,0])
		
		loss_foot = self.foot_loss(motion_g, motion_y_0)*self.footSlideWeight
		

		# loss_joint = self.joint_len_loss(motion_g) * self.joint_len_weight
		loss_predict = tf.constant(0.0)
		if self.useControlPrediction:
			loss_predict = self.predict_loss(x, generated[:,:,-(self.xDimension):])
		
		loss = loss_root + loss_pose + loss_foot + loss_predict*0.3
		
		return loss, [loss, loss_root, loss_pose, loss_foot, loss_predict]

	@tf.function
	def predict_loss(self, x, output_x):
		loss_predict = tf.reduce_mean(tf.square(output_x - x))
		return loss_predict

	@tf.function
	def motion_mse_loss(self, y, output):

		rootStart = 0
		poseStart = self.rootDimension

		output_root = tf.slice(output, [0, 0, rootStart], [-1, -1, self.rootDimension])
		output_pose = tf.slice(output, [0, 0, poseStart], [-1, -1, self.yDimension - self.rootDimension])
		y_root = tf.slice(y, [0, 0, rootStart], [-1, -1, self.rootDimension])
		y_pose = tf.slice(y, [0, 0, poseStart], [-1, -1, self.yDimension - self.rootDimension])
		loss_root = tf.reduce_mean(tf.square(output_root - y_root)*[6,1,1,1])
		loss_pose = tf.reduce_mean(tf.square(output_pose - y_pose))

		       
		return loss_root, loss_pose

	@tf.function
	def foot_loss(self, output, prev_motion):
		r_idx = self.rootDimension
		a_idx = self.rootAngleIdx
		output_root = output[:,:,0:r_idx]
		output_pose = output[:,:,r_idx:]
		c_root = output_root
		
		# root, root height, Head_End, LeftHand
		foot_indices = [13, 14, 17, 18]
		dist_list = []
		for i in range(self.stepSize):
			if (i == 0):
				prev_pose = prev_motion[:,r_idx:]
			else:
				prev_pose = output_pose[:,i-1,:]
			
			current_root = output_root[:,i,:]
			current_pose = output_pose[:,i,:]
			
			cos = tf.cos(current_root[:,a_idx])
			sin = tf.sin(current_root[:,a_idx])
			dx_x = cos
			dx_z = sin
			dy_x = -sin
			dy_z = cos
			t_x = current_root[:, a_idx+1]
			t_z = -current_root[:, a_idx+2]
			
			for j in range(len(foot_indices)):
				idx = self.yNormal.get_zero_removed_index(1 + 3*foot_indices[j])
				if (j < 2):
					fc_idx = self.yNormal.get_zero_removed_index(0)
					if fc_idx == -1:
						f_contact = tf.convert_to_tensor([self.yNormal.get_mean(0)]*c_root.shape[0])
					else:
						f_contact = c_root[:,i,0]
				else:
					fc_idx = self.yNormal.get_zero_removed_index(1)
					if fc_idx == -1:
						f_contact = tf.convert_to_tensor([self.yNormal.get_mean(1)]*c_root.shape[0])
					else:
						f_contact = c_root[:,i,1]
				f_contact = tf.sign(tf.maximum(f_contact - 0.5, 0))
	#				 f_contact = tf.clip_by_value(f_contact - 0.5, 0, 0.5)
				moved_x = dx_x*current_pose[:,idx] + dy_x*current_pose[:,idx+2] + t_x
				moved_y = current_pose[:,idx+1]
				moved_z = dx_z*current_pose[:,idx] + dy_z*current_pose[:,idx+2] + t_z
				diff_x = (prev_pose[:, idx] - moved_x)*f_contact
				diff_y = (prev_pose[:, idx + 1] - moved_y)*f_contact
				diff_z = (prev_pose[:, idx + 2] - moved_z)*f_contact
				dist_list.extend([diff_x, diff_y, diff_z])
		
		return tf.reduce_mean(tf.square(dist_list))

	# def joint_len_loss(self, output):
	#	r_idx = ROOT_DIMENSION
	#	dist_list = []
	#	for sIdx in range(STEP_SIZE):
	#		current_pose = output[:,sIdx,r_idx:]
	#		for pIdx in range(len(self.jointPairs)):
	#			pair = self.jointPairs[pIdx]
	#			lenOrigin = self.jointLengths[pIdx]
	#			jLen = self.joint_len(current_pose, pair[0], pair[1])
	#			dist_list.append(tf.square(jLen - lenOrigin))
	#	return tf.reduce_mean(dist_list)

	# def joint_len(self, current_pose, j1, j2):
	#	idx1 = 1 + 3*j1
	#	idx2 = 1 + 3*j2
	#	dx = current_pose[:,idx1] - current_pose[:,idx2]
	#	dy = current_pose[:,idx1+1] - current_pose[:,idx2+1]
	#	dz = current_pose[:,idx1+2] - current_pose[:,idx2+2]
	#	d_len = tf.sqrt(tf.square(dx) + tf.square(dy) + tf.square(dz))
	#	return d_len

	




def train(motion_name):
	RNNConfig.instance().loadData(motion_name)
	data = MotionData()
	model = RNNModel()
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

	loss_name = ["total", "root", "pose", "foot", "pred"]
	loss_list = np.array([[]]*5)
	loss_list_smoothed = np.array([[]]*5)
	st = time.time()
	for c in range(100000000):
		data.resetBatch()
		model.resetState(data.batchSize)
		for n in range(data.numEpoch):
			batch_x, batch_y = data.getNextBatch()
			with tf.GradientTape() as tape:
				controls = tf.transpose(batch_x[:,:-1], perm=[1,0,2])
				generated = []
				pose = batch_y[:,0]
				for i in range(data.stepSize):
					control = controls[i]

					output = model.forwardOneStep(control, pose, True)
					generated.append(output)

					if data.useControlPrediction:
						pose = output[:,:data.yDimension]
					else:
						pose = output

				generated = tf.convert_to_tensor(generated)
				generated = tf.transpose(generated, perm=[1,0,2])

				loss, loss_detail = data.loss(batch_y, batch_x[:,1:], generated)
				reg_loss = model.losses
				total_loss = loss + reg_loss

			gradients = tape.gradient(total_loss, model.trainable_variables)
			gradients, _grad_norm = tf.clip_by_global_norm(gradients, 0.5)
			optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		if c%100 == 0:
			loss_detail = tf.convert_to_tensor(loss_detail).numpy()
			loss_list = np.insert(loss_list, loss_list.shape[1], loss_detail, axis=1)
			# loss_list_smoothed = np.insert(loss_list_smoothed, loss_list_smoothed.shape[1], np.array([np.mean(loss_list[-10:], axis=1)]), axis=1)
			Plot([*zip(loss_list, loss_name)], "loss_s", 1)
			print("\rElapsed : {:8.2f}s, Total : {:.6f}, [ root : {:.6f}, pose : {:.6f}, foot : {:.6f}, pred : {:.6f} ]".format(time.time()-st,*loss_detail))

			model.save("../motions/{}/train/network".format(motion_name))
		else:
			print("\r{}/100 : {:.6f}, {:.6f}".format(c%100, np.array(loss), np.array(reg_loss)), end="")



