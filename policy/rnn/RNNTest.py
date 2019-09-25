import tensorflow as tf
import numpy as np
import time

from Utils import Plot

from util.Pose2d import Pose2d
from util.dataLoader import loadData
from RNNConfig import RNNConfig
from IPython import embed

motion_name = "walkrunfall"

class MotionData(object):
	def __init__(self):
		motion = RNNConfig.instance().motion

		self.x_data = loadData("../motions/%s/data/xData.dat"%(motion))
		self.y_data = loadData("../motions/%s/data/yData.dat"%(motion))

	def getBatch(self):
	 	idxList = np.random.randint(0, len(self.y_data)-RNNConfig.instance().stepSize, size=RNNConfig.instance().batchSize)
	 	batch_x = []
	 	batch_y = []
	 	for idx in idxList:
	 		batch_x.append(self.x_data[idx:idx+RNNConfig.instance().stepSize])
	 		batch_y.append(self.y_data[idx:idx+RNNConfig.instance().stepSize+1])
	 	batch_x = np.array(batch_x, dtype=np.float32)
	 	batch_y = np.array(batch_y, dtype=np.float32)

	 	return batch_x, batch_y



	def loss(self, y, generated):
		motion_y = y
		motion_g = generated
		
		loss_root, loss_pose = self.motion_mse_loss(motion_y[:,1:], motion_g)
		loss_root = loss_root
		loss_pose = loss_pose
		
				
		motion_g = RNNConfig.instance().yNormal.de_normalize(motion_g)
		motion_y_0 = RNNConfig.instance().yNormal.de_normalize(motion_y[:,0])
		
		loss_foot = self.foot_loss(motion_g, motion_y_0)*RNNConfig.instance().footSlideWeight
		
		# loss_joint = self.joint_len_loss(motion_g) * self.joint_len_weight
			
		
		loss = loss_root + loss_pose + loss_foot# + loss_joint
		return loss, [loss, loss_root, loss_pose, loss_foot]#, loss_joint]

	def motion_mse_loss(self, y, output):
		rootStart = 0
		poseStart = RNNConfig.instance().rootDimension

		output_root = tf.slice(output, [0, 0, rootStart], [-1, -1, RNNConfig.instance().rootDimension])
		output_pose = tf.slice(output, [0, 0, poseStart], [-1, -1, RNNConfig.instance().yDimension - RNNConfig.instance().rootDimension])
		y_root = tf.slice(y, [0, 0, rootStart], [-1, -1, RNNConfig.instance().rootDimension])
		y_pose = tf.slice(y, [0, 0, poseStart], [-1, -1, RNNConfig.instance().yDimension - RNNConfig.instance().rootDimension])
		loss_root = tf.reduce_mean(tf.square(output_root - y_root)*[6, 6, 1, 1, 1])
		loss_pose = tf.reduce_mean(tf.square(output_pose - y_pose))
		
			
		return loss_root, loss_pose

	def foot_loss(self, output, prev_motion):
		r_idx = RNNConfig.instance().rootDimension
		a_idx = 2
		output_root = output[:,:,0:r_idx]
		output_pose = output[:,:,r_idx:]
		c_root = output_root
		
		# root, root height, Head_End, LeftHand
		foot_indices = [2, 3, 5, 6]
		dist_list = []
		for i in range(STEP_SIZE):
			if (i == 0):
				prev_pose = prev_motion[:,r_idx:]
			else:
				prev_pose = output_pose[:,i-1,:]
			
			current_root = output_root[:,i,:]
			current_pose = output_pose[:,i,:]
			
			cos = tf.cos(current_root[:,a_idx])
			sin = tf.sin(current_root[:,a_idx])
			dx_x = cos
			dx_z = -sin
			dy_x = sin
			dy_z = cos
			t_x = current_root[:, a_idx+1]
			t_z = -current_root[:, a_idx+2]
			
			for j in range(len(foot_indices)):
				idx = 1 + 3*foot_indices[j]
				if (j < 2):
					f_contact = c_root[:,i,0]
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
	# 	r_idx = ROOT_DIMENSION
	# 	dist_list = []
	# 	for sIdx in range(STEP_SIZE):
	# 		current_pose = output[:,sIdx,r_idx:]
	# 		for pIdx in range(len(self.jointPairs)):
	# 			pair = self.jointPairs[pIdx]
	# 			lenOrigin = self.jointLengths[pIdx]
	# 			jLen = self.joint_len(current_pose, pair[0], pair[1])
	# 			dist_list.append(tf.square(jLen - lenOrigin))
	# 	return tf.reduce_mean(dist_list)

	# def joint_len(self, current_pose, j1, j2):
	# 	idx1 = 1 + 3*j1
	# 	idx2 = 1 + 3*j2
	# 	dx = current_pose[:,idx1] - current_pose[:,idx2]
	# 	dy = current_pose[:,idx1+1] - current_pose[:,idx2+1]
	# 	dz = current_pose[:,idx1+2] - current_pose[:,idx2+2]
	# 	d_len = tf.sqrt(tf.square(dx) + tf.square(dy) + tf.square(dz))
	# 	return d_len

	




if __name__ == "__main__":
	RNNConfig.instance().loadData(motion_name)
	data = MotionData()
	model = RNNModel()
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

	# TODO
	loss_name = ["total", "root", "pose", "foot"]
	loss_list = np.array([[]]*4)
	loss_list_smoothed = np.array([[]]*4)
	st = time.time()
	for c in range(10000000):
		batch_x, batch_y = data.getBatch()
		with tf.GradientTape() as tape:
			generated = model.forwardMultiple(batch_x, batch_y[:,0])
			loss = data.loss(batch_y, generated)
		gradients = tape.gradient(loss, model.trainable_variables)
		gradients, _grad_norm = tf.clip_by_global_norm(gradients, 0.5)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		if c%100 == 0:
			loss_detail = tf.convert_to_tensor(loss[1]).numpy()
			loss_list = np.insert(loss_list, loss_list.shape[1], loss_detail, axis=1)
			loss_list_smoothed = np.insert(loss_list_smoothed, loss_list_smoothed.shape[1], np.array([np.mean(loss_list[-10:], axis=1)]), axis=1)
			Plot([*zip(loss_list_smoothed, loss_name)], "loss_s", 1)
			print("Elapsed : {:8.2f}s, Total : {:.6f}, [ root : {:.6f}, pose : {:.6f}, foot : {:.6f} ]".format(time.time()-st,*loss_detail))

			model.save("../motions/train/{}/network".format(motion_name))



