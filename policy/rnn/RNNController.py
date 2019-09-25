import math

import numpy as np
import tensorflow as  tf
from tensorflow.python import pywrap_tensorflow
from util.dataLoader import loadData

from util.Pose2d import Pose2d
from IPython import embed
from copy import deepcopy

from Configurations import Configurations
from rnn.RNNModel import RNNModel
from rnn.RNNConfig import RNNConfig

class RNNController(object):
	def __init__(self, motion, num_slaves):
		RNNConfig.instance().loadData(motion)
		RNNConfig.instance().batchSize = num_slaves

		self.motion = motion
		self.num_slaves = num_slaves

		# initialize pose to zero
		self.pose = []
		self.characterPose = []
		self.initialCharacterPose = np.zeros(RNNConfig.instance().yDimension, dtype=np.float32)
		for _ in range(self.num_slaves):
			self.pose.append(Pose2d())
			self.characterPose.append(self.initialCharacterPose)

		self.model = None

	def loadNetworks(self):
		# initialize rnn model
		self.model = RNNModel()

		# load network
		self.model.restore("../motions/{}/train/network".format(self.motion))


		self.resetAll()

	def resetAll(self):
		self.characterPose = np.array(self.characterPose)
		for i in range(self.num_slaves):
			self.pose[i] = Pose2d()
			self.characterPose[i] = self.initialCharacterPose

		if self.model is not None:
			self.model.resetState(self.num_slaves)

	# convert local to global
	def getGlobalPositions(self, output, index):
		output = output[2:] # first two elements is about foot contact
		# move root
		self.pose[index] = self.pose[index].transform(output)

		points = [[0, output[3], 0]]

		positions = np.zeros(Configurations.instance().TCMotionSize)

		# root 
		positions[0:3] = self.pose[index].global_point_3d(points[0])
		positions[3:4] = self.pose[index].rotatedAngle()

		# other joints
		output = output[4:] # 4 : root
		output = output[57:] # 57 : 3d positions
		positions[4:52] = output[0:48] # only use joint angles

		return positions

	def step(self, controls):
		if self.model is None:
			self.loadNetworks()
		controls = deepcopy(controls)
		for j in range(self.num_slaves):
			t = controls[j][:2]
			t = Pose2d(t)
			t = self.pose[j].relativePose(t)
			t = t.p
			t_len = math.sqrt(t[0]*t[0] + t[1]*t[1])
			clip_len = 150
			if (t_len > clip_len):
				ratio = clip_len/t_len
				t[0] *= ratio
				t[1] *= ratio

			controls[j] = RNNConfig.instance().xNormal.normalize_l(controls[j])
		controls = np.array(controls, dtype=np.float32)
		# run rnn model
		output = self.model.forwardOneStep(tf.convert_to_tensor(controls), tf.convert_to_tensor(self.characterPose), training=False)
		self.characterPose = output

		# convert outputs
		output = output.numpy()
		pose_list = []
		for j in range(self.num_slaves):
			pose = RNNConfig.instance().yNormal.de_normalize_l(output[j])
			pose = RNNConfig.instance().yNormal.get_data_with_zeros(pose)
			pose = self.getGlobalPositions(pose, j)
			pose_list.append(pose)

		return pose_list



	def getOriginalTrajectory(self, frame, origin_offset=0): # return global goals
		x_dat = loadData("../motions/{}/data/xData.dat".format(self.motion))
		y_dat = loadData("../motions/{}/data/yData.dat".format(self.motion))

		x_dat = x_dat[1+origin_offset:frame+1+origin_offset]
		y_dat = y_dat[1+origin_offset:frame+1+origin_offset]

		x_dat = np.array([RNNConfig.instance().xNormal.get_data_with_zeros(RNNConfig.instance().xNormal.de_normalize_l(x)) for x in x_dat])
		y_dat = np.array([RNNConfig.instance().yNormal.get_data_with_zeros(RNNConfig.instance().yNormal.de_normalize_l(y)) for y in y_dat])


		self.resetAll()

		trajectories = []
		targets = []

		for x, y in zip(x_dat, y_dat):
			localPose = Pose2d(x[:2])
			targets.append(self.pose[0].localToGlobal(localPose).p)
			trajectories.append(self.getGlobalPositions(y, 0))

		trajectories = np.asarray(trajectories, dtype=np.float32)
		targets = np.asarray(targets, dtype=np.float32)
		return trajectories, targets