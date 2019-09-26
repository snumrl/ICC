import tensorflow as tf
import numpy as np

from rnn.RNNModel import RNNModel
from rnn.RNNConfig import RNNConfig
from rl.Configurations import Configurations
from util.dataLoader import loadData
from util.Pose2d import Pose2d
from copy import deepcopy

from IPython import embed
import time
import math

class MotionGenerator(object):
	def __init__(self, num_slaves, motion="walk"):
		np.random.seed(int(time.time()))
		self.num_slaves = num_slaves
		self.motion = motion

		RNNConfig.instance().loadData(motion)


		# initialize kinematic poses(root and character poses)
		self.rootPose = []
		self.characterPose = []
		self.initialCharacterPose = np.zeros(RNNConfig.instance().yDimension, dtype=np.float32)
		for _ in range(self.num_slaves):
			self.rootPose.append(Pose2d())
			self.characterPose.append(self.initialCharacterPose)

		self.model = None
		self.isModelLoaded = False


		# parameter for root height
		self.target_height = 88.

		# random target parameters
		self.target_dist_lower = 600.0
		self.target_dist_upper = 650.0
		self.target_angle_upper = math.pi*0.5
		self.target_angle_lower = math.pi*(-0.5)

		# initialize targets
		self.targets = []
		for i in range(self.num_slaves):
			self.targets.append(self.randomTarget(i))

		self.resetAll()
		

	def resetAll(self, targets=None):
		# reset root and character poses
		self.characterPose = np.array(self.characterPose)
		for i in range(self.num_slaves):
			self.rootPose[i] = Pose2d()
			self.characterPose[i] = self.initialCharacterPose

		# reset state
		if self.model is not None:
			self.model.resetState(self.num_slaves)

		if targets is not None:
			self.targets = targets
		else:
			for i in range(self.num_slaves):
				self.targets[i] = self.randomTarget(i)
		for _ in range(100):
			self.getReferences(targets)


	def loadNetworks(self):
		# initialize rnn model
		self.model = RNNModel()

		# load network
		self.model.restore("../motions/{}/train/network".format(self.motion))
		self.isModelLoaded = True


		self.resetAll()

	def randomTarget(self, index):
		target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
		target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
		local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
		local_pose = Pose2d(local_target)
		target = self.rootPose[index].localToGlobal(local_pose).p
		if self.motion == "walkrunfall":
			target = target + [self.target_height]
		else:
			print("policy/rnn/RNNManager.py/randomTarget: use default target generation")
	
		return np.array(target, dtype=np.float32)

	def convertAndClipTarget(self, targets):
		# clip target and change to local coordinate
		targets = deepcopy(targets)
		for j in range(self.num_slaves):
			t = targets[j][:2]
			t = Pose2d(t)
			t = self.rootPose[j].relativePose(t)
			t = t.p
			t_len = math.sqrt(t[0]*t[0] + t[1]*t[1])
			clip_len = 250
			if (t_len > clip_len):
				ratio = clip_len/t_len
				t[0] *= ratio
				t[1] *= ratio
			targets[j][:2] = t
			targets[j] = RNNConfig.instance().xNormal.normalize_l(targets[j])
		return np.array(targets, dtype=np.float32)




	# convert local to global
	def getGlobalPositions(self, output, index):
		output = output[2:] # first two elements is about foot contact
		# move root
		self.rootPose[index] = self.rootPose[index].transform(output)

		points = [[0, output[3], 0]]

		positions = np.zeros(Configurations.instance().TCMotionSize)

		# root 
		positions[0:3] = self.rootPose[index].global_point_3d(points[0])
		positions[3:4] = self.rootPose[index].rotatedAngle()

		# other joints
		output = output[4:] # 4 : root
		output = output[57:] # 57 : 3d positions
		positions[4:52] = output[0:48] # only use joint angles

		return positions


	def getReferences(self, targets=None):
		# if target is given, set target
		if targets is not None:
			self.targets = targets
		# else use random generated targets which are generated when the charater is close enough
		else:
			for i in range(self.num_slaves):
				cur_pose = self.rootPose[i].p
				target = self.targets[i]
				dx = cur_pose[0] - target[0]
				dy = cur_pose[1] - target[1]
				if(dx*dx+dy*dy<100*100):
					self.targets[i] = self.randomTarget(i)

		convertedTargets = self.convertAndClipTarget(self.targets)

		# run rnn model
		if self.isModelLoaded is False:
			self.loadNetworks()
		self.characterPose = self.model.forwardOneStep(tf.convert_to_tensor(convertedTargets), tf.convert_to_tensor(self.characterPose), training=False)
		
		# convert outputs to global coordinate
		output = self.characterPose.numpy()
		pose_list = []
		for j in range(self.num_slaves):
			pose = RNNConfig.instance().yNormal.de_normalize_l(output[j])
			pose = RNNConfig.instance().yNormal.get_data_with_zeros(pose)
			pose = self.getGlobalPositions(pose, j)
			pose_list.append(pose)

		return pose_list


	def getTrajectory(self, frame=2000, targets=None):
		self.resetAll(targets)
		trajectories = []
		target_trajectories = []
		for _ in  range(frame):
			tr = self.getReferences(targets)
			trajectories.append(tr)
			t = self.getTargets()
			if self.motion == "walkrunfall":
				t = t[:,:2]
			target_trajectories.append(t)

		trajectories = np.asarray([*zip(*trajectories)], dtype=np.float32)
		target_trajectories = np.asarray([*zip(*target_trajectories)], dtype=np.float32)

		return trajectories, target_trajectories

	def getOriginalTrajectory(self, frame, origin_offset=0):
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
			targets.append(self.rootPose[0].localToGlobal(localPose).p)
			trajectories.append(self.getGlobalPositions(y, 0))

		trajectories = np.asarray(trajectories, dtype=np.float32)
		targets = np.asarray(targets, dtype=np.float32)
		return trajectories, targets

	def getTargets(self):
		return np.asarray(self.targets, dtype=np.float32)




# output : 
# [
# 	foot_contact 	: 2
# 	root_angle 		: 1
# 	root_translate 	: 2
# 	root_height		: 1
# 	joint_positions : 57 = 19 * 3
	# "Head_End",			: 6
	# "LeftHand",			: 9 
	# "LeftFoot",			: 12
	# "LeftToeBase",		: 15
	# "RightHand",			: 18
	# "RightFoot",			: 21
	# "RightToeBase",		: 24

	# "LeftArm",			: 27
	# "RightArm",			: 30

	# "LeftForeArm",		: 33
	# "LeftLeg",			: 36
	# "RightForeArm",		: 39
	# "RightLeg",			: 42

	# "Spine",				: 45
	# "LeftHandIndex1",		: 48
	# "RightHandIndex1",	: 51
	# "Neck1",				: 54
	# "LeftUpLeg",			: 57
	# "RightUpLeg",			: 60
# 	joint_angles 	: 75 = 25 * 3
	# "Head",				: 63
	# "Hips",				: 66
	# "LHipJoint",			: 69
	# "LeftArm",			: 72
	# "LeftFoot",			: 75
	# "LeftForeArm",		: 78
	# "LeftHand",			: 81
	# "LeftLeg",			: 84
	# "LeftShoulder",		: 87
	# "LeftToeBase",		: 90
	# "LeftUpLeg",			: 93
	# "LowerBack",			: 96
	# "Neck",				: 99
	# "Neck1",				: 102
	# "RHipJoint",			: 105
	# "RightArm",			: 108
	# "RightFoot",			: 111
	# "RightForeArm",		: 114
	# "RightHand",			: 117
	# "RightLeg",			: 120
	# "RightShoulder",		: 123
	# "RightToeBase",		: 126
	# "RightUpLeg",			: 129
	# "Spine",				: 132
	# "Spine1",				: 135
# 	total			: 138
# ]
# embed()
# exit()