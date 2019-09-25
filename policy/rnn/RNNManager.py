from rnn.RNNController import RNNController
from util.Pose2d import Pose2d
from copy import deepcopy

import numpy as np
from IPython import embed
import time
import math

class RNNManager(object):
	def __init__(self, num_slaves, motion="walk"):
		np.random.seed(int(time.time()))
		self.num_slaves = num_slaves
		self.motion = motion
		print("Loading RNN : {}".format(self.motion))
		self.controller = RNNController(self.motion, self.num_slaves)

		self.target_dist_lower = 300.0
		self.target_dist_upper = 350.0
		self.target_angle_upper = math.pi*0.1
		self.target_angle_lower = math.pi*(-0.1)
		self.targets = []
		self.target_of_target = []
		self.target_height = 88.

		for i in range(self.num_slaves):
			self.targets.append(self.randomTarget(i));

		self.resetAll()
		

	def resetAll(self, targets=None):
		self.controller.resetAll()
		if targets is not None:
			self.targets = targets
		for _ in range(100):
			self.getReferences(targets)


	def randomTarget(self, index):
		target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
		target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
		local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
		local_pose = Pose2d(local_target)
		target = self.controller.pose[index].localToGlobal(local_pose).p
		if self.motion == "walkrunfall":
			target = target + [self.target_height]
		else:
			print("policy/rnn/RNNManager.py/randomTarget: use default target generation")
	
		return np.array(target, dtype=np.float32)


	def getReferences(self, targets=None):
		if targets is not None:
			self.targets = targets
		else:
			for i in range(self.num_slaves):
				cur_pose = self.controller.pose[i].p
				target = self.targets[i]
				dx = cur_pose[0] - target[0]
				dy = cur_pose[1] - target[1]
				if(dx*dx+dy*dy<100*100):
					self.targets[i] = self.randomTarget(i)
					print(self.targets)
		return self.controller.step(self.targets)


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
		return self.controller.getOriginalTrajectory(frame, origin_offset)

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