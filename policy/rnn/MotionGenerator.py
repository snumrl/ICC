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
		self.rootPosePrev = []

		self.characterPose = []
		self.controlPrediction = []
		self.initialCharacterPose = np.zeros(RNNConfig.instance().yDimension, dtype=np.float32)
		self.initialControlPrediction = np.zeros(RNNConfig.instance().xDimension, dtype=np.float32)
		for _ in range(self.num_slaves):
			self.rootPose.append(Pose2d())
			self.rootPosePrev.append(Pose2d())
			self.characterPose.append(self.initialCharacterPose)
			self.controlPrediction.append(self.initialControlPrediction)

		self.model = None
		self.isModelLoaded = False


		# parameter for root height
		self.target_height = 88.

		# parameter for fall over and stand up
		self.fallOverState = [0]*self.num_slaves
		self.fallOverCount = [0]*self.num_slaves
		self.standUpCount = [40]*self.num_slaves

		self.isWalk = True

		# random target parameters
		self.target_dist_lower = 600.0
		self.target_dist_upper = 650.0
		self.target_angle_upper = math.pi*0.5
		self.target_angle_lower = math.pi*(-0.5)

		# initialize targets
		self.targets = []
		for i in range(self.num_slaves):
			self.targets.append(self.randomTarget(i))

	def resetAll(self, targets=None):
		# reset root and character poses
		self.characterPose = np.array(self.characterPose)
		for i in range(self.num_slaves):
			self.rootPose[i] = Pose2d()
			self.rootPosePrev[i] = Pose2d()
			self.characterPose[i] = self.initialCharacterPose
			self.controlPrediction[i] = self.initialControlPrediction

		self.fallOverState = [0]*self.num_slaves
		self.fallOverCount = [0]*self.num_slaves
		self.standUpCount = [40]*self.num_slaves

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


	def loadNetworks(self, targets=None):
		# initialize rnn model
		self.model = RNNModel()

		# load network
		print("Loading rnn network from ../motions/{}/train/network".format(self.motion))
		self.model.restore("../motions/{}/train/network".format(self.motion))
		self.isModelLoaded = True

		self.resetAll(targets)

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
			# self.updateCharacterFallState(j)

			t = targets[j][:2]
			t = Pose2d(t)
			t = self.rootPose[j].relativePose(t)
			t = t.p
			t_len = math.sqrt(t[0]*t[0] + t[1]*t[1])

			t_angle = math.atan2(t[1], t[0]);
			t_angle = np.clip(t_angle, -0.4*np.pi, 0.4*np.pi)

			if(self.isWalk):
				clip_len = 60
			else:
				clip_len = 250

			# t_len = np.clip(t_len, 0.0, clip_len)
			if (t_len > 250):
				ratio = 250/t_len
				t[0] *= ratio
				t[1] *= ratio
			targets[j][0] = t[0]
			targets[j][1] = t[1]
			
			# t[0] = np.cos(t_angle)*t_len;
			# t[1] = np.sin(t_angle)*t_len;

			# if(self.fallOverState[j] == 0):
			# 	if(self.standUpCount[j] < 40):
			# 		targets[j][0] = 60
			# 		targets[j][1] = 0
			# 		targets[j][2] = self.target_height
			# 	else:
			# 		targets[j][0] = t[0]
			# 		targets[j][1] = t[1]
			# 		targets[j][2] = self.target_height
			# else:
			# 	if(self.fallOverCount[j] < 100):
			# 		targets[j][0] = 0#pred[0]
			# 		targets[j][1] = 0#pred[1]
			# 		targets[j][2] = 0#0#self.target_height
			# 	else:
			# 		if(RNNConfig.instance().useControlPrediction):
			# 			pred = RNNConfig.instance().xNormal.de_normalize_l(self.characterPose[j][-3:])
			# 			targets[j][0] = pred[0]# self.config.x_normal.mean[0]
			# 			targets[j][1] = pred[1]# self.config.x_normal.mean[1]
			# 			targets[j][2] = pred[2]#self.target_height #min(88, self.config.y_normal.de_normalize_l(self.current_y[j])[5]+20)
			# 		else:
			# 			targets[j][0] = 0# self.config.x_normal.mean[0]
			# 			targets[j][1] = 0# self.config.x_normal.mean[1]
			# 			targets[j][2] = self.target_height#min(88, self.config.y_normal.de_normalize_l(self.current_y[j])[5]+20)
			# 	# print(targets[j])
			targets[j] = RNNConfig.instance().xNormal.normalize_l(targets[j])
			print(targets[j])

		return np.array(targets, dtype=np.float32)




	# convert local to global
	def getGlobalPositions(self, output, index):
		output = output[2:] # first two elements is about foot contact
		# move root
		self.rootPosePrev[index] = deepcopy(self.rootPose[index])
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
		if self.isModelLoaded is False:
			self.loadNetworks()
		# if target is given, set target
		if targets is not None:
			self.targets = np.array(targets, dtype=np.float32)
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
		output = self.model.forwardOneStep(tf.convert_to_tensor(convertedTargets), tf.convert_to_tensor(self.characterPose), training=False)
		
		if RNNConfig.instance().useControlPrediction:
			self.characterPose = tf.slice(output, [0, 0], [-1, RNNConfig.instance().yDimension])
			self.controlPrediction = tf.slice(output, [0, RNNConfig.instance().yDimension], [-1, -1])
		else:
			self.characterPose = output

		# convert outputs to global coordinate
		output = self.characterPose.numpy()
		pose_list = []
		for j in range(self.num_slaves):
			pose = RNNConfig.instance().yNormal.de_normalize_l(output[j])
			pose = RNNConfig.instance().yNormal.get_data_with_zeros(pose)
			pose = self.getGlobalPositions(pose, j)
			pose_list.append(pose)
		return np.array(pose_list, dtype=np.float32)


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


	def saveState(self):
		self.model.saveState()
		self.savedCharacterPose = deepcopy(self.characterPose)
		self.savedRootPose = deepcopy(self.rootPose)
		self.savedRootPosePrev = deepcopy(self.rootPosePrev)

		self.savedFallOverState = deepcopy(self.fallOverState)
		self.savedFallOverCount = deepcopy(self.fallOverCount)
		self.savedStandUpCount = deepcopy(self.standUpCount)

	def loadState(self):
		self.model.loadState()
		self.characterPose = deepcopy(self.savedCharacterPose)
		self.rootPose = deepcopy(self.savedRootPose)
		self.rootPosePrev = deepcopy(self.savedRootPosePrev)

		self.fallOverState = deepcopy(self.savedFallOverState)
		self.fallOverCount = deepcopy(self.savedFallOverCount)
		self.standUpCount = deepcopy(self.savedStandUpCount)

	def updateCharacterFallState(self, index):
		root_height = RNNConfig.instance().yNormal.de_normalize_l(self.characterPose[index])[5]
		if( root_height < 30 ):
			if(self.fallOverState[index] == 1):
				self.fallOverState[index] = 2
				print("standing up")
			self.fallOverCount[index] %= 200
		elif( root_height < 70 ):
			if(self.fallOverState[index] == 0):
				self.fallOverState[index] = 1
				self.fallOverCount[index] = 0
				print("falling over")
		elif(root_height > 83):
			if(self.fallOverState[index] == 2):
				self.fallOverState[index] = 0
				self.standUpCount[index] = 0
				print("normal")

		if(self.fallOverState[index] != 0):
			self.fallOverCount[index] += 1
		else:
			self.standUpCount[index] += 1

		# print(self.fallOverCount[index])
		# print(self.standUpCount[index])

	def setTargetHeight(self, h):
		self.target_height = h

	def setDynamicPose(self, dpose, index=0):
		return
		print("a")
		self.characterPose = self.characterPose.numpy()
		new_rootPose = Pose2d().transform([dpose[3],dpose[0],-dpose[2]])
		pose_delta = self.rootPosePrev[index].relativePose(new_rootPose)
		new_characterPose = deepcopy(RNNConfig.instance().yNormal.de_normalize_l(self.characterPose[index]))

		new_characterPose[0:2] = dpose[52:54] 			# foot contact
		new_characterPose[2] = pose_delta.rotatedAngle() 	# root rotate
		new_characterPose[3:5] = pose_delta.p 				# root translate
		new_characterPose[5] = dpose[1]					# root height
		new_characterPose[6:63] = dpose[54:111] 			# positions
		new_characterPose[63:111] = dpose[4:52] 			# orientations

		self.rootPose[index] = new_rootPose
		self.rootPosePrev[index] = new_rootPose
		self.characterPose[index] = RNNConfig.instance().yNormal.normalize_l(new_characterPose)
		self.characterPose = tf.convert_to_tensor(self.characterPose)
