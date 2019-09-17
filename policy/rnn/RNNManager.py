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

		self.target_dist_lower = 30.0
		self.target_dist_upper = 150.0
		self.target_angle_upper = math.pi*0.5
		self.target_angle_lower = math.pi*(-0.5)
		self.targets = []
		self.target_of_target = []
		self.target_height = 88.

		self.traj_memory = []
		self.goal_memory = []

		for i in range(self.num_slaves):
			self.targets.append(self.randomTarget(i));
			self.target_of_target.append(self.randomTarget(i));

		# self.outputs = self.controller.step(self.targets)
		# for _ in range(100):
		# 	self.getReferences()

		self.traj_memory = []
		self.goal_memory = []

	def setTargetHeight(self, height):
		self.target_height = height
		self.controller.setTargetHeight(height)

	def setWalkRun(self, isWalk):
		self.controller.setWalkRun(isWalk)
		
	def randomTarget(self, index):
		if self.motion == "walk_extension":
			# target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			# target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# # if np.random.uniform(0,1) < 0.5:
			# # 	target_dist = 200
			# # 	target_angle = 0.0
			# # else:
			# # 	target_dist = 200
			# # 	target_angle = math.pi
			# local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
			# local_pose = Pose2d(local_target)
			# target = self.controller.pose[index].localToGlobal(local_pose).p
			target_x = np.random.uniform(-200, 200)
			target_y = np.random.uniform(-200, 200)
			target = [target_x, target_y]
		elif self.motion == "walk_new":
			target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# if np.random.uniform(0,1) < 0.5:
			# 	target_dist = 200
			# 	target_angle = 0.0
			# else:
			# 	target_dist = 200
			# 	target_angle = math.pi
			local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
			local_pose = Pose2d(local_target)
			target = self.controller.pose[index].localToGlobal(local_pose).p
		elif self.motion == "walkrun":

			# TODO: randomly generate velcoity input and convert it to position based input.
			target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# if np.random.uniform(0,1) < 0.5:
			# 	target_dist = 200
			# 	target_angle = 0.0
			# else:
			# 	target_dist = 200
			# 	target_angle = math.pi
			local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
			local_pose = Pose2d(local_target)
			target = self.controller.pose[index].localToGlobal(local_pose).p

		elif self.motion == "walkrunfall":

			# TODO: randomly generate velcoity input and convert it to position based input.
			target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# if np.random.uniform(0,1) < 0.5:
			# 	target_dist = 200
			# 	target_angle = 0.0
			# else:
			# 	target_dist = 200
			# 	target_angle = math.pi
			local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
			local_pose = Pose2d(local_target)
			target = self.controller.pose[index].localToGlobal(local_pose).p

			# TODO: PLZ check this line!
			target = target + [self.target_height]

		elif self.motion == "jog_roll":

			# TODO: randomly generate velcoity input and convert it to position based input.
			target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# if np.random.uniform(0,1) < 0.5:
			# 	target_dist = 200
			# 	target_angle = 0.0
			# else:
			# 	target_dist = 200
			# 	target_angle = math.pi
			local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
			local_pose = Pose2d(local_target)

		elif self.motion == "walkfall":
			target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# if np.random.uniform(0,1) < 0.5:
			# 	target_dist = 200
			# 	target_angle = 0.0
			# else:
			# 	target_dist = 200
			# 	target_angle = math.pi
			local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
			local_pose = Pose2d(local_target)
			target = self.controller.pose[index].localToGlobal(local_pose).p
			target = target + [self.target_height]
		elif self.motion == "punchfall":
			target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# if np.random.uniform(0,1) < 0.5:
			# 	target_dist = 200
			# 	target_angle = 0.0
			# else:
			# 	target_dist = 200
			# 	target_angle = math.pi
			local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
			local_pose = Pose2d(local_target)
			target = self.controller.pose[index].localToGlobal(local_pose).p
			target = target + [0]
		elif self.motion.startswith("walkfall_prediction"):
			target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# if np.random.uniform(0,1) < 0.5:
			# 	target_dist = 200
			# 	target_angle = 0.0
			# else:
			# 	target_dist = 200
			# 	target_angle = math.pi
			local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
			local_pose = Pose2d(local_target)
			target = self.controller.pose[index].localToGlobal(local_pose).p
			target = target + [self.target_height, 0]
		elif self.motion == "walkfall_variation":
			target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# if np.random.uniform(0,1) < 0.5:
			# 	target_dist = 200
			# 	target_angle = 0.0
			# else:
			# 	target_dist = 200
			# 	target_angle = math.pi
			local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
			local_pose = Pose2d(local_target)
			target = self.controller.pose[index].localToGlobal(local_pose).p
			target = target + [self.target_height, 0]
		elif self.motion == "basketball":
			target_action = [1,2,-1,-2,-1,-2,-1,-2,-1,-2]
			target_time = [1, 0]
			# target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			# target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)

			target_x = np.random.uniform(-500, 500)
			target_y = np.random.uniform(-500, 500)

			# local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
			# local_pose = Pose2d(local_target)
			# target_pos = self.controller.pose[index].localToGlobal(local_pose).p
			target_pos = [target_x, target_y]
			target_dir = self.controller.config.x_normal.mean[14:]
			target = target_action + target_time + target_pos + target_dir
		elif self.motion == "zombie":
			target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# if np.random.uniform(0,1) < 0.5:
			# 	target_dist = 200
			# 	target_angle = 0.0
			# else:
			# 	target_dist = 200
			# 	target_angle = math.pi
			local_target = [target_dist*math.cos(target_angle), target_dist*math.sin(target_angle)]
			local_pose = Pose2d(local_target)
			target = self.controller.pose[index].localToGlobal(local_pose).p

		elif self.motion == "gorilla":
			target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# if np.random.uniform(0,1) < 0.5:
			# 	target_dist = 200
			# 	target_angle = 0.0
			# else:
			# 	target_dist = 200
			# 	target_angle = math.pi
			local_target = [target_dist * math.cos(target_angle), target_dist * math.sin(target_angle)]
			local_pose = Pose2d(local_target)
			target = self.controller.pose[index].localToGlobal(local_pose).p
			# target_x = np.random.uniform(-200, 200)
			# target_y = np.random.uniform(-200, 200)
			# target = [target_x, target_y]

		else:
			print("py_code/rnn/RNNManager.py/randomTarget: use default parameters for random target generation")
			target_dist = np.random.uniform(self.target_dist_lower, self.target_dist_upper)
			target_angle = np.random.uniform(self.target_angle_lower, self.target_angle_upper)
			# if np.random.uniform(0,1) < 0.5:
			# 	target_dist = 200
			# 	target_angle = 0.0
			# else:
			# 	target_dist = 200
			# 	target_angle = math.pi
			local_target = [target_dist * math.cos(target_angle), target_dist * math.sin(target_angle)]
			local_pose = Pose2d(local_target)
			target = self.controller.pose[index].localToGlobal(local_pose).p
			# target_x = np.random.uniform(-200, 200)
			# target_y = np.random.uniform(-200, 200)
			# target = [target_x, target_y]

		return target

	def globalPosToTarget(self, t):
		if self.motion == "basketball" :
			target_action = [1,2,-1,-2,-1,-2,-1,-2,-1,-2]
			target_time = [1, 0]
			target_pos = t
			target_dir = self.controller.config.x_normal.mean[14:]
			target = target_action + target_time + target_pos + target_dir
		elif self.motion == "walk_extension":
			target = t
		elif self.motion == "walkrun":
			target = t
		elif self.motion == "jog_roll":
			target = t
		elif self.motion == "walk_new":
			target = t
		elif self.motion == "walkfall":
			target = t+[88.]
		elif self.motion == "walkrunfall":
			target = t+[88.]
		elif self.motion.startswith("walkfall_prediction"):
			target = t+[88., 0]
		elif self.motion == "walkfall_variation":
			target = t + [88., 0]
		elif self.motion == "zombie":
			target = t
		elif self.motion == "gorilla":
			target = t
		else:
			target = t
			print("py_code/rnn/RNNManager.py/globalPosToTarget: use default target")
		# print(target)

		return target


	def getReferences(self, targets=None):
		if targets is not None:
			for i in range(self.num_slaves):
				self.targets[i] = deepcopy(targets[i])
		else:
			if self.motion == "basketball":				
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i][12:14]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<100*100):
						self.target_of_target[i] = self.randomTarget(i)

					# dx = self.target_of_target[i][12] - self.targets[i][12]
					# dy = self.target_of_target[i][13] - self.targets[i][13]

					# l = math.sqrt(dx*dx+dy*dy)
					# if(l > 12):
					# 	dx = dx*12/l
					# 	dy = dy*12/l

					# self.targets[i][12] += dx
					# self.targets[i][13] += dy
					self.targets[i][12] = self.target_of_target[i][12]
					self.targets[i][13] = self.target_of_target[i][13]
			elif self.motion == "walk_extension":
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<60*60):
						self.target_of_target[i] = self.randomTarget(i)

					self.targets[i] = self.target_of_target[i]
			elif self.motion == "walk_new":
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<60*60):
						self.target_of_target[i] = self.randomTarget(i)

					self.targets[i] = self.target_of_target[i]
			elif self.motion == "jog_roll":
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<60*60):
						self.target_of_target[i] = self.randomTarget(i)

					self.targets[i] = self.target_of_target[i]
			elif self.motion == "walkrun":
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<60*60):
						self.target_of_target[i] = self.randomTarget(i)

					self.targets[i] = self.target_of_target[i]

			elif self.motion == "walkrunfall":
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<100*100):
						self.target_of_target[i] = self.randomTarget(i)

					self.targets[i] = self.target_of_target[i]

			elif self.motion == "walkfall":			
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i][:2]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<100*100):
						self.target_of_target[i] = self.randomTarget(i)

					# dx = self.target_of_target[i][12] - self.targets[i][12]
					# dy = self.target_of_target[i][13] - self.targets[i][13]

					# l = math.sqrt(dx*dx+dy*dy)
					# if(l > 12):
					# 	dx = dx*12/l
					# 	dy = dy*12/l

					# self.targets[i][12] += dx
					# self.targets[i][13] += dy
					self.targets[i][0] = self.target_of_target[i][0]
					self.targets[i][1] = self.target_of_target[i][1]
			elif self.motion == "punchfall":
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i][:2]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<100*100):
						self.target_of_target[i] = self.randomTarget(i)

					# dx = self.target_of_target[i][12] - self.targets[i][12]
					# dy = self.target_of_target[i][13] - self.targets[i][13]

					# l = math.sqrt(dx*dx+dy*dy)
					# if(l > 12):
					# 	dx = dx*12/l
					# 	dy = dy*12/l

					# self.targets[i][12] += dx
					# self.targets[i][13] += dy
					self.targets[i][0] = self.target_of_target[i][0]
					self.targets[i][1] = self.target_of_target[i][1]
			elif self.motion.startswith("walkfall_prediction"):
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i][:2]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<100*100):
						self.target_of_target[i] = self.randomTarget(i)

					# dx = self.target_of_target[i][12] - self.targets[i][12]
					# dy = self.target_of_target[i][13] - self.targets[i][13]

					# l = math.sqrt(dx*dx+dy*dy)
					# if(l > 12):
					# 	dx = dx*12/l
					# 	dy = dy*12/l

					# self.targets[i][12] += dx
					# self.targets[i][13] += dy
					self.targets[i][0] = self.target_of_target[i][0]
					self.targets[i][1] = self.target_of_target[i][1]
			elif self.motion == "walkfall_variation":
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i][:2]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<100*100):
						self.target_of_target[i] = self.randomTarget(i)

					# dx = self.target_of_target[i][12] - self.targets[i][12]
					# dy = self.target_of_target[i][13] - self.targets[i][13]

					# l = math.sqrt(dx*dx+dy*dy)
					# if(l > 12):
					# 	dx = dx*12/l
					# 	dy = dy*12/l

					# self.targets[i][12] += dx
					# self.targets[i][13] += dy
					self.targets[i][0] = self.target_of_target[i][0]
					self.targets[i][1] = self.target_of_target[i][1]
			elif self.motion == "zombie":
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<60*60):
						self.target_of_target[i] = self.randomTarget(i)

					self.targets[i] = self.target_of_target[i]

			elif self.motion == "gorilla":
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<60*60):
						self.target_of_target[i] = self.randomTarget(i)

					self.targets[i] = self.target_of_target[i]
			else:
				print("py_code/rnn/RNNManager.py/getReferences: use default target generation")
				for i in range(self.num_slaves):
					cur_pose = self.controller.pose[i].p
					target = self.target_of_target[i]
					dx = cur_pose[0] - target[0]
					dy = cur_pose[1] - target[1]
					if(dx*dx+dy*dy<60*60):
						self.target_of_target[i] = self.randomTarget(i)

					self.targets[i] = self.target_of_target[i]


		old = self.outputs
		new = self.controller.step(self.targets)
		self.outputs = new

		self.traj_memory.append(old)
		t = self.getTargets()
		if self.motion == "basketball":
			t = t[:,12:14]
		elif self.motion == "walkfall":
			t = t[:,:2]
		elif self.motion == "walkrunfall":
			t = t[:,:2]
		elif self.motion.startswith("walkfall_prediction"):
			t = t[:,:2]
		elif self.motion == "walkfall_variation":
			t = t[:,:2]
		self.goal_memory.append(t)

		return old, new

	def saveMemory(self):
		trajectories = np.asarray([*zip(*self.traj_memory)])
		target_trajectories = np.asarray([*zip(*self.goal_memory)])

		np.save("traj.npy", [trajectories[0][::2]])
		np.save("goal.npy", [target_trajectories[0][::2]])

	def getTrajectory(self, frame=2000, targets=None):
		self.resetAll(targets)
		trajectories = []
		target_trajectories = []
		for _ in  range(frame):
			tr, _ = self.getReferences(targets)
			trajectories.append(tr)
			t = self.getTargets()
			if self.motion == "basketball":
				t = t[:,12:14]
			elif self.motion == "walkfall":
				t = t[:,:2]
			elif self.motion == "walkrunfall":
				t = t[:,:2]
			elif self.motion.startswith("walkfall_prediction"):
				t = t[:,:2]
			target_trajectories.append(t)

		trajectories = np.asarray([*zip(*trajectories)])
		target_trajectories = np.asarray([*zip(*target_trajectories)])

		return trajectories, target_trajectories

	def getOriginalTrajectory(self, frame, origin_offset=0):
		return self.controller.getOriginalTrajectory(frame, origin_offset)

	# This function load original traj. and original goal traj. without transfer goal from local to global.
	def getOriginalTrajectoryWithLocalGoal(self, frame, origin_offset=0):
		return self.controller.getOriginalTrajectoryWithLocalGoal(frame, origin_offset)

	def getTargets(self):
		return np.asarray(self.targets, dtype=np.float32)

	def getClippedTargets(self):
		return self.controller.get_clipped_targets()


	def resetAll(self, targets=None):
		self.controller.resetAll()
		if targets is not None:
			for i in range(self.num_slaves):
				self.targets[i] = deepcopy(targets[i])
		self.outputs = self.controller.step(self.targets)
		for _ in range(100):
			self.getReferences(targets)

		self.traj_memory = []
		self.goal_memory = []

	def saveState(self):
		self.controller.saveState()
	def loadState(self):
		self.controller.loadState()
	def saveStateInterval(self):
		self.controller.saveStateInterval()
	def loadStateInterval(self):
		self.controller.loadStateInterval()
	def setNewY(self, index, y):
		self.controller.setNewY(index, y)




# if __name__ == "__main__":
# 	env = dphy.Env('foot', 4, False)
# 	controller = RNNController("walk", 4)
# 	target = [[500.0, 0.0], [-500.0, 0.0], [0.0, 500.0], [0.0, -500.0]]
# 	for i in range(1000):
# 		o = np.asarray(controller.step(target),dtype=np.float32)
# 		# print("")
# 		# print(o)
# 		for j in range(4):
# 			env.SetReference(j, o[j])
# 			env.FollowReference(j)
# 		# print(o[0][1])
# 	env.WriteRecords("./test_records/")






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