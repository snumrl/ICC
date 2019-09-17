from rnn.RNNController import RNNController
from rnn.RNNManager import RNNManager
from TrajectoryManager import TrajectoryManager

import numpy as np
from IPython import embed
import dphy
import os
import time

import CharacterConfigurations

os.environ['CUDA_VISIBLE_DEVICES'] = ''
if __name__ == "__main__":
	st = time.time()
	num_slaves = 1
	num_trajectories = 1
	env = dphy.Env('humanoid', num_slaves, True, True, True)
	# controller = RNNManager(num_slaves, motion="walk_extension")
	# o_c, o_n = controller.getReferences()
	# env.SetReferenceToTargets(o_c, o_n)
	frame = 8000
	traj_manager = TrajectoryManager(motion=CharacterConfigurations.motion)
	# traj_manager = TrajectoryManager(motion="basketball")
	# traj_manager = TrajectoryManager(motion="walk_extension")
	traj_manager.generateTrajectories(num_trajectories,frame, origin=True, origin_offset=0)
	print("generation done")
	print(time.time()-st)


	# traj_manager.generateTrajectories(4, frame)
	# traj_manager.generateTrajectories(4, frame)
	# traj_manager.generateTrajectories(4, frame)
	# traj_manager.generateTrajectories(4, frame)
	# traj_manager.generateTrajectories(4, frame)
	# traj_manager.generateTrajectories(4, frame)
	# traj_manager.generateTrajectories(4, frame)
	# traj_manager.generateTrajectories(4, frame)
	# traj_manager.load("../output/tf_foot_lr_0.001_rnn_no_goal_success/trajectories/", 4)
	# target = [[2000.0, 0.0], [-2000.0, 0.0], [0.0, 2000.0], [0.0, -2000.0]]
	# target = None
	indices = [0] * num_slaves
	for i in range(num_slaves):
		traj, goal_traj, index, t_index, t = traj_manager.getTrajectory(i);
		env.SetReferenceTrajectory(i, len(traj), traj)
		env.SetGoalTrajectory(i, len(traj), goal_traj)
		indices[i] = index
		frame = len(traj)
		# print(index)

	env.Resets(False)
	print(time.time()-st)
	for i in range(frame*10):
		# print(i)
		# o_c, o_n = controller.getReferences()
		# env.SetReferenceToTargets(o_c, o_n)

		for j in range(num_slaves):
			env.FollowReference(j)

		# if i == 1:
		# 	env.Resets(False)
		# 	# controller.resetAll()
		# 	# cur_target, next_target = controller.getReferences()
		# 	# env.SetReferenceToTargets(cur_target, next_target)
		# 	# env.Resets(False)
		# 	for j in range(num_slaves):
		# 		o_c, o_n = controller.reset(j)
		# 		env.SetReferenceToTarget(j, o_c, o_n)
		# 		# env.Reset(j, True)
		# 		env.FollowReference(j)
		# print(o[0][1])
	env.WriteRecords("./test_result/")
	traj_manager.save("./test_result/{}_".format(frame))
	print(time.time()-st)






# output : basketball
# [57 + 66 + 14 = 137
#	ball condition  : 8
# 	foot_contact 	: 2
# 	root_angle 		: 1
# 	root_translate 	: 2
# 	root_height		: 1
# 	joint_positions : 57 = 19 * 3
	# "Head_End",			: 14
	# "LeftHand",			: 17
	# "LeftFoot",			: 20
	# "LeftToe_End",		: 23
	# "RightHand",			: 26
	# "RightFoot",			: 29
	# "RightToe_End",		: 32
	# "LeftArm",			: 35
	# "RightArm",			: 38
	# "LeftForeArm",		: 41
	# "LeftLeg",			: 44
	# "RightForeArm",		: 47
	# "RightLeg",			: 50
	# "Spine",				: 53
	# "LeftHand_End",		: 56
	# "RightHand_End",		: 59
	# "Neck",				: 62
	# "LeftUpLeg",			: 65
	# "RightUpLeg",			: 68
# 	joint_angles 	: 66 = 22 * 3
	# "Head",				: 71
	# "Hips",				: 74
	# "LeftArm",			: 77
	# "LeftFoot",			: 80
	# "LeftForeArm",		: 83
	# "LeftHand",			: 86
	# "LeftLeg",			: 89
	# "LeftShoulder",		: 92
	# "LeftToe",			: 95
	# "LeftUpLeg",			: 98
	# "Neck",				: 101
	# "RightArm",			: 104
	# "RightFoot",			: 107
	# "RightForeArm",		: 110
	# "RightHand",			: 113
	# "RightLeg",			: 116
	# "RightShoulder",		: 119
	# "RightToe",			: 122
	# "RightUpLeg",			: 125
	# "Spine",				: 128
	# "Spine1",				: 131
	# "Spine2",				: 134
#   add? 1					: 137
# 	total			: 138
# ]

# output : walk
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

# output : zombie
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

# output : new character
# [
# 	foot_contact 	: 2
# 	root_angle 		: 1
# 	root_translate 	: 2
# 	root_height		: 1
# 	joint_angles 	: 48 = 16 * 3
# "Hips",				: 6
# "Spine",				: 9
# "Neck",				: 12
# "Head",				: 15
# "LeftArm",			: 18
# "LeftForeArm",		: 21
# "LeftHand",			: 24
# "RightArm",			: 27
# "RightForeArm",		: 30
# "RightHand",			: 33
# "LeftUpLeg",			: 36
# "LeftLeg",			: 39
# "LeftFoot",			: 42
# "RightUpLeg",			: 45
# "RightLeg",			: 48
# "RightFoot",			: 51
# 	total			: 138
# ]

# embed()
# exit()
