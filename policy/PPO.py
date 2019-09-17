import math
import random
import time
import os
import sys
import datetime

from collections import namedtuple
from collections import deque
from itertools import count
from copy import deepcopy

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
from environment_wrapper import environment
from IPython import embed


from ReplayBuffer import ReplayBuffer
from ReplayBuffer import Transition
from ReplayBuffer import Episode
from RunningMeanStd import RunningMeanStd

from rnn.RNNController import RNNController
from rnn.RNNManager import RNNManager
from TrajectoryManager import TrajectoryManager

import CharacterConfigurations

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


activ = tf.nn.relu
kernel_initialize_func = tf.contrib.layers.xavier_initializer()
# kernel_initialize_func = ortho_init(0.1)
# class DiagGaussianPd(Pd):
#     def __init__(self, flat):
#         self.flat = flat
#         mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
#         self.mean = mean
#         self.logstd = logstd
#         self.std = tf.exp(logstd)
#     def neglogp(self, x):
#         return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
#                + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
#                + tf.reduce_sum(self.logstd, axis=-1)
#     def sample(self):
#         return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

import matplotlib
import matplotlib.pyplot as plt
plt.ion()

actor_layer_size = 1024
critic_layer_size = 512
initial_state_layer_size = 512
l2_regularizer_scale = 0.0
regularizer = tf.contrib.layers.l2_regularizer(l2_regularizer_scale)


def Plot(y_list, title,num_fig=1,ylim=True,path=None):
	plt.figure(num_fig, clear=True, figsize=(5.5, 4))
	plt.title(title)

	i = 0
	for y in y_list:
		plt.plot(y[0], label=y[1])
		i+= 1

	plt.legend(loc=2)
	plt.show()
	if ylim:
		plt.ylim([0,1])
	plt.pause(0.001)
	if path is not None:
		plt.savefig(path, format="png")

class AdaptiveInitialState(object):
	def __init__(self, sess, scope, state, num_pos_and_vel):
		self.sess = sess
		self.scope = scope
		# self.sigma = np.array([1.]*num_actions, dtype=np.float32)

		self.mean, self.logstd, self.std = self.CreateNetwork(state, num_pos_and_vel, False, None)
		# self.dist = tf.distributions.Normal(loc=self.mean, scale=self.sigma)
		# self.policy = self.dist.sample()
		# self.logprob = tf.reduce_sum(self.dist.log_prob(self.policy+1e-5),1)
		self.policy = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
		self.neglogprob = self.neglogp(self.policy)


	def neglogp(self, x):
		return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) + tf.reduce_sum(self.logstd, axis=-1)




	def CreateNetwork(self, state, num_pos_and_vel, reuse, is_training):
		with tf.variable_scope(self.scope, reuse=reuse):
			L1 = tf.layers.dense(state,initial_state_layer_size,activation=activ,name='L1',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			L2 = tf.layers.dense(L1,initial_state_layer_size,activation=activ,name='L2',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			mean = tf.layers.dense(L2,num_pos_and_vel,name='mean',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			# sigma = tf.layers.dense(L2,num_actions,name='sigma',
			# 	bias_initializer=tf.contrib.layers.xavier_initializer(),
			#      kernel_initializer=tf.contrib.layers.xavier_initializer()
			# )

			# sigma = tf.nn.softplus(sigma + 1e-5)
			self.logstdvar = logstd = tf.get_variable(name='std', 
				shape=[num_pos_and_vel], initializer=tf.constant_initializer(0)
			)

			# flat = tf.concat([mean, mean * 0.0 + logstd], axis=1)
			# mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
			sigma = tf.exp(logstd)
			#sigma = tf.get_variable(name='std', shape=[1, num_actions], initializer=tf.ones_initializer())

			return mean, logstd, sigma

	def GetInitialStateDelta(self, states):
		with tf.variable_scope(self.scope):
			action, neglogp = self.sess.run([self.policy, self.neglogprob], feed_dict={'state:0':states})
			return action, neglogp

	def GetMeanInitialStateDelta(self, states):
		with tf.variable_scope(self.scope):
			action = self.sess.run([self.mean], feed_dict={'state:0':states})
			return action[0]



class Actor(object):
	def __init__(self, sess, scope, state, num_actions):
		self.sess = sess
		self.scope = scope
		# self.sigma = np.array([1.]*num_actions, dtype=np.float32)

		self.mean, self.logstd, self.std = self.CreateNetwork(state, num_actions, False, None)
		# self.dist = tf.distributions.Normal(loc=self.mean, scale=self.sigma)
		# self.policy = self.dist.sample()
		# self.logprob = tf.reduce_sum(self.dist.log_prob(self.policy+1e-5),1)
		self.policy = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
		self.neglogprob = self.neglogp(self.policy)


	def neglogp(self, x):
		return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) + tf.reduce_sum(self.logstd, axis=-1)




	def CreateNetwork(self, state, num_actions, reuse, is_training):
		with tf.variable_scope(self.scope, reuse=reuse):
			L1 = tf.layers.dense(state,actor_layer_size,activation=activ,name='L1',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			L2 = tf.layers.dense(L1,actor_layer_size,activation=activ,name='L2',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			L3 = tf.layers.dense(L2,actor_layer_size,activation=activ,name='L3',
				kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			L4 = tf.layers.dense(L3,actor_layer_size,activation=activ,name='L4',
				kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			mean = tf.layers.dense(L4,num_actions,name='mean',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			# sigma = tf.layers.dense(L2,num_actions,name='sigma',
			# 	bias_initializer=tf.contrib.layers.xavier_initializer(),
			#      kernel_initializer=tf.contrib.layers.xavier_initializer()
			# )

			# sigma = tf.nn.softplus(sigma + 1e-5)
			self.logstdvar = logstd = tf.get_variable(name='std', 
				shape=[num_actions], initializer=tf.constant_initializer(0)
			)

			# flat = tf.concat([mean, mean * 0.0 + logstd], axis=1)
			# mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
			sigma = tf.exp(logstd)
			#sigma = tf.get_variable(name='std', shape=[1, num_actions], initializer=tf.ones_initializer())

			return mean, logstd, sigma

	def GetAction(self, states):
		with tf.variable_scope(self.scope):
			action, logprob = self.sess.run([self.policy, self.neglogprob], feed_dict={'state:0':states})
			# print("sigma")
			# print(s)
			return action, logprob

	def GetMeanAction(self, states):
		with tf.variable_scope(self.scope):
			action = self.sess.run([self.mean], feed_dict={'state:0':states})
			# print("sigma")
			# print(s)
			return action[0]

	

	def GetVariable(self, trainable_only=False):
		if trainable_only:
			return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
		else:
			return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)


class Critic(object):
	def __init__(self, sess, scope, state):
		self.sess = sess
		self.scope = scope
		self.value = self.CreateNetwork(state, False, None)

	def CreateNetwork(self, state, reuse, is_training):	
		with tf.variable_scope(self.scope, reuse=reuse):
			L1 = tf.layers.dense(state,critic_layer_size,activation=activ,name='L1',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			L2 = tf.layers.dense(L1,critic_layer_size,activation=activ,name='L2',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			# L3 = tf.layers.dense(L2,512,activation=activ,name='L3',
	  #           kernel_initializer=kernel_initialize_func
			# )

			# L4 = tf.layers.dense(L3,512,activation=activ,name='L4',
	  #           kernel_initializer=kernel_initialize_func
			# )

			out = tf.layers.dense(L2,1,name='out',
	            kernel_initializer=kernel_initialize_func,
	            kernel_regularizer=regularizer
			)

			return out[:,0]

	def GetValue(self, states):
		with tf.variable_scope(self.scope):
			return self.sess.run(self.value, feed_dict={'state:0':states})

	def GetVariable(self, trainable_only=False):
		if trainable_only:
			return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
		else:
			return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

class PPO(object):
	def __init__(self):
		print("PPO initializing...")
		random.seed(int(time.time()))
		np.random.seed(int(time.time()))
		tf.set_random_seed(int(time.time()))
		self.start_time = time.time()
		self.sim_time = 0
		self.train_time = 0


	def InitializeForViewer(self, num_state, num_action, pretrain=None, use_adaptive_initial_state=False):
		config = tf.ConfigProto()
		config.intra_op_parallelism_threads = 4
		config.inter_op_parallelism_threads = 4
		self.sess = tf.Session(config=config)
		self.use_adaptive_initial_state = use_adaptive_initial_state

		self.num_trajectories = 1
		self.traj_frame = 200
		self.motion = CharacterConfigurations.motion

		self.num_state = num_state
		self.num_action = num_action

		self.pretrain = pretrain


		self.RMS = RunningMeanStd(shape=(self.num_state))
		rms_dir = "../rms/"+str(self.num_state)+"/"
		if os.path.exists(rms_dir+"mean.npy"):
			print("load RMS parameters")
			self.RMS.mean = np.load(rms_dir+"mean.npy")
			self.RMS.var = np.load(rms_dir+"var.npy")
			self.RMS.count = 16384*200

		print("Num state : {}, num action : {}".format(self.num_state, self.num_action))

		self.state = tf.placeholder(tf.float32, shape=[None,self.num_state], name='state')
		pstate = tf.to_float(self.state)
		self.actor = Actor(self.sess, 'Actor_new', pstate, self.num_action)
		# self.actor_old = Actor(self.sess, 'Actor_old', state, self.num_action)
		self.critic = Critic(self.sess, 'Critic', pstate)


		self.num_slaves = 2
		self.rnn_manager = RNNManager(self.num_slaves, self.motion)
		self.sess.run(tf.global_variables_initializer())

		# self.trajectory_manager = TrajectoryManager(self.motion)
		# self.trajectory_manager.generateTrajectories(self.num_trajectories, self.traj_frame)

		if self.pretrain is not None:
			self.LoadPreTrainedVariables(self.pretrain)

	def ResetRNN(self, target_x, target_y):
		t = self.rnn_manager.globalPosToTarget([target_x, target_y])
		target = []
		for i in range(self.num_slaves):
			target.append(deepcopy(t))

		self.rnn_manager.resetAll(target)

	def GetLastReferenceCharacterPose2D(self):
		# for converting in C++ during interactive mode
		# print(np.array(self.rnn_manager.controller.pose[0].toArray(), dtype=np.float32))
		return np.array(self.rnn_manager.controller.pose[0].toArray(), dtype=np.float32)

	## HS) Be Called 33 times in Initializing at Interative mode...

	def SetTargetHeight(self, height):
		self.rnn_manager.setTargetHeight(height)

	def SetWalkRun(self, isWalk):
		self.rnn_manager.setWalkRun(isWalk)

	def SaveMemory(self):
		self.rnn_manager.saveMemory()

	def GetNextPos(self, target_x, target_y):
		t = self.rnn_manager.globalPosToTarget([target_x, target_y])
		target = []
		for i in range(self.num_slaves):
			target.append(deepcopy(t))
		# using current (p,v) to calibrate target...



		_, pos = self.rnn_manager.getReferences(target)
		pos = np.array(pos)
		return pos

	def GetGoal(self):
		# print(self.rnn_manager.getClippedTargets())
		return self.rnn_manager.getClippedTargets()

	def SaveState(self):
		self.rnn_manager.saveState()

	def LoadState(self):
		self.rnn_manager.loadState()

	def SaveStateInterval(self):
		self.rnn_manager.saveStateInterval()

	def LoadStateInterval(self):
		self.rnn_manager.loadStateInterval()

	def SetNewY(self, index, y):
		self.rnn_manager.setNewY(index, y)


	def OriginToRandomRate(self):
		return

	def InitializeForTraining(self, env_name, num_slaves=8, use_trajectory=True, useBothOriginNRandom= False,
		learning_rate=2e-4, num_trajectories=32, 
		origin=False, frame=2000, origin_offset=0,
		num_time_piecies=20,
		gamma=0.99, lambd=0.95,
		batch_size=512, steps_per_iteration=8192,
		use_adaptive_initial_state=True,
		use_evaluation=True,
		motion="walk", detail="", pretrain=""):

		config = tf.ConfigProto()
		config.intra_op_parallelism_threads = num_slaves
		config.inter_op_parallelism_threads = num_slaves
		# config.gpu_options.allow_growth = True
		self.useBothOriginNRandom= useBothOriginNRandom
		self.use_adaptive_initial_state = use_adaptive_initial_state
		self.sess = tf.Session(config=config)

		self.num_slaves = num_slaves
		self.env_name = env_name
		self.use_trajectory = use_trajectory
		self.use_terminal = True
		self.use_discrete_reference = True

		self.use_evaluation = use_evaluation
		self.motion = motion
		self.Env = environment()

		self.num_state = self.Env.getStateSize()
		self.num_action = self.Env.getActionSize()

		self.RMS = RunningMeanStd(shape=(self.num_state)) ##SM) state normalize
		rms_dir = "../rms/"+str(self.num_state)+"/"
		if os.path.exists(rms_dir+"mean.npy"):
			print("load RMS parameters")
			self.RMS.mean = np.load(rms_dir+"mean.npy")
			self.RMS.var = np.load(rms_dir+"var.npy")
			self.RMS.count = 16384*200

		self.actionRMS = RunningMeanStd(shape=(self.num_action))

		print("Num state : {}, num action : {}".format(self.num_state, self.num_action))

		self.gamma = gamma
		self.lambd = lambd

		# self.batch_size = 4096
		# self.steps_per_iteration = 32768
		self.batch_size = batch_size
		self.steps_per_iteration = steps_per_iteration

		self.epsilon = 0.2  ##SM) ppo clipping constant- never modified(?)
		self.learning_rate = learning_rate ##SM) learning_rate of actor
		self.learning_rate_decay = 0.9993
		self.learning_rate_critic = 0.001


		# for statistics
		self.num_evaluation = 0
		self.num_episodes_total = 0
		self.num_transitions_total = 0

		self.num_episodes_per_iteration = 0
		self.num_transitions_per_iteration = 0
		self.max_episode_length = 0

		self.iteration_count = 0

		self.detail = detail

		self.total_rewards = []
		self.total_rewards_by_parts = np.array([[]]*5)
		self.mean_rewards = []
		self.transition_per_episodes = []
		self.noise_records = []

		self.evaluation_total_rewards = []
		self.evaluation_total_rewards_by_parts = np.array([[]]*5)
		self.evaluation_mean_rewards = []
		self.evaluation_transition_per_episodes = []

		self.total_episodes = []
		self.total_episodes_idx = []

		self.state = tf.placeholder(tf.float32, shape=[None,self.num_state], name='state')
		pstate = tf.to_float(self.state)
		self.actor = Actor(self.sess, 'Actor_new', pstate, self.num_action)
		# self.actor_old = Actor(self.sess, 'Actor_old', state, self.num_action)
		self.critic = Critic(self.sess, 'Critic', pstate)

		self.BuildOptimize()

		
		if(self.useBothOriginNRandom):
			self.num_trajectories= num_trajectories+1
		else:
			self.num_trajectories = num_trajectories

		self.traj_frame = frame

		if self.use_trajectory:
			self.trajectory_manager = TrajectoryManager(self.motion, num_time_piecies)  ##SM) RNN generate and manage ..
			print("Elapsed time : {:.2f}s".format(time.time() - self.start_time))
			self.trajectory_manager.generateTrajectories(self.num_trajectories, self.traj_frame, origin, origin_offset, self.useBothOriginNRandom)
			print("Elapsed time : {:.2f}s".format(time.time() - self.start_time))
		else: ##SM) e.g- interactive
			self.reference_manager = RNNManager(self.num_slaves, self.motion)

		self.traj_indices = [None]*self.num_slaves
		self.time_indices = [None]*self.num_slaves
		self.sess.run(tf.global_variables_initializer())
		self.trajectory_manager.setPPO(self)
		# self.trajectory_manager.max_episode_length.fill(0)
		# print("Grnerating trajectory...")
		# self.trajectory_manager.generateTrajectories(self.num_trajectories, self.traj_frame)
		traj, g_traj, index, t_index, timet = self.trajectory_manager.getTrajectory()  ##SM) traj, g_traj, index: constant now(legacy of multiple trajectories)
		self.traj_frame = len(traj)
		self.Env.setReferenceTrajectories(len(traj), traj)
		for i in range(self.num_slaves):
			self.traj_indices[i] = index

		self.sess.run(tf.global_variables_initializer())

		# replay buffer
		self.replay_buffer = ReplayBuffer()

		if self.use_adaptive_initial_state:
			self.adaptive_initial_state_tuples = []
		save_list = tf.trainable_variables()
		self.saver = tf.train.Saver(var_list=save_list,max_to_keep=1)
		self.smax = 0
		self.rmax = 0

		# pretrain
		self.is_pretrained = False
		self.pretrain = pretrain
		if self.pretrain != "":
			self.LoadPreTrainedVariables(self.pretrain)
			self.is_pretrained = True


		self.GenerateName()


		# Initialize enviroment
		self.traj_regeneration_period = 200  ##SM) legacy (yes)
		# if self.use_trajectory:
		# 	traj, goal_traj = self.reference_manager.getTrajectory(self.traj_frame)
		# 	for i in range(self.num_slaves):
		# 		self.Env.SetReferenceTrajectory(i, self.traj_frame, traj[i])
		# 		self.Env.SetGoalTrajectory(i, self.traj_frame, goal_traj[i])
		# else:
		# 	cur_target, next_target = self.reference_manager.getReferences()
		# 	self.Env.SetReferenceToTargets(cur_target, next_target)
		# 	self.Env.SetGoals(self.reference_manager.getTargets())
		# self.Env.Resets(False)


		print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print("Elapsed time : {:.2f}s".format(time.time()-self.start_time))
		print("test_name : {}".format(self.detail))
		print("env_name : {}".format(self.env_name))
		print("motion : {}".format(self.motion))
		print("num_slaves : {}".format(self.num_slaves))
		print("use_trajectory : {}".format(self.use_trajectory))
		print("learning_rate : {}".format(self.learning_rate))
		print("gamma : {}".format(self.gamma))
		print("lambd : {}".format(self.lambd))
		print("batch_size : {}".format(self.batch_size))
		print("steps_per_iteration : {}".format(self.steps_per_iteration))
		print("clip ratio : {}".format(self.epsilon))
		print("trajectory frame : {}".format(self.traj_frame))
		print("num_trajectories : {}".format(self.num_trajectories))

		if  detail == "Run":
			self.directory = './run_result/'
		else:
			if not os.path.exists("../output/"):
				os.mkdir("../output/")
			self.directory = '../output/'+self.name+'/'
		if not os.path.exists(self.directory):
			os.mkdir(self.directory)
		directory = self.directory + "rms/"
		if not os.path.exists(directory):
			os.mkdir(directory)
		directory = directory + "cur/"
		if not os.path.exists(directory):
			os.mkdir(directory)

		out = open(self.directory+"parameters", "w")
		out.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S\n"))
		out.write("env_name : {}\n".format(self.env_name))
		out.write("motion : {}\n".format(self.motion))
		out.write("num_slaves : {}\n".format(self.num_slaves))
		out.write("num state : {}\n".format(self.num_state))
		out.write("num action : {}\n".format(self.num_action))
		out.write("use_trajectory : {}\n".format(self.use_trajectory))
		out.write("learning_rate : {}\n".format(self.learning_rate))
		out.write("gamma : {}\n".format(self.gamma))
		out.write("lambd : {}\n".format(self.lambd))
		out.write("batch_size : {}\n".format(self.batch_size))
		out.write("steps_per_iteration : {}\n".format(self.steps_per_iteration))
		out.write("clip ratio : {}\n".format(self.epsilon))
		out.write("pretrain : {}\n".format(self.pretrain))
		out.write("trajectory frame : {}\n".format(self.traj_frame))
		out.write("num_trajectories : {}\n".format(self.num_trajectories))
		out.close()

		# pre make results file
		out = open(self.directory+"results", "w")
		out.close()

		# save trajectories
		self.traj_count = 0
		if self.use_trajectory:
			directory = self.directory + "trajectories/"
			if not os.path.exists(directory):
				os.mkdir(directory)
			self.trajectory_manager.save(directory+"{}_{}_".format(self.traj_count, self.traj_frame))

	def GenerateName(self):
		# generate name
		if self.detail == "":
			self.name = self.motion
			# self.name = "tf_"+self.env_name
			self.name += "_lr_"+str(self.learning_rate)
			if self.is_pretrained:
				self.name += "_pretrained"
		else:
			self.name = self.detail

	def LoadPreTrainedVariables(self, path):
		print("Loading parameters from {}".format(path))

		def get_tensors_in_checkpoint_file(file_name):
			varlist=[]
			var_value =[]
			reader = pywrap_tensorflow.NewCheckpointReader(file_name)
			var_to_shape_map = reader.get_variable_to_shape_map()
			for key in sorted(var_to_shape_map):
				varlist.append(key)
				var_value.append(reader.get_tensor(key))
			return (varlist, var_value)


		saved_variables, saved_values = get_tensors_in_checkpoint_file(path)
		saved_dict = {n : v for n, v in zip(saved_variables, saved_values)}
		restore_op = []
		for v in tf.trainable_variables():
			if v.name[:5] != "Actor" and v.name[:6] != "Critic":
				continue
			if v.name[:-2] in saved_dict:
				saved_v = saved_dict[v.name[:-2]]
				if v.shape == saved_v.shape:
					print("Restore {}".format(v.name[:-2]))
					restore_op.append(v.assign(saved_v))
				# elif v.shape[1] == saved_v.shape[1] and v.shape[0] == saved_v.shape[0]-1:					
				# 	print("Restore {}, trimmed".format(v.name[:-2]))
				# 	restore_op.append(v.assign(saved_v[:-1,:]))
				# else:
				# 	print("Restore {}, adjusted".format(v.name[:-2]))
				# 	if v.shape[0] > saved_v.shape[0]:
				# 	elif v.shape[0] < saved_v

		restore_op = tf.group(*restore_op)
		self.sess.run(restore_op)
		self.is_pretrained = True


	def GetTrajectoryForCpp(self):
		traj, g_traj, index, t_index, t = self.trajectory_manager.getTrajectory(0)
		self.traj = traj
		self.g_traj = g_traj

		return self.traj

	def GetGoalTrajectoryForCpp(self):
		return self.g_traj


	def GetMeanActionForCpp(self, state):
		state = np.array([state])
		state = self.RMS.apply(state)
		action = self.actor.GetMeanAction(state)
		return action


	def setActions(self, actions):
		self.Env.setActions(actions)

	def SetAction(self, actions, num):
		self.Env.SetAction(actions, num)



	def BuildOptimize(self):
		with tf.variable_scope('Optimize'):
			self.action = tf.placeholder(tf.float32, shape=[None,self.num_action], name='action')
			self.TD = tf.placeholder(tf.float32, shape=[None], name='TD')
			self.GAE = tf.placeholder(tf.float32, shape=[None], name='GAE')
			self.old_logprobs = tf.placeholder(tf.float32, shape=[None], name='old_logprobs')
			self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='learning_rate')

			self.cur_neglogp = self.actor.neglogp(self.action)
			self.ratio = tf.exp(self.old_logprobs-self.cur_neglogp)
			clipped_ratio = tf.clip_by_value(self.ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)

			surrogate = -tf.reduce_mean(tf.minimum(self.ratio*self.GAE, clipped_ratio*self.GAE))
			value_loss = tf.reduce_mean(tf.square(self.critic.value - self.TD))
			reg_l2_actor = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'Actor')
			reg_l2_critic = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'Critic')

			loss_actor = surrogate + tf.reduce_sum(reg_l2_actor)
			loss_critic = value_loss + tf.reduce_sum(reg_l2_critic)

		actor_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
		grads, params = zip(*actor_trainer.compute_gradients(loss_actor));
		grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
		grads_and_vars = list(zip(grads, params))
		self.actor_train_op = actor_trainer.apply_gradients(grads_and_vars)


		critic_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_critic)
		grads, params = zip(*critic_trainer.compute_gradients(loss_critic));
		grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
		grads_and_vars = list(zip(grads, params))
		self.critic_train_op = critic_trainer.apply_gradients(grads_and_vars)


		if self.use_adaptive_initial_state:
			adaptive_initial_trainer = tf.train.AdamOptimizer(learning_rate=0.001)
			grads, params = zip(*adaptive_initial_trainer.compute_gradients(adaptive_initial_surrogate));
			grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
			
			grads_and_vars = list(zip(grads, params))
			self.adaptive_initial_train_op = adaptive_initial_trainer.apply_gradients(grads_and_vars)


	def Optimize(self):
		self.ComputeTDandGAE()
		if len(self.replay_buffer.buffer) < self.batch_size:
			return

		transitions = np.array(self.replay_buffer.buffer)
		GAE = np.array(Transition(*zip(*transitions)).GAE)
		GAE = (GAE - GAE.mean())/(GAE.std() + 1e-5)

		# state = np.array(transitions.s)
		# TD = np.array(transitions.TD)
		# action = np.array(transitions.a)
		# logprob = np.array(transitions.logprob)

		ind = np.arange(len(GAE))
		for _ in range(1):
			np.random.shuffle(ind)

			for s in range(int(len(ind)//self.batch_size)):
				selectedIndex = ind[s*self.batch_size:(s+1)*self.batch_size]
				selectedTransitions = transitions[selectedIndex]

				batch = Transition(*zip(*selectedTransitions))

				# GAE = np.array(batch.GAE)
				# GAE = (GAE - GAE.mean())/(GAE.std() + 1e-5)
				self.sess.run([self.actor_train_op, self.critic_train_op], 
					feed_dict={
						self.state:batch.s, 
						self.TD:batch.TD, 
						self.action:batch.a, 
						self.old_logprobs:batch.logprob, 
						self.GAE:GAE[selectedIndex],
						self.learning_rate_ph:self.learning_rate
					}
				)
				# variables = self.sess.run(tf.global_variables())
				# if any(np.isnan(a).any() for a in variables):
				# 	print("Variable Nan")
				# 	embed()
				# 	exit()

		# update adaptive initial


	def ComputeTDandGAE(self):
		self.replay_buffer.Clear()
		for epi, idx, t_idx in zip(self.total_episodes, self.total_episodes_idx, self.total_time_idx):
			data = epi.GetData()
			size = len(data)
			# update trajectory rewards and max length
			epi_rew = self.GetEpiReward(epi, gamma=self.gamma)
			epi_len = size
			# self.trajectory_manager.updateEpiReward(idx, t_idx[0], epi_len, t_idx[1])
			# save initial tuples

			if size > self.trajectory_manager.max_episode_length[idx]:
				self.trajectory_manager.max_episode_length[idx] = size
			# get values
			states, actions, rewards, values, logprobs, TDs, GAEs = zip(*data)
			embed()
			exit()
			values = np.concatenate((values, [0]), axis=0)
			advantages = np.zeros(size)
			ad_t = 0

			for i in reversed(range(len(data))):
				delta = rewards[i] + values[i+1] * self.gamma - values[i]
				ad_t = delta + self.gamma * self.lambd * ad_t
				advantages[i] = ad_t

			TD = values[:size] + advantages
			for i in range(size):
				self.replay_buffer.Push(states[i], actions[i], rewards[i], values[i], logprobs[i], TD[i], advantages[i])


	def Save(self, path):
		self.saver.save(self.sess, path, global_step = 0)

	def Restore(self, path):
		self.saver.restore(self.sess, path)

	def RunWithSavedData(self, rsi=False):
		# self.trajectory_manager.generateTrajectories(self.num_trajectories, self.traj_frame)
		for i in range(self.num_slaves):
			traj, g_traj, index, t_index, t = self.trajectory_manager.getTrajectory(i%self.num_trajectories)
			self.Env.SetReferenceTrajectory(i, self.traj_frame, traj)
			self.Env.SetGoalTrajectory(i, self.traj_frame, g_traj)
			# print(t)
			self.Env.reset(i, float(i)/self.num_slaves)
		# self.Env.Resets(rsi)
		# self.Env.Reset(0, False)
	
		total_reward = 0
		total_step = 0
		action_mean = []
		state_arr = []
		# adaptive initial states
		if self.use_adaptive_initial_state:
			state_origins = self.Env.getStates()
			state_origins = self.RMS.apply(state_origins)

			state_deltas = self.adaptive_initial_state.GetMeanInitialStateDelta(state_origins)
			self.Env.UpdateInitialStates(state_deltas)

		# get new states
		states = self.Env.getStates()
		states = self.RMS.apply(states)
		for t in count():			
			state_arr.append(states)
			actions = self.actor.GetMeanAction(states)
			self.actionRMS.update(actions)
			action_mean.append(np.array(actions))
			self.setActions(actions)

			if not self.use_trajectory:
				cur_target, new_target = self.reference_manager.getReferences()
				self.Env.SetReferenceToTargets(cur_target, new_target)
				self.Env.SetGoals(self.reference_manager.getTargets())
			self.Env.steps()

			for j in range(self.num_slaves):
				if self.Env.IsTerminalState(j) is not True:
					total_step += 1
					total_reward += self.Env.GetReward(j)

			terminated = self.Env.IsTerminalStates()
			states = self.Env.getStates()
			states_for_update = states[~np.array(terminated)]  
			states_for_update = self.RMS.apply(states_for_update)
			states[~np.array(terminated)] = states_for_update

			if all( terminate == True for terminate in terminated):
				break

		print("Elapsed time : {:.2f}s".format(time.time() - self.start_time))
		print('Num eval : {}, Epi reward : {}, Step reward : {}'.format(self.num_evaluation, total_reward/self.num_slaves, total_reward/total_step))
		print('episode count : {}'.format(self.num_slaves))
		self.Env.WriteRecords(self.directory)
		# print(self.actionRMS.mean)
		# print(self.actionRMS.var)
		# print(self.RMS.mean)
		# print(self.RMS.var)

	def PrintSummary(self):
		np.save(self.directory+"rms/cur/mean.npy",self.RMS.mean)
		np.save(self.directory+"rms/cur/var.npy",self.RMS.var)

		np.save(self.directory+"rms/mean_{}.npy".format(self.num_evaluation),self.RMS.mean)
		np.save(self.directory+"rms/var_{}.npy".format(self.num_evaluation),self.RMS.var)
		np.save(self.directory+"rms/action_mean_{}.npy".format(self.num_evaluation),self.actionRMS.mean)
		np.save(self.directory+"rms/action_var_{}.npy".format(self.num_evaluation),self.actionRMS.var)

		print('===============================================================')
		print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print("Elapsed time : {:.2f}s".format(time.time() - self.start_time))
		print("sim : {:.2f}s, train : {:.2f}s".format(self.sim_time, self.train_time))

		print("test_name : {}".format(self.detail))
		print('Num eval : {}'.format(self.num_evaluation))
		print('noise : {:.3f}'.format(self.sess.run(self.actor.std).mean()))
		print('learning rate : {:.6f}'.format(self.learning_rate))
		print('total episode count : {}'.format(self.num_episodes_total))
		print('total transition count : {}'.format(self.num_transitions_total))
		t_per_e = 0
		if self.num_episodes_total is not 0:
			t_per_e = self.num_transitions_total / self.num_episodes_total

		print('total transition per episodes : {:.2f}'.format(t_per_e))
		print('episode count : {}'.format(self.num_episodes_per_iteration))
		print('transition count : {}'.format(self.num_transitions_per_iteration))
		t_per_e = 0
		if self.num_episodes_per_iteration is not 0:
			t_per_e = self.num_transitions_per_iteration / self.num_episodes_per_iteration
		self.transition_per_episodes.append(t_per_e)
		print('transition per episodes : {:.2f}'.format(t_per_e))
		print('max episode length : {}'.format(self.trajectory_manager.max_episode_length.max()))

		print('rewards per episodes : {:.2f}'.format(self.total_rewards[-1]))

		if(self.use_evaluation):
			evaluation_t_per_e = self.evaluation_num_transitions_per_iteration/self.num_slaves
			self.evaluation_transition_per_episodes.append(evaluation_t_per_e)
			print('evaluation transition per episodes : {:.2f}'.format(evaluation_t_per_e))
			print('evaluation rewards per episodes : {:.2f}'.format(self.evaluation_total_rewards[-1]))
		print('===============================================================')
		print(self.trajectory_manager.max_episode_length)
		print(self.trajectory_manager.epi_rewards_mean)
		print(self.trajectory_manager.epi_rewards_count)
		# print(self.trajectory_manager.rewards_per_time_pieces)
		# print(self.trajectory_manager.reward_counts_per_time_pieces)
		# e = np.exp(-self.trajectory_manager.epi_rewards_mean*0.1)
		# # e = 1.0/(np.array(self.trajectory_manager.epi_rewards_mean)+1.0)
		# e = e / e.sum()
		# print(e)

		# directory = '../output_'+self.name+'/'
		if(self.use_evaluation):
			y_list = [[np.asarray(self.evaluation_total_rewards_by_parts[0]), 'r'], 
						[np.asarray(self.evaluation_mean_rewards), 'r_mean'],
						[np.asarray(self.evaluation_transition_per_episodes), 'steps'], 
						[np.asarray(self.evaluation_total_rewards_by_parts[1]), 'p'], 
						[np.asarray(self.evaluation_total_rewards_by_parts[2]), 'v'], 
						# [np.asarray(self.total_rewards_by_parts[3]), 'p_upper'], 
						# [np.asarray(self.total_rewards_by_parts[4]), 'v_upper'], 
						[np.asarray(self.evaluation_total_rewards_by_parts[3]), 'com'],
						# [np.asarray(self.total_rewards_by_parts[6]), 'com_v'], 
						# [np.asarray(self.total_rewards_by_parts[7]), 'ori'],
						[np.asarray(self.evaluation_total_rewards_by_parts[4]), 'ee']]
						# [np.asarray(self.total_rewards_by_parts[8]), 'av'], 
						# [np.asarray(self.total_rewards_by_parts[5]), 'mass']] 
						# [np.asarray(self.total_rewards_by_parts[10]), 'ee_ori']]
			Plot(y_list, self.name,1,False, path=self.directory+"result.png")

			for i in range(len(y_list)):
				y_list[i][0] = np.array(y_list[i][0])/np.array(self.evaluation_transition_per_episodes)
			y_list[1][0] = np.asarray(self.noise_records)
			y_list[1][1] = 'noise'

			Plot(y_list, self.name+"_per_step",2,False, path=self.directory+"result_per_step.png")
		else:
			y_list = [[np.asarray(self.total_rewards_by_parts[0]), 'r'], 
						[np.asarray(self.mean_rewards), 'r_mean'],
						[np.asarray(self.transition_per_episodes), 'steps'], 
						[np.asarray(self.total_rewards_by_parts[1]), 'p'], 
						[np.asarray(self.total_rewards_by_parts[2]), 'v'], 
						# [np.asarray(self.total_rewards_by_parts[3]), 'p_upper'], 
						# [np.asarray(self.total_rewards_by_parts[4]), 'v_upper'], 
						[np.asarray(self.total_rewards_by_parts[3]), 'com'],
						# [np.asarray(self.total_rewards_by_parts[6]), 'com_v'], 
						# [np.asarray(self.total_rewards_by_parts[7]), 'ori'],
						[np.asarray(self.total_rewards_by_parts[4]), 'ee']]
						# [np.asarray(self.total_rewards_by_parts[8]), 'av'], 
						# [np.asarray(self.total_rewards_by_parts[5]), 'mass']] 
						# [np.asarray(self.total_rewards_by_parts[10]), 'ee_ori']]
			Plot(y_list, self.name,1,False, path=self.directory+"result.png")

			for i in range(len(y_list)):
				y_list[i][0] = np.array(y_list[i][0])/np.array(self.transition_per_episodes)
			y_list[1][0] = np.asarray(self.noise_records)
			y_list[1][1] = 'noise'

			Plot(y_list, self.name+"_per_step",2,False, path=self.directory+"result_per_step.png")

		out = open(self.directory+"results", "a")
		out.write('===============================================================\n')
		out.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S\n"))
		out.write("Elapsed time : {:.2f}s\n".format(time.time() - self.start_time))
		out.write("sim : {:.2f}s, train : {:.2f}s\n".format(self.sim_time, self.train_time))

		out.write('Num eval : {}\n'.format(self.num_evaluation))
		out.write('noise : {:.3f}\n'.format(self.sess.run(self.actor.std).mean()))
		out.write('learning rate : {:.6f}\n'.format(self.learning_rate))

		out.write('total episode count : {}\n'.format(self.num_episodes_total))
		out.write('total transition count : {}\n'.format(self.num_transitions_total))
		t_per_e = 0
		if self.num_episodes_total is not 0:
			t_per_e = self.num_transitions_total / self.num_episodes_total

		out.write('total transition per episodes : {:.2f}\n'.format(t_per_e))
		out.write('episode count : {}\n'.format(self.num_episodes_per_iteration))
		out.write('transition count : {}\n'.format(self.num_transitions_per_iteration))
		t_per_e = 0
		if self.num_episodes_per_iteration is not 0:
			t_per_e = self.num_transitions_per_iteration / self.num_episodes_per_iteration
		out.write('transition per episodes : {:.2f}\n'.format(t_per_e))
		out.write('max episode length : {}\n'.format(self.trajectory_manager.max_episode_length.max()))

		out.write('rewards per episodes : {:.2f}\n'.format(self.total_rewards[-1]))

		if(self.use_evaluation):
			evaluation_t_per_e = self.evaluation_num_transitions_per_iteration/self.num_slaves
			out.write('evaluation transition per episodes : {:.2f}\n'.format(evaluation_t_per_e))
			out.write('evaluation rewards per episodes : {:.2f}\n'.format(self.evaluation_total_rewards[-1]))
		out.write('===============================================================\n')
		out.write(str(self.trajectory_manager.max_episode_length)+"\n")
		out.close()

		self.trajectory_manager.saveTimeWeight(self.directory, self.num_evaluation)

		self.Save(self.directory+"network")

		if t_per_e > self.smax:
			self.smax = t_per_e
			np.save(self.directory+"rms/mean_smax.npy",self.RMS.mean)
			np.save(self.directory+"rms/var_smax.npy",self.RMS.var)

			os.system("cp {}/network-{}.data-00000-of-00001 {}/network-smax.data-00000-of-00001".format(self.directory, 0, self.directory))
			os.system("cp {}/network-{}.index {}/network-smax.index".format(self.directory, 0, self.directory))
			os.system("cp {}/network-{}.meta {}/network-smax.meta".format(self.directory, 0, self.directory))

		if self.total_rewards[-1] > self.rmax:
			self.rmax = self.total_rewards[-1]
			np.save(self.directory+"rms/mean_rmax.npy",self.RMS.mean)
			np.save(self.directory+"rms/var_rmax.npy",self.RMS.var)

			os.system("cp {}/network-{}.data-00000-of-00001 {}/network-rmax.data-00000-of-00001".format(self.directory, 0, self.directory))
			os.system("cp {}/network-{}.index {}/network-rmax.index".format(self.directory, 0, self.directory))
			os.system("cp {}/network-{}.meta {}/network-rmax.meta".format(self.directory, 0, self.directory))

		self.num_evaluation = self.num_evaluation + 1

	def GetEpiReward(self, epi, gamma = 1.0):
		data = epi.GetData()
		rew = 0
		for i in reversed(range(len(data))):
			rew = data[i].r + rew*gamma
		return rew

	def Evaluation(self):
		print("Evaluation start")
		self.evaluation_num_transitions_per_iteration = 0
		self.evaluation_reward_per_iteration = 0
		self.evaluation_reward_by_part_per_iteration = []

		for i in range(self.num_slaves):
			self.Env.reset(i, float(i)/self.num_slaves)
	
		terminated = [False]*self.num_slaves

		# adaptive initial states
		if self.use_adaptive_initial_state:
			state_origins = self.Env.getStates()
			state_origins = self.RMS.apply(state_origins)

			state_deltas = self.adaptive_initial_state.GetMeanInitialStateDelta(state_origins)
			self.Env.UpdateInitialStates(state_deltas)

		# get new states
		states = self.Env.getStates()
		states = self.RMS.apply(states)
		local_step = 0
		for t in count():			
			actions = self.actor.GetMeanAction(states)
			self.actionRMS.update(actions)
			self.setActions(actions)
			
			if not self.use_trajectory:
				cur_target, new_target = self.reference_manager.getReferences()
				self.Env.SetReferenceToTargets(cur_target, new_target)
				self.Env.SetGoals(self.reference_manager.getTargets())
			self.Env.steps()

			for j in range(self.num_slaves):
				if terminated[j]:
					continue
				is_terminal, nan_occur, time_end = self.Env.isNanAtTerminal(j)
				if nan_occur is not True:
					r = self.Env.getReward(j)
					self.evaluation_reward_per_iteration += r[0]
					self.evaluation_reward_by_part_per_iteration.append(r)
					local_step += 1

				# if episode is terminated
				if is_terminal:
					terminated[j] = True

			if all(t is True for t in terminated):
				print('{}/{}'.format(np.array(terminated).sum(), local_step),end='\r')
				break

			if last_print + 100 < local_step: 
				print('{}/{}'.format(np.array(terminated).sum(), local_step),end='\r')
				last_print = local_step

			# update states				
			states = self.Env.getStates()
			states_for_update = states[~np.array(terminated)]  
			states_for_update = self.RMS.apply(states_for_update)
			states[~np.array(terminated)] = states_for_update

		print('')

		self.evaluation_num_transitions_per_iteration += local_step

		self.evaluation_total_rewards.append(self.evaluation_reward_per_iteration/self.num_slaves)
		self.evaluation_total_rewards_by_parts = np.insert(self.evaluation_total_rewards_by_parts, self.evaluation_total_rewards_by_parts.shape[1], np.asarray(self.evaluation_reward_by_part_per_iteration).sum(axis=0)/self.num_slaves, axis=1)
		self.evaluation_mean_rewards.append(np.asarray(self.evaluation_total_rewards)[-10:].mean())


		print("Evaluation end\n")

	def RunTraining(self, num_iteration=1):  ##SM) num_iteration= #iteration per summary
		print("\nTraining start")
		self.num_episodes_per_iteration = 0
		self.num_transitions_per_iteration = 0
		self.reward_per_iteration = 0
		self.reward_by_part_per_iteration = []
		self.max_episode_length = 0

		# self.trajectory_manager.resetTimeWeight()

		for it in range(num_iteration):
			self.sim_time -= time.time()
			# st = time.time()
			self.total_episodes = []
			self.total_episodes_idx = []
			self.total_time_idx = []
			nan_count = 0

			for i in range(self.num_slaves):
				t_index, timet = self.trajectory_manager.selectTime(self.traj_indices[i])
				self.time_indices[i] = t_index
				self.Env.reset(i, timet)

			# get new states
			states = self.Env.getStates()
			states = self.RMS.apply(states)

			actions = [None]*self.num_slaves
			rewards = [None]*self.num_slaves
			episodes = [None]*self.num_slaves

			terminated = [False]*self.num_slaves

			for j in range(self.num_slaves):
				episodes[j] = Episode()

			local_step = 0
			last_print = 0
			while True:
				# set action
				actions, logprobs = self.actor.GetAction(states)
				self.actionRMS.update(np.array(actions))
				values = self.critic.GetValue(states)
				self.setActions(actions)

				# run one step
				self.Env.steps()

				for j in range(self.num_slaves):
					if terminated[j]:
						continue
					is_terminal, nan_occur, time_end = self.Env.isNanAtTerminal(j)
					if nan_occur is not True:
						r = self.Env.getReward(j)
						rewards[j] = r[0]
						self.reward_per_iteration += rewards[j]
						self.reward_by_part_per_iteration.append(r)
						episodes[j].Push(states[j], actions[j], rewards[j], values[j], logprobs[j])
						local_step += 1
					else:
						nan_count += 1

					# if episode is terminated
					if is_terminal:
						# print("{} is terminated at {} with {}".format(j, local_step, len(episodes[j].GetData())))
						# push episodes
						if len(episodes[j].GetData()) != 0:
							# epi_rew = self.GetEpiReward(episodes[j])
							# self.trajectory_manager.updateEpiReward(traj_indices[j], epi_rew)
							self.total_episodes.append(episodes[j])
							self.total_episodes_idx.append(self.traj_indices[j])
							self.total_time_idx.append([self.time_indices[j], time_end])
							self.trajectory_manager.updateEpiReward(self.traj_indices[j], self.time_indices[j], self.GetEpiReward(episodes[j]), len(episodes[j].GetData()), time_end)

						if local_step < self.steps_per_iteration:
							episodes[j] = Episode()

							t_index, timet = self.trajectory_manager.selectTime(self.traj_indices[j])
							self.time_indices[j] = t_index
							
							self.Env.reset(j, timet) 
						else:
							terminated[j] = True

				if local_step >= self.steps_per_iteration:  ##SM) if local step exceeds s_p_i: wait for others to terminate
					if all(t is True for t in terminated):
						print('{}/{} : {}/{}'.format(it+1, num_iteration, local_step, self.steps_per_iteration),end='\r')
						break

				if last_print + 100 < local_step: 
					print('{}/{} : {}/{}'.format(it+1, num_iteration, local_step, self.steps_per_iteration),end='\r')
					last_print = local_step
				# update states				
				states = self.Env.getStates()
				states_for_update = states[~np.array(terminated)]  
				states_for_update = self.RMS.apply(states_for_update)
				states[~np.array(terminated)] = states_for_update

				# states = self.RMS.apply(states)
				# self.RMS.update(states)

			self.sim_time += time.time()

			self.train_time -= time.time()
			# optimization
			print('')
			if(nan_count > 0):
				print("nan_count : {}".format(nan_count))
			self.Optimize()  ##SM) after getting all tuples, optimize once
			self.num_episodes_per_iteration += len(self.total_episodes)
			self.num_transitions_per_iteration += local_step
			self.train_time += time.time()

			# if self.traj_frame < 4000:
			# 	if all(self.trajectory_manager.max_episode_length >= self.steps_per_iteration/self.num_slaves * 0.8):	
			# 		# self.batch_size *= 2
			# 		self.steps_per_iteration *= 2 
			# 		self.learning_rate /= 2
			# 	elif all(self.trajectory_manager.max_episode_length >= (self.traj_frame-30.0)*1.1):
			# 		self.traj_frame *= 2
			# 		self.iteration_count = 0
			# 		if self.use_trajectory:
			# 			print("Reach at max epi length, regenerating trajectories...")
			# 			self.trajectory_manager.generateTrajectories(self.num_trajectories, self.traj_frame)
			# 			# save trajectories
			# 			self.traj_count += 1
			# 			directory = self.directory + "trajectories/"
			# 			if not os.path.exists(directory):
			# 				os.mkdir(directory)
			# 			self.trajectory_manager.save(directory+"{}_{}_".format(self.traj_count, self.traj_frame))


		# trajectory regeneration
		self.iteration_count += 1
	# if self.iteration_count%self.traj_regeneration_period == 0:
		# 	if self.traj_frame < 12800:
		# 		self.traj_frame *= 2
		# 		self.trajectory_manager.generateTrajectories(self.num_trajectories, self.traj_frame)

		if self.learning_rate > 1e-5:
			self.learning_rate = self.learning_rate * self.learning_rate_decay



		print('Training end\n')
		if(self.use_evaluation):
			self.Evaluation()

		self.total_rewards.append(self.reward_per_iteration/self.num_episodes_per_iteration)
		self.total_rewards_by_parts = np.insert(self.total_rewards_by_parts, self.total_rewards_by_parts.shape[1], np.asarray(self.reward_by_part_per_iteration).sum(axis=0)/self.num_episodes_per_iteration, axis=1)
		self.mean_rewards.append(np.asarray(self.total_rewards)[-10:].mean())
		self.noise_records.append(self.sess.run(self.actor.std).mean())

		self.num_episodes_total += self.num_episodes_per_iteration
		self.num_transitions_total += self.num_transitions_per_iteration

		# print(self.actionRMS.mean)
		# print(self.actionRMS.var)
		self.PrintSummary()



if __name__=="__main__":
	ppo = PPO()
	##SM) steps_per_iteration= ??
	##SM) frame= total trajectory length,
	##SM) num_time_piecies= dividing segment num (for choosing where to start)
	ppo.InitializeForTraining(env_name='humanoid',num_slaves=8, num_trajectories=1, useBothOriginNRandom= False,
		learning_rate=2e-4,
		gamma=0.99, lambd=0.95,
		batch_size=1024, steps_per_iteration=200, 
		origin=True, frame=1000, origin_offset=0,
		num_time_piecies=1000,
		use_adaptive_initial_state=False,
		use_evaluation=False,
		motion = CharacterConfigurations.motion,
		# pretrain="../output/testpunchfall/network-smax",
		detail="wrf_test")

	for i in range(1000000):
		ppo.RunTraining(1)
