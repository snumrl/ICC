import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import random
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from rl.AdaptiveSampler import AdaptiveSampler
import util.Util

from rl.ReplayBuffer import ReplayBuffer
from rl.ReplayBuffer import Transition
from rl.ReplayBuffer import Episode

from rl.RunningMeanStd import RunningMeanStd
from environment_wrapper import environment
from rl.Configurations import Configurations
from util.Util import Plot

from rnn.MotionGenerator import MotionGenerator

from IPython import embed

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

class Policy:
	def __init__(self, action_size):
		self._scope = "policy"

		self.createModel(action_size) 

	def createModel(self, action_size):
		# log std
		self._logstd = tf.Variable(
			initial_value = np.zeros(action_size),
			trainable=True,
			name='logstd',
			dtype=tf.float32
		)

		# std

		self._layers = [tf.keras.layers.Dense(Configurations.instance().policyLayerSize,
						activation=Configurations.instance().activationFunction,
						dtype=tf.float32) for _ in range(Configurations.instance().policyLayerNumber)]
		self._layers.append(tf.keras.layers.Dense(action_size, dtype=tf.float32))

		self._mean = tf.keras.Sequential(self._layers)

	@tf.function
	def getActionAndNeglogprob(self, states):
		actions = self.action(states)
		return actions, self.neglogprob(actions, states)

	@tf.function
	def getMeanAction(self, states):
		return self._mean(states)

	@tf.function
	def std(self):
		return tf.exp(self._logstd)

	@property
	def logstd(self):
		return self._logstd
	
	@property
	def mean(self):
		return self._mean

	@tf.function
	def action(self, states):
		mean = self.mean(states)
		return mean + self.std()*tf.random.normal(tf.shape(mean))
	
	@tf.function
	def neglogprob(self, actions, states):
		return 0.5 * tf.math.reduce_sum(tf.math.square((actions - self._mean(states)) / self.std()), axis=-1) + 0.5 * tf.math.log(tf.constant(2.0 * 3.1415926535, dtype=tf.float32)) * tf.cast(tf.shape(actions), tf.float32)[-1] + tf.math.reduce_sum(self._logstd, axis=-1)

	def trainable_variables(self):
		return self._mean.trainable_variables + [self._logstd]

	def build(self, state_size):
		self._mean.build((None, state_size))

class ValueFunction:
	def __init__(self):
		self._scope = "valueFunction"
		self.createModel()

	def createModel(self):
		self._layers = [tf.keras.layers.Dense(Configurations.instance().valueLayerSize,
						activation=Configurations.instance().activationFunction,
						dtype=tf.float32) for _ in range(Configurations.instance().valueLayerNumber)]
		self._layers.append(tf.keras.layers.Dense(1, dtype=tf.float32))

		self._value = tf.keras.Sequential(self._layers)

	@property
	def value(self):
		return self._value

	@tf.function
	def getValue(self, states):
		return self.value(states)[:,0]

	def trainable_variables(self):
		return self._value.trainable_variables
	
	def build(self, state_size):
		self._value.build((None, state_size))

class TrackingController:
	def __init__(self):
		random.seed(int(time.time()))
		np.random.seed(int(time.time()))

		self._startTime = time.time()
		self._summary_sim_time = 0
		self._summary_train_time = 0

		self._timeChecker = util.Util.TimeChecker()

	def initialize(self, configuration_filepath="", trajectory_length=None, origin=None, origin_offset=None):
		self._configurationFilePath = configuration_filepath
		Configurations.instance().loadData(configuration_filepath)

		# get parameters from config
		self._numSlaves 				= Configurations.instance().numSlaves
		self._motion 					= Configurations.instance().motion

		self._gamma 					= Configurations.instance().gamma
		self._lambd 					= Configurations.instance().lambd
		self._clipRange					= Configurations.instance().clipRange

		self._learningRatePolicy 		= Configurations.instance().learningRatePolicy
		self._learningRatePolicyDecay	= Configurations.instance().learningRatePolicyDecay
		self._learningRateValueFunction = Configurations.instance().learningRateValueFunction		

		self._batchSize 				= Configurations.instance().batchSize
		self._transitionsPerIteration 	= Configurations.instance().transitionsPerIteration
 

		self._trajectoryLength          = Configurations.instance().trajectoryLength
		if trajectory_length is not None:
			self._trajectoryLength = trajectory_length

		self._useOrigin					= Configurations.instance().useOrigin
		if origin is not None:
			self._useOrigin = origin

		self._originOffset				= Configurations.instance().originOffset
		if origin_offset is not None:
			self._originOffset = origin_offset


		self._adaptiveSamplingSize		= Configurations.instance().adaptiveSamplingSize

		# if useEvaluation is true, evaluation of training progress is performed by evaluation function, else it is done by transitions collected in training session.
		self._useEvaluation 			= Configurations.instance().useEvaluation

		self._sessionName				= Configurations.instance().sessionName

		# initialize environment
		self._env = environment(configuration_filepath)
		self._stateSize = self._env.getStateSize()
		self._actionSize = self._env.getActionSize()
		self._rms = RunningMeanStd(shape=(self._stateSize))

		# initialize networks
		self._policy = Policy(self._actionSize)
		self._policy.build(self._stateSize)
		self._valueFunction = ValueFunction()
		self._valueFunction.build(self._stateSize)

		# initialize RunningMeanStd
		self._rms = RunningMeanStd(shape=(self._stateSize))

		# initialize replay buffer
		self._replayBuffer = ReplayBuffer()


		self._policyOptimizer = tf.keras.optimizers.Adam(learning_rate=self.decayedLearningRatePolicy)
		self._valueFunctionOptimizer = tf.keras.optimizers.Adam(learning_rate=self._learningRateValueFunction)

		# initialize motion generator
		self._motionGenerator = MotionGenerator(1, self._motion)
		self._timeIndices = [None]*self._numSlaves

		# generate trajectories
		self.generateTrajectory()
		self._env.setReferenceTrajectories(self._trajectoryLength, self._trajectory)
		self._env.setReferenceTargetTrajectories(self._trajectoryLength, self._targetTrajectory)

		# initialize adaptive sampler
		self._adaptiveSampler = AdaptiveSampler(self._adaptiveSamplingSize)

		# initialize saver
		# self._saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)
		# save maximum step network
		self._smax = 0
		# save maximum reward network
		self._rmax = 0


		# initialize statistics variables
		# TODO
		self._summary_num_log = 0
		self._summary_num_episodes_total = 0
		self._summary_num_transitions_total = 0

		self._summary_max_episode_length = 0

		self._summary_total_rewards = []
		self._summary_total_rewards_by_parts = np.array([[]]*5)
		self._summary_mean_rewards = []
		self._summary_transition_per_episodes = []
		self._summary_noise_records = []

		self._summary_evaluation_total_rewards = []
		self._summary_evaluation_total_rewards_by_parts = np.array([[]]*5)
		self._summary_evaluation_mean_rewards = []
		self._summary_evaluation_transition_per_episodes = []

		# initialize checkpoint 
		self._ckpt = tf.train.Checkpoint(
			policy_mean=self._policy.mean,
			policy_logstd=self._policy.logstd,
			valueFunction=self._valueFunction.value
			# policyOptimizer=self._policyOptimizer,
			# valueFunctionOptimizer=self._valueFunctionOptimizer
		)

		self._isNetworkLoaded = False
		self._loadedNetwork = ""

	def decayedLearningRatePolicy(self):
		return self._learningRatePolicy


	# load trained networks & rms
	def loadNetworks(self, directory, network_num=None):
		# load rms
		rms_dir = "{}/rms/".format(directory)
		if network_num is not None:
			mean_dir = rms_dir+"mean_{}.npy".format(network_num)
			var_dir = rms_dir+"var_{}.npy".format(network_num)
		else:
			mean_dir = rms_dir+"mean.npy"
			var_dir = rms_dir+"var.npy"

		if os.path.exists(mean_dir):
			print("Loading RMS parameters")
			self._rms.mean = np.load(mean_dir)
			self._rms.var = np.load(var_dir)
			self._rms.count = 200000000

		# load netowrk
		if network_num is not None:
			network_dir = "{}/network-{}".format(directory, network_num)
		else:
			network_dir = "{}/network".format(directory)
		print("Loading networks from {}".format(network_dir))


		self.restore(network_dir)

		self._isNetworkLoaded = True
		self._loadedNetwork = "{}".format(network_dir)

	def generateTrajectory(self):
		if self._useOrigin:
			self._trajectory, self._targetTrajectory = self._motionGenerator.getOriginalTrajectory(self._trajectoryLength, self._originOffset)
		else:
			traj_filename = "../trajectories/{}/traj.npy".format(self._motion)
			goal_filename = "../trajectories/{}/goal.npy".format(self._motion)
			if os.path.exists(traj_filename) and os.path.exists(goal_filename):
				print("Loading trajectories from {}".format(traj_filename))
				traj = np.load(traj_filename)
				target_traj = np.load(goal_filename)
				if traj.shape[1] < self._trajectoryLength:
					print("motion is too short, required : {}, maximum : {}".format(self._trajectoryLength, traj.shape[1]))
					# exit()
					self._trajectoryLength = traj.shape[1]
				self._trajectory = traj[0][:self._trajectoryLength]
				self._targetTrajectory = target_traj[0][:self._trajectoryLength]
			else:
				traj, target_traj = self._motionGenerator.getTrajectory(self._trajectoryLength)
				self._trajectory = traj[0]
				self._targetTrajectory = target_traj[0]

	def computeTDAndGAE(self):
		self._collectedStates = [None] * self._summary_num_transitions_per_iteration
		self._collectedActions = [None] * self._summary_num_transitions_per_iteration
		self._collectedNeglogprobs = [None] * self._summary_num_transitions_per_iteration
		self._collectedTDs = [None] * self._summary_num_transitions_per_iteration
		self._collectedGAEs = [None] * self._summary_num_transitions_per_iteration

		startIdx = 0
		for epi in self._collectedEpisodes:
			data = epi.data
			size = len(data)

			# update max episorde length
			if size > self._summary_max_episode_length:
				self._summary_max_episode_length = size

			states, actions, rewards, values, neglogprobs, TDs, GAEs = zip(*data)
			values = tf.convert_to_tensor(values).numpy()
			values = np.concatenate((values, [0]), axis=0)
			advantages = np.zeros(size)
			ad_t = 0

			for i in reversed(range(size)):
				delta = rewards[i] + values[i+1] * self._gamma - values[i]
				ad_t = delta + self._gamma * self._lambd * ad_t
				advantages[i] = ad_t

			TD = values[:size] + advantages
			self._collectedStates[startIdx:startIdx+size] = list(states)
			self._collectedActions[startIdx:startIdx+size] = list(actions)
			self._collectedNeglogprobs[startIdx:startIdx+size] = list(neglogprobs)
			self._collectedTDs[startIdx:startIdx+size] = list(TD)
			self._collectedGAEs[startIdx:startIdx+size] = list(advantages)

			startIdx += size


		self._collectedStates = np.array(self._collectedStates, dtype=np.float32)
		self._collectedActions = tf.convert_to_tensor(self._collectedActions).numpy()
		self._collectedNeglogprobs = tf.convert_to_tensor(self._collectedNeglogprobs).numpy()
		self._collectedTDs = np.array(self._collectedTDs, dtype=np.float32)
		self._collectedGAEs = np.array(self._collectedGAEs, dtype=np.float32)


	def optimize(self):
		self.computeTDAndGAE()
		if len(self._collectedStates) < self._batchSize:
			return

		GAE = np.array(self._collectedGAEs)
		GAE = (GAE - GAE.mean())/(GAE.std() + 1e-5)

		ind = np.arange(len(GAE))

		np.random.shuffle(ind)


		for s in range(int(len(ind)//self._batchSize)):
			selectedIndex = ind[s*self._batchSize:(s+1)*self._batchSize]

			selectedStates = tf.convert_to_tensor(self._collectedStates[selectedIndex])
			selectedActions = tf.convert_to_tensor(self._collectedActions[selectedIndex])
			selectedNeglogprobs = tf.convert_to_tensor(self._collectedNeglogprobs[selectedIndex])
			selectedTDs = tf.convert_to_tensor(self._collectedTDs[selectedIndex])
			selectedGAEs = tf.convert_to_tensor(GAE[selectedIndex])

			self.optimizeStep(selectedActions, selectedStates, selectedNeglogprobs, selectedTDs, selectedGAEs)

	def optimizeStep(self, a, s, nl, td, gae):
		with tf.GradientTape() as tape:
			curNeglogprob = self._policy.neglogprob(a, s)
			ratio = tf.exp(nl - curNeglogprob)
			clippedRatio = tf.clip_by_value(ratio, 1.0 - self._clipRange, 1.0 + self._clipRange)
			policyLoss = -tf.reduce_mean(tf.minimum(ratio*gae, clippedRatio*gae))
			
		gradients = tape.gradient(policyLoss, self._policy.trainable_variables())
		gradients, _grad_norm = tf.clip_by_global_norm(gradients, 0.5)
		self._policyOptimizer.apply_gradients(zip(gradients, self._policy.trainable_variables()))

		# optimize value function
		with tf.GradientTape() as tape:
			valueLoss = tf.reduce_mean(tf.square(self._valueFunction.getValue(s) - td))
		gradients = tape.gradient(valueLoss, self._valueFunction._value.trainable_variables)
		gradients, _grad_norm = tf.clip_by_global_norm(gradients, 0.5)
		self._valueFunctionOptimizer.apply_gradients(zip(gradients, self._valueFunction._value.trainable_variables))


	def reset(self):
		return

	def followReference(self):
		# create logging directory
		if not os.path.exists("../output/"):
			os.mkdir("../output/")
		self._directory = '../output/'+self._sessionName+'/'

		directory = self._directory + "trajectory/"
		if not os.path.exists(directory):
			os.mkdir(directory)

		self.printParameters()


		for i in range(self._numSlaves):
			self._env.reset(i, i*1.0/self._numSlaves)

		# get new states
		terminated = [False]*self._numSlaves

		while True:
			# run one step
			self._env.followReferences()

			for j in range(self._numSlaves):
				if terminated[j]:
					continue

				is_terminal, nan_occur, end_of_trajectory = self._env.isNanAtTerminal(j)

				# if episode is terminated
				if is_terminal:
					terminated[j] = True

			# if local step exceeds t_p_i: wait for others to terminate
			if all(t is True for t in terminated):
				break

		self._env.writeRecords(self._directory+"ref_")


	def runTraining(self, num_iteration=1):
		# create logging directory
		if not os.path.exists("../output/"):
			os.mkdir("../output/")
		self._directory = '../output/'+self._sessionName+'/'

		if not os.path.exists(self._directory):
			os.mkdir(self._directory)

		directory = self._directory + "trajectory/"
		if not os.path.exists(directory):
			os.mkdir(directory)

		directory = self._directory + "rms/"
		if not os.path.exists(directory):
			os.mkdir(directory)

		directory = directory + "cur/"
		if not os.path.exists(directory):
			os.mkdir(directory)

		self.printParameters()

		while True:
			print("\nTraining start")
			self._summary_num_episodes_per_epoch = 0
			self._summary_num_transitions_per_epoch = 0
			self._summary_reward_per_epoch = 0
			self._summary_reward_by_part_per_epoch = []
			self._summary_max_episode_length = 0

			for it in range(num_iteration):
				self._summary_sim_time -= time.time()
				self._collectedEpisodes = []

				nan_count = 0

				for i in range(self._numSlaves):
					# select time indices
					index, timet = self._adaptiveSampler.selectTime()
					self._timeIndices[i] = index
					self._env.reset(i, timet)

				# get new states
				states = self._env.getStates()
				states = self._rms.apply(states)


				actions = [None]*self._numSlaves
				rewards = [None]*self._numSlaves
				episodes = [None]*self._numSlaves

				terminated = [False]*self._numSlaves

				for j in range(self._numSlaves):
					episodes[j] = Episode()

				self._summary_num_transitions_per_iteration = 0
				last_print = 0
				while True:
					# set action
					actions, logprobs = self._policy.getActionAndNeglogprob(states)
					values = self._valueFunction.getValue(states)			
					self._env.setActions(actions.numpy())

					# run one step
					self._env.steps(False)

					for j in range(self._numSlaves):
						if terminated[j]:
							continue

						is_terminal, nan_occur, end_of_trajectory = self._env.isNanAtTerminal(j)
						# push tuples only if nan did not occur
						if nan_occur is not True:
							r = self._env.getReward(j)
							rewards[j] = r[0]
							self._summary_reward_per_epoch += rewards[j]
							self._summary_reward_by_part_per_epoch.append(r)
							episodes[j].push(states[j], actions[j], rewards[j], values[j], logprobs[j])
							self._summary_num_transitions_per_iteration += 1
						else:
							nan_count += 1

						# if episode is terminated
						if is_terminal:
							# push episodes
							if len(episodes[j].data) != 0:
								self._collectedEpisodes.append(episodes[j])

								# update adaptive sampling weights
								self._adaptiveSampler.updateWeights(self._timeIndices[j], episodes[j].getTotalReward(), end_of_trajectory)

							if self._summary_num_transitions_per_iteration < self._transitionsPerIteration:
								episodes[j] = Episode()

								# select time index
								index, timet = self._adaptiveSampler.selectTime()
								self._timeIndices[j] = index
								
								self._env.reset(j, timet) 
							else:
								terminated[j] = True

					# if local step exceeds t_p_i: wait for others to terminate
					if self._summary_num_transitions_per_iteration >= self._transitionsPerIteration:  
						if all(t is True for t in terminated):
							print('{}/{} : {}/{}'.format(it+1, num_iteration, self._summary_num_transitions_per_iteration, self._transitionsPerIteration),end='\r')
							break

					# print progress per 100 steps
					if last_print + 100 < self._summary_num_transitions_per_iteration: 
						print('{}/{} : {}/{}'.format(it+1, num_iteration, self._summary_num_transitions_per_iteration, self._transitionsPerIteration),end='\r')
						last_print = self._summary_num_transitions_per_iteration

					# update states				
					states = self._env.getStates()
					states_for_update = states[~np.array(terminated)]  
					states_for_update = self._rms.apply(states_for_update)
					states[~np.array(terminated)] = states_for_update

				self._summary_sim_time += time.time()
				self._summary_train_time -= time.time()

				# optimization
				print('')
				if(nan_count > 0):
					print("nan_count : {}".format(nan_count))

				self._summary_num_episodes_per_epoch += len(self._collectedEpisodes)
				self._summary_num_transitions_per_epoch += self._summary_num_transitions_per_iteration

				self.optimize()  ##SM) after getting all tuples, optimize once

				self._summary_train_time += time.time()


			# decay learning rate
			if self._learningRatePolicy > 1e-5:
				self._learningRatePolicy = self._learningRatePolicy * self._learningRatePolicyDecay



			print('Training end\n')
			if(self._useEvaluation):
				self.evaluation()


			self._summary_total_rewards.append(self._summary_reward_per_epoch/self._summary_num_episodes_per_epoch)
			self._summary_total_rewards_by_parts = np.insert(self._summary_total_rewards_by_parts, self._summary_total_rewards_by_parts.shape[1], np.asarray(self._summary_reward_by_part_per_epoch).sum(axis=0)/self._summary_num_episodes_per_epoch, axis=1)
			self._summary_mean_rewards.append(np.asarray(self._summary_total_rewards)[-10:].mean())
			self._summary_noise_records.append(self._policy.std().numpy().mean())


			self._summary_num_episodes_total += self._summary_num_episodes_per_epoch
			self._summary_num_transitions_total += self._summary_num_transitions_per_epoch
			t_per_e = 0
			if self._summary_num_episodes_per_epoch is not 0:
				t_per_e = self._summary_num_transitions_per_epoch / self._summary_num_episodes_per_epoch
			self._summary_transition_per_episodes.append(t_per_e)

			# print summary
			self.printSummary()


	def evaluation(self):
		return

	def play(self):
		# create logging directory
		self._directory = self._sessionName

		if not os.path.exists("../output/"):
			os.mkdir("../output/")
		self._directory = '../output/'+self._sessionName+'/'
		if not os.path.exists(self._directory):
			os.mkdir(self._directory)
		self._directory = '../output/'+self._sessionName+'/play/'
		if not os.path.exists(self._directory):
			os.mkdir(self._directory)

		directory = self._directory + "trajectory/"
		if not os.path.exists(directory):
			os.mkdir(directory)

		directory = self._directory + "rms/"
		if not os.path.exists(directory):
			os.mkdir(directory)
		directory = directory + "cur/"
		if not os.path.exists(directory):
			os.mkdir(directory)

		self.printParameters()

		for i in range(self._numSlaves):
			self._env.reset(i, i*1.0/self._numSlaves)

		# get new states
		states = self._env.getStates()
		states = self._rms.apply(states)

		actions = [None]*self._numSlaves
		terminated = [False]*self._numSlaves


		local_step = 0
		last_print = 0
		while True:
			# set action
			actions, logprobs = self._policy.getActionAndNeglogprob(states)
			values = self._valueFunction.getValue(states)
			self._env.setActions(actions.numpy())

			# run one step
			self._env.steps(True)

			for j in range(self._numSlaves):
				if terminated[j]:
					continue

				is_terminal, nan_occur, end_of_trajectory = self._env.isNanAtTerminal(j)
				# push tuples only if nan did not occur
				if nan_occur is not True:
					local_step += 1
				else:
					nan_count += 1

				# if episode is terminated
				if is_terminal:
					terminated[j] = True

			# if local step exceeds t_p_i: wait for others to terminate
			if all(t is True for t in terminated):
				print('{}'.format(local_step),end='\r')
				break

			# print progress per 100 steps
			if last_print + 100 < local_step: 
				print('{}'.format(local_step),end='\r')
				last_print = local_step

			# update states				
			states = self._env.getStates()
			states_for_update = states[~np.array(terminated)]  
			states_for_update = self._rms.apply(states_for_update)
			states[~np.array(terminated)] = states_for_update

		print('')
		self._env.writeRecords(self._directory)






	def printParameters(self):
		np.save(self._directory+"trajectory/traj.npy", self._trajectory)
		np.save(self._directory+"trajectory/goal.npy", self._targetTrajectory)
		
		# print on shell
		print("===============================================================")
		print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print("Elapsed time         : {:.2f}s".format(time.time()-self._startTime))
		print("Session Name         : {}".format(self._sessionName))
		print("Motion               : {}".format(self._motion))
		print("Slaves number        : {}".format(self._numSlaves))
		print("State size           : {}".format(self._stateSize))
		print("Action size          : {}".format(self._actionSize))
		print("Learning rate        : {:.6f}".format(self._learningRatePolicy))
		print("Gamma                : {}".format(self._gamma))
		print("Lambda               : {}".format(self._lambd))
		print("Batch size           : {}".format(self._batchSize))
		print("Transitions per iter : {}".format(self._transitionsPerIteration))
		print("PPO clip range       : {}".format(self._clipRange))
		print("Trajectory length    : {}".format(self._trajectoryLength))
		print("Use original         : {}".format(self._useOrigin))
		print("Origin offset        : {}".format(self._originOffset))
		print("Loaded netowrks      : {}".format(self._loadedNetwork))
		print("===============================================================")

		# print to file
		out = open(self._directory+"parameters", "w")
		out.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S\n"))
		out.write("Session Name         : {}\n".format(self._sessionName))
		out.write("Motion               : {}\n".format(self._motion))
		out.write("Slaves number        : {}\n".format(self._numSlaves))
		out.write("State size           : {}\n".format(self._stateSize))
		out.write("Action size          : {}\n".format(self._actionSize))
		out.write("Learning rate        : {:.6f}\n".format(self._learningRatePolicy))
		out.write("Gamma                : {}\n".format(self._gamma))
		out.write("Lambda               : {}\n".format(self._lambd))
		out.write("Batch size           : {}\n".format(self._batchSize))
		out.write("Transitions per iter : {}\n".format(self._transitionsPerIteration))
		out.write("PPO clip range       : {}\n".format(self._clipRange))
		out.write("Trajectory length    : {}\n".format(self._trajectoryLength))
		out.write("Use original         : {}\n".format(self._useOrigin))
		out.write("Origin offset        : {}\n".format(self._originOffset))
		out.write("Loaded netowrks      : {}\n".format(self._loadedNetwork))
		out.close()

		# pre make results file
		out = open(self._directory+"results", "w")
		out.close()

		# copy configuration file
		cmd = "cp {} {}/configuration.xml".format(self._configurationFilePath, self._directory)
		os.system(cmd)

		return


	def printSummary(self):
		np.save(self._directory+"rms/mean.npy".format(self._summary_num_log),self._rms.mean)
		np.save(self._directory+"rms/var.npy".format(self._summary_num_log),self._rms.var)

		print('===============================================================')
		print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print("Elapsed time         : {:.2f}s".format(time.time()-self._startTime))
		print("Simulation time      : {}s".format(self._summary_sim_time))
		print("Training time        : {}s".format(self._summary_train_time))
		print("Session Name         : {}".format(self._sessionName))
		print("Logging Count        : {}".format(self._summary_num_log))
		print('Noise                : {:.3f}'.format(self._summary_noise_records[-1]))
		print('Learning rate        : {:.6f}'.format(self._learningRatePolicy))
		print('Total episode        : {}'.format(self._summary_num_episodes_total))
		print('Total trans          : {}'.format(self._summary_num_transitions_total))
		total_t_per_e = 0
		if self._summary_num_episodes_total is not 0:
			total_t_per_e = self._summary_num_transitions_total / self._summary_num_episodes_total
		print('Total trans per epi  : {:.2f}'.format(total_t_per_e))
		print('Episode              : {}'.format(self._summary_num_episodes_per_epoch))
		print('Transition           : {}'.format(self._summary_num_transitions_per_epoch))
		print('Trans per epi        : {:.2f}'.format(self._summary_transition_per_episodes[-1]))
		print('Max episode length   : {}'.format(self._summary_max_episode_length))
		print('Rewards per episodes : {:.2f}'.format(self._summary_total_rewards[-1]))

		if(self._useEvaluation):
			evaluation_t_per_e = self._summary_evaluation_num_transitions_per_epoch/self._numSlaves
			self._summary_evaluation_transition_per_episodes.append(evaluation_t_per_e)
			print('Eval trans per epi   : {:.2f}'.format(evaluation_t_per_e))
			print('Eval rew per epi     : {:.2f}'.format(self._summary_evaluation_total_rewards[-1]))
		print('===============================================================')


		# print plot
		if(self._useEvaluation):
			y_list = [[np.asarray(self._summary_evaluation_total_rewards_by_parts[0]), 'r'], 
						[np.asarray(self._summary_evaluation_mean_rewards), 'r_mean'],
						[np.asarray(self._summary_evaluation_transition_per_episodes), 'steps'], 
						[np.asarray(self._summary_evaluation_total_rewards_by_parts[1]), 'p'], 
						[np.asarray(self._summary_evaluation_total_rewards_by_parts[2]), 'v'], 
						[np.asarray(self._summary_evaluation_total_rewards_by_parts[3]), 'com'],
						[np.asarray(self._summary_evaluation_total_rewards_by_parts[4]), 'ee']]
			Plot(y_list,self._sessionName,1,path=self._directory+"result.png")

			for i in range(len(y_list)):
				y_list[i][0] = np.array(y_list[i][0])/np.array(self._summary_evaluation_transition_per_episodes)
			y_list[1][0] = np.asarray(self._summary_noise_records)
			y_list[1][1] = 'noise'

			Plot(y_list,self._sessionName+"_per_step",2,path=self._directory+"result_per_step.png")
		else:
			y_list = [[np.asarray(self._summary_total_rewards_by_parts[0]), 'r'], 
						[np.asarray(self._summary_mean_rewards), 'r_mean'],
						[np.asarray(self._summary_transition_per_episodes), 'steps'], 
						[np.asarray(self._summary_total_rewards_by_parts[1]), 'p'], 
						[np.asarray(self._summary_total_rewards_by_parts[2]), 'v'], 
						[np.asarray(self._summary_total_rewards_by_parts[3]), 'com'],
						[np.asarray(self._summary_total_rewards_by_parts[4]), 'ee']]
			Plot(y_list,self._sessionName,1,path=self._directory+"result.png")

			for i in range(len(y_list)):
				y_list[i][0] = np.array(y_list[i][0])/np.array(self._summary_transition_per_episodes)
			y_list[1][0] = np.asarray(self._summary_noise_records)
			y_list[1][1] = 'noise'

			Plot(y_list,self._sessionName+"_per_step",2,path=self._directory+"result_per_step.png")


		# log to file
		out = open(self._directory+"results", "a")
		out.write('===============================================================\n')
		out.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S\n"))
		out.write("Elapsed time         : {:.2f}s\n".format(time.time()-self._startTime))
		out.write("Simulation time      : {}s\n".format(self._summary_sim_time))
		out.write("Training time        : {}s\n".format(self._summary_train_time))
		out.write("Session Name         : {}\n".format(self._sessionName))
		out.write("Logging Count        : {}\n".format(self._summary_num_log))
		out.write('Noise                : {:.3f}\n'.format(self._summary_noise_records[-1]))
		out.write('Learning rate        : {:.6f}\n'.format(self._learningRatePolicy))
		out.write('Total episode        : {}\n'.format(self._summary_num_episodes_total))
		out.write('Total trans          : {}\n'.format(self._summary_num_transitions_total))
		out.write('Total trans per epi  : {:.2f}\n'.format(total_t_per_e))
		out.write('Episode              : {}\n'.format(self._summary_num_episodes_per_epoch))
		out.write('Transition           : {}\n'.format(self._summary_num_transitions_per_epoch))
		out.write('Trans per epi        : {:.2f}\n'.format(self._summary_transition_per_episodes[-1]))
		out.write('Max episode length   : {}\n'.format(self._summary_max_episode_length))
		out.write('Rewards per episodes : {:.2f}\n'.format(self._summary_total_rewards[-1]))

		if(self._useEvaluation):
			evaluation_t_per_e = self._summary_evaluation_num_transitions_per_epoch/self._numSlaves
			self._summary_evaluation_transition_per_episodes.append(evaluation_t_per_e)
			out.write('Eval trans per epi   : {:.2f}\n'.format(evaluation_t_per_e))
			out.write('Eval rew per epi     : {:.2f}\n'.format(self._summary_evaluation_total_rewards[-1]))
		out.write('===============================================================\n')
		out.close()


		# save reward
		self._adaptiveSampler.save(self._directory)


		# save network
		self.save(self._directory+"network")

		if self._useEvaluation:
			t_per_e = evaluation_t_per_e
			tr = self._summary_evaluation_total_rewards[-1]
		else:
			t_per_e = self._summary_transition_per_episodes[-1]
			tr = self._summary_total_rewards[-1]

		if t_per_e > self._smax:
			self._smax = t_per_e
			np.save(self._directory+"rms/mean_smax.npy",self._rms.mean)
			np.save(self._directory+"rms/var_smax.npy",self._rms.var)

			os.system("cp {}/network.data-00000-of-00002 {}/network-smax.data-00000-of-00002".format(self._directory, self._directory))
			os.system("cp {}/network.data-00001-of-00002 {}/network-smax.data-00001-of-00002".format(self._directory, self._directory))
			os.system("cp {}/network.index {}/network-smax.index".format(self._directory, self._directory))



		if tr > self._rmax:
			self._rmax = tr
			np.save(self._directory+"rms/mean_rmax.npy",self._rms.mean)
			np.save(self._directory+"rms/var_rmax.npy",self._rms.var)

			os.system("cp {}/network.data-00000-of-00002 {}/network-rmax.data-00000-of-00002".format(self._directory, self._directory))
			os.system("cp {}/network.data-00001-of-00002 {}/network-rmax.data-00001-of-00002".format(self._directory, self._directory))
			os.system("cp {}/network.index {}/network-rmax.index".format(self._directory, self._directory))

		self._summary_num_log = self._summary_num_log + 1

		return

	def save(self, path):
		self._ckpt.write(path)

	def restore(self, path):
		self._ckpt.restore(path)