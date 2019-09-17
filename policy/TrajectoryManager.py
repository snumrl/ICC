from rnn.RNNController import RNNController
from rnn.RNNManager import RNNManager
import numpy as np
from copy import deepcopy
from IPython import embed
import os
import CharacterConfigurations

class TrajectoryManager(object):
	def __init__(self, motion="walk_extension", num_time_piecies=20):
		self.num_slaves = 4
		self.motion = motion
		self.rnn_manager = RNNManager(self.num_slaves, self.motion)
		self.num_time_piecies = num_time_piecies

	def setPPO(self, ppo):
		self.ppo= ppo
		self.last_time =-1
		self.last_t_index= -1
		self.last_fx =-1
		self.first_MCHT= True

	def generateTrajectories(self, num_trajectories=32, frame=2000, origin=False, origin_offset=0, useBothOriginNRandom= False):
		self.frame = frame
		self.num_trajectories = num_trajectories

		self.trajectories = [None]*self.num_trajectories
		self.goal_trajectories = [None]*self.num_trajectories

		self.epi_rewards_mean = np.zeros(self.num_trajectories)
		self.epi_rewards_count = np.zeros(self.num_trajectories)

		self.rewards_per_time_pieces = np.zeros((self.num_trajectories, self.num_time_piecies))
		self.reward_counts_per_time_pieces = np.zeros((self.num_trajectories, self.num_time_piecies))

		self.rewards_per_time_pieces_EPI_LEN = np.zeros((self.num_trajectories, self.num_time_piecies))
		self.rewards_per_time_pieces_REW = np.zeros((self.num_trajectories, self.num_time_piecies))

		self.max_episode_length = np.zeros(self.num_trajectories)

		traj_index=0
		if useBothOriginNRandom: # 0: origin, else: #num_trajectories of rnn trajectories from random target
			self.trajectories[traj_index], self.goal_trajectories[traj_index] = self.rnn_manager.getOriginalTrajectory(self.frame, origin_offset)
			traj_index= traj_index+1

		if origin: ##SM) original(used in RNN training - y) data
			if num_trajectories != 1:
				print("use original trajectory but num_trajectories is not 1!")
				exit()
			self.trajectories[0], self.goal_trajectories[0] = self.rnn_manager.getOriginalTrajectory(self.frame, origin_offset)

			return

		##SM) generate traj using RNN from random targets
		if self.motion == "walk_extension":
			if self.num_trajectories == 1:
				traj_filename = "../trajectories/walk/{}/traj.npy".format(self.num_trajectories)
				goal_filename = "../trajectories/walk/{}/goal.npy".format(self.num_trajectories)
				if os.path.exists(traj_filename) and os.path.exists(goal_filename):
					print("Loading trajectories from {}".format(traj_filename))
					traj = np.load(traj_filename)
					goal_traj = np.load(goal_filename)
					if traj.shape[1] < frame:
						print("motion is too short, required : {}, maximum : {}".format(frame, traj.shape[1]))
						# exit()
						frame = traj.shape[1]
					self.trajectories[0] = traj[0][:frame]
					self.goal_trajectories[0] = goal_traj[0][:frame]
				else:
					traj, goal_traj = self.rnn_manager.getTrajectory(frame)
					self.trajectories[0] = traj[0]
					self.goal_trajectories[0] = goal_traj[0]

				return
			traj_filename = "../trajectories/walk/{}/{}_traj.npy".format(self.num_trajectories, frame)
			goal_filename = "../trajectories/walk/{}/{}_goal.npy".format(self.num_trajectories, frame)
			if os.path.exists(traj_filename) and os.path.exists(goal_filename):
				print("Loading trajectories from {}".format(traj_filename))
				traj = np.load(traj_filename)
				goal_traj = np.load(goal_filename)
				for i in range(len(traj)):
					self.trajectories[i] = traj[i]
					self.goal_trajectories[i] = goal_traj[i]
			else:
				traj, goal_traj = self.rnn_manager.getTrajectory(frame, [[5000., 0.],[-5000, 0.0],[0.0, 5000],[0.0, -5000]])
				for i in range(4):
					self.trajectories[i] = traj[i]
					self.goal_trajectories[i] = goal_traj[i]
				for j in range(int(self.num_trajectories/self.num_slaves) - 1):
					traj, goal_traj = self.rnn_manager.getTrajectory(frame)
					for i in range(4):
						self.trajectories[4*(j+1)+i] = traj[i]
						self.goal_trajectories[4*(j+1)+i] = goal_traj[i]

				residual = self.num_trajectories%self.num_slaves
				if residual != 0:
					traj, goal_traj = self.rnn_manager.getTrajectory(frame)
					for i in range(residual):
						c = -(i+1)
						self.trajectories[c] = traj[c]
						self.goal_trajectories[c] = goal_traj[c]
		elif self.motion == "basketball" :
			if self.num_trajectories == 1:
				traj_filename = "../trajectories/{}/{}/traj.npy".format(self.motion, self.num_trajectories)
				goal_filename = "../trajectories/{}/{}/goal.npy".format(self.motion,self.num_trajectories)
				if os.path.exists(traj_filename) and os.path.exists(goal_filename):
					print("Loading trajectories from {}".format(traj_filename))
					traj = np.load(traj_filename)
					goal_traj = np.load(goal_filename)
					if traj.shape[1] < frame:
						print("motion is too short, required : {}, maximum : {}".format(frame, traj.shape[1]))
						# exit()
						frame = traj.shape[1]
					self.trajectories[0] = traj[0][:frame]
					self.goal_trajectories[0] = goal_traj[0][:frame]
				else:
					traj, goal_traj = self.rnn_manager.getTrajectory(frame)
					self.trajectories[0] = traj[0]
					self.goal_trajectories[0] = goal_traj[0]

				return
			traj_filename = "../trajectories/{}/{}/{}_traj.npy".format(self.motion, self.num_trajectories, frame)
			goal_filename = "../trajectories/{}/{}/{}_goal.npy".format(self.motion, self.num_trajectories, frame)
			if os.path.exists(traj_filename) and os.path.exists(goal_filename):
				print("Loading trajectories from {}".format(traj_filename))
				traj = np.load(traj_filename)
				goal_traj = np.load(goal_filename)
				for i in range(len(traj)):
					self.trajectories[i] = traj[i]
					self.goal_trajectories[i] = goal_traj[i]
			else:
				for j in range(int(self.num_trajectories/self.num_slaves)):
					traj, goal_traj = self.rnn_manager.getTrajectory(frame)
					for i in range(4):
						self.trajectories[4*j+i] = traj[i]
						self.goal_trajectories[4*j+i] = goal_traj[i]

				residual = self.num_trajectories%self.num_slaves
				if residual != 0:
					traj, goal_traj = self.rnn_manager.getTrajectory(frame)
					for i in range(residual):
						c = -(i+1)
						self.trajectories[c] = traj[c]
						self.goal_trajectories[c] = goal_traj[c]
		else:
			if self.num_trajectories == 1:
				traj_filename = "../trajectories/{}/{}/traj.npy".format(self.motion, self.num_trajectories)
				goal_filename = "../trajectories/{}/{}/goal.npy".format(self.motion, self.num_trajectories)
				if os.path.exists(traj_filename) and os.path.exists(goal_filename):
					print("Loading trajectories from {}".format(traj_filename))
					traj = np.load(traj_filename)
					goal_traj = np.load(goal_filename)
					if traj.shape[1] < frame:
						print("motion is too short, required : {}, maximum : {}".format(frame, traj.shape[1]))
						# exit()
						frame = traj.shape[1]
					self.trajectories[traj_index] = traj[0][:frame]
					self.goal_trajectories[traj_index] = goal_traj[0][:frame]
				else:
					traj, goal_traj = self.rnn_manager.getTrajectory(frame)
					self.trajectories[traj_index] = traj[0]
					self.goal_trajectories[traj_index] = goal_traj[0]

				return

			traj_filename = "../trajectories/{}/{}/{}_traj.npy".format(self.motion, self.num_trajectories, frame)
			goal_filename = "../trajectories/{}/{}/{}_goal.npy".format(self.motion, self.num_trajectories, frame)
			if os.path.exists(traj_filename) and os.path.exists(goal_filename):
				print("Loading trajectories from {}".format(traj_filename))
				traj = np.load(traj_filename)
				goal_traj = np.load(goal_filename)
				for i in range(len(traj)):
					self.trajectories[i] = traj[i]
					self.goal_trajectories[i] = goal_traj[i]
			else:
				traj, goal_traj = self.rnn_manager.getTrajectory(frame, [[5000., 0.],[-5000, 0.0],[0.0, 5000],[0.0, -5000]])
				for i in range(4):
					self.trajectories[i] = traj[i]
					self.goal_trajectories[i] = goal_traj[i]
				for j in range(int(self.num_trajectories/self.num_slaves) - 1):
					traj, goal_traj = self.rnn_manager.getTrajectory(frame)
					for i in range(4):
						self.trajectories[4*(j+1)+i] = traj[i]
						self.goal_trajectories[4*(j+1)+i] = goal_traj[i]

				residual = self.num_trajectories%self.num_slaves
				if residual != 0:
					traj, goal_traj = self.rnn_manager.getTrajectory(frame)
					for i in range(residual):
						c = -(i+1)
						self.trajectories[c] = traj[c]
						self.goal_trajectories[c] = goal_traj[c]

	# def reset(self, frame):
	def save(self, path):
		np.save(path+"traj.npy", np.array(self.trajectories))
		np.save(path+"goal.npy", np.array(self.goal_trajectories))

	def saveTimeWeight(self, path, evaluation_index):
		np.save(path+"timeweight_EPI_LEN.npy", np.array(self.rewards_per_time_pieces_EPI_LEN))
		np.save(path+"timeweight_REW.npy", np.array(self.rewards_per_time_pieces_REW))
		np.save(path+"timeweight_count.npy", np.array(self.reward_counts_per_time_pieces))

		np.save(path+"timeweight_EPI_LEN_{}.npy".format(evaluation_index), np.array(self.rewards_per_time_pieces_EPI_LEN))
		np.save(path+"timeweight_REW_{}.npy".format(evaluation_index), np.array(self.rewards_per_time_pieces_REW))
		np.save(path+"timeweight_count_{}.npy".format(evaluation_index), np.array(self.reward_counts_per_time_pieces))

	def load(self, path, count=None):
		print("Loading trajectories from {}".format(path))
		tr = np.load(path+"traj.npy")
		goal_tr = np.load(path+"goal.npy")

		print("trajectory count : {}".format(tr.shape[0]))
		print("frame count : {}".format(tr.shape[1]))
		self.frame = tr.shape[1]

		num = len(tr)
		if count is not None:
			num = count

		self.num_trajectories = num
		self.trajectories = [None]*self.num_trajectories
		self.goal_trajectories = [None]*self.num_trajectories

		self.epi_rewards_mean = np.zeros(self.num_trajectories)
		self.epi_rewards_count = np.zeros(self.num_trajectories)

		self.rewards_per_time_pieces = np.zeros((self.num_trajectories, self.num_time_piecies))
		self.reward_counts_per_time_pieces = np.zeros((self.num_trajectories, self.num_time_piecies))

		self.rewards_per_time_pieces_EPI_LEN = np.zeros((self.num_trajectories, self.num_time_piecies))
		self.rewards_per_time_pieces_REW = np.zeros((self.num_trajectories, self.num_time_piecies))

		self.max_episode_length = np.zeros(self.num_trajectories)

		for i in range(num):
			self.trajectories[i] = tr[i]
			self.goal_trajectories[i] = goal_tr[i]


	def updateEpiReward(self, index, t_index, rew, length, te):
		# if CharacterConfigurations.USE_MCHT:
		# 	return

		rew = float(rew)
		delta = rew - self.epi_rewards_mean[index]
		self.epi_rewards_mean[index] += delta * (1.0/(self.epi_rewards_count[index]+1))
		self.epi_rewards_count[index] += 1


		control_hz = 30.
		motion_hz = 30.
		if te:
			elapsed_time = length/control_hz
		else:
			elapsed_time = length/control_hz - 1.0
		elapsed_t_index = max(0,int(elapsed_time * motion_hz / (float(self.frame) / self.num_time_piecies)))
		for i in range(elapsed_t_index+1):
			if t_index + i >= self.num_time_piecies:
				break
			self.rewards_per_time_pieces_EPI_LEN[index, t_index+i] += 1.0
			if CharacterConfigurations.USE_EPI_LEN:
				self.rewards_per_time_pieces[index, t_index+i] += 1.0

		if te:
			rew *= float(self.num_time_piecies) / (self.num_time_piecies - t_index)
		delta = rew - self.rewards_per_time_pieces_REW[index, t_index]
		self.rewards_per_time_pieces_REW[index, t_index] += delta * (1.0/(self.reward_counts_per_time_pieces[index, t_index]+1))
		if not CharacterConfigurations.USE_EPI_LEN:
			self.rewards_per_time_pieces[index, t_index] += delta * (1.0/(self.reward_counts_per_time_pieces[index, t_index]+1))

		self.reward_counts_per_time_pieces[index, t_index] += 1

	def selectIndex(self):
		e = np.exp(-(self.epi_rewards_mean-self.epi_rewards_mean.min())*0.1)
		# e = 1.0/(np.array(self.epi_rewards_mean)+1.0)
		if any(self.epi_rewards_count<10):
			e = np.ones(self.epi_rewards_count.shape)
		tot = e.sum()
		e = e/tot

		index = self.num_trajectories-1
		cur = 0
		ran = np.random.uniform(0.0,1.0)
		for i in range(self.num_trajectories):
			cur += e[i]
			if ran < cur:
				index = i
				break

		return index

	def selectTime(self, index, slave_index=0): #traj의 index.. 지금은 전부다 0이다.
		if CharacterConfigurations.USE_MCHT:
			return self.selectTimeMCHT(index, slave_index)

		if CharacterConfigurations.USE_EXP_WEIGHT:
			e = (self.rewards_per_time_pieces[index]-self.rewards_per_time_pieces[index].min())
			e = np.clip(e, a_min=None, a_max=50)
			e = np.exp(-e*0.2)
		else:
			e = 1.0/(np.array(self.rewards_per_time_pieces[index])+1.0)
		if any(self.reward_counts_per_time_pieces[index]<10):
			e = np.ones(e.shape)
		tot = e.sum()
		e = e/tot

		t_index = self.num_time_piecies-1
		cur = 0
		ran = np.random.uniform(0.0,1.0)
		for i in range(self.num_time_piecies):
			cur += e[i]
			if ran < cur:
				t_index = i
				break

		time = (t_index + np.random.uniform(0.0, 1.0))/self.num_time_piecies

		return t_index, time

	def selectTimeMCHT(self, index, slave_index):
		# ran = np.random.uniform(0.0, 1.0)
		# self.ppo.Env.ResetWithTime(slave_index, ran)
		# state= np.array([self.ppo.Env.GetState(slave_index)])
		# state = self.ppo.RMS.apply(state)
		# value = self.ppo.critic.GetValue(state)[0]
		# fx= np.exp(-value)
		ran_walk= np.random.uniform(0.0, 1.0)
		self.ppo.Env.ResetWithTime(slave_index, ran_walk)
		state= np.array([self.ppo.Env.GetState(slave_index)])
		state = self.ppo.RMS.apply(state)
		value = self.ppo.critic.GetValue(state)[0]
		fx_walk= np.exp(-value*0.2)

		if self.first_MCHT==True:
			t_index= np.minimum(int(ran_walk*self.num_time_piecies),self.num_time_piecies-1)
			time= ran_walk
			self.last_time= time
			self.last_t_index= t_index
			self.last_fx= fx_walk
			self.first_MCHT= False
			return t_index, time

		accept_ratio= fx_walk/self.last_fx
		accept_ran= np.random.uniform(0.0, 1.0)

		# print(accept_ratio)
		# embed()
		if(accept_ran< accept_ratio):
			t_index= np.minimum(int(ran_walk*self.num_time_piecies),self.num_time_piecies-1)
			time= ran_walk
			self.last_time= time
			self.last_t_index= t_index
			self.last_fx= fx_walk
			return t_index, time
		else:
			return self.last_t_index, self.last_time

	def resetTimeWeight(self):		
		for index in range(self.num_trajectories):
			if all(self.reward_counts_per_time_pieces[index]>10):
				self.rewards_per_time_pieces[index] = np.zeros(self.num_time_piecies)
				self.reward_counts_per_time_pieces[index] = np.zeros(self.num_time_piecies)


	def getRandomTime(self, t_index):
		time = (t_index + np.random.uniform(0.0, 1.0))/self.num_time_piecies

		return time


	def getTrajectory(self, index=None, t_index=None):
		if index is None:
			index = self.selectIndex()
		if t_index is None:
			t_index, time = self.selectTime(index)
		else:
			time = float(t_index)/self.num_time_piecies

		return self.trajectories[index], self.goal_trajectories[index], index, t_index, time
