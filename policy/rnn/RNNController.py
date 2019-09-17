import math

import numpy as np
import tensorflow as  tf
from tensorflow.python import pywrap_tensorflow
import rnn.Configurations
from util.dataLoader import loadData

from util.Pose2d import Pose2d
from IPython import embed
from copy import deepcopy

import CharacterConfigurations

class RNNController(object):
    def __init__(self, motion, num_slaves):
        self.motion = motion
        self.num_slaves = num_slaves
        self.config = rnn.Configurations.get_config(motion)
        self.config.load_normal_data(motion)
        self.pose = []
        for _ in range(self.num_slaves):
            self.pose.append(Pose2d())

        # self.LoadPreTrainedVariables("../motions/%s/train/ckpt"%(motion))


        self.outputs = []

    def resetAll(self):
        for i in range(self.num_slaves):
            self.pose[i] = Pose2d()

        self.state = None
        self.outputs = []

    # This function load original traj. and original goal traj. without transfer goal from local to global.
    def getOriginalTrajectoryWithLocalGoal(self, frame, origin_offset=0):
        x_dat = loadData("{}/data/xData.dat".format(self.motion))
        y_dat = loadData("{}/data/yData.dat".format(self.motion))

        x_dat = x_dat[1+origin_offset:frame+1+origin_offset]
        y_dat = y_dat[1+origin_offset:frame+1+origin_offset]

        x_dat = np.array([self.config.x_normal.get_data_with_zeros(self.config.x_normal.de_normalize_l(x)) for x in x_dat])
        y_dat = np.array([self.config.y_normal.get_data_with_zeros(self.config.y_normal.de_normalize_l(y)) for y in y_dat])


        self.resetAll()

        trajectories = []
        targets = []

        for x, y in zip(x_dat, y_dat):
            if self.motion == "basketball":
                localPose = Pose2d(x[12:14], x[14:])
            elif self.motion == "walkfall":
                localPose = Pose2d(x[:2])
            elif self.motion == "walkrunfall":
                localPose = Pose2d(x[:2])
            elif self.motion.startswith("walkfall_prediction"):
                localPose = Pose2d(x[:2])
            else:
                localPose = Pose2d(x)

            targets.append(localPose.p)
            trajectories.append(self.get_positions(y, 0))

        trajectories = np.asarray(trajectories, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        return trajectories, targets


    def getOriginalTrajectory(self, frame, origin_offset=0): # return global goals
        x_dat = loadData("../motions/{}/data/xData.dat".format(self.motion))
        y_dat = loadData("../motions/{}/data/yData.dat".format(self.motion))

        x_dat = x_dat[1+origin_offset:frame+1+origin_offset]
        y_dat = y_dat[1+origin_offset:frame+1+origin_offset]

        x_dat = np.array([self.config.x_normal.get_data_with_zeros(self.config.x_normal.de_normalize_l(x)) for x in x_dat])
        y_dat = np.array([self.config.y_normal.get_data_with_zeros(self.config.y_normal.de_normalize_l(y)) for y in y_dat])


        self.resetAll()

        trajectories = []
        targets = []

        for x, y in zip(x_dat, y_dat):
            if self.motion == "basketball":
                localPose = Pose2d(x[12:14], x[14:])
            elif self.motion == "walkfall":
                localPose = Pose2d(x[:2])
            elif self.motion == "walkrunfall":
                localPose = Pose2d(x[:2])
            elif self.motion.startswith("walkfall_prediction"):
                localPose = Pose2d(x[:2])
            else:
                localPose = Pose2d(x)

            targets.append(self.pose[0].localToGlobal(localPose).p)
            trajectories.append(self.get_positions(y, 0))

        trajectories = np.asarray(trajectories, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        return trajectories, targets


    def get_positions(self, output, index):
        if self.motion == "basketball":
            output = output[8:]
        foot_contact = output[:2]
        output = output[2:]
        # move root
        self.pose[index] = self.pose[index].transform(output)

        points = [[0, output[3], 0]]
        output = output[4:]
        dof = CharacterConfigurations.INPUT_MOTION_SIZE
        positions = np.zeros(dof)

        positions[0:3] = self.pose[index].global_point_3d(points[0])
        positions[3:4] = self.pose[index].rotatedAngle()
        output = output[57:]
        positions[4:52] = output[0:48]


        positions[52] = foot_contact[0]     # left foot contact
        positions[53] = foot_contact[1]     # right foot contact

        return positions
