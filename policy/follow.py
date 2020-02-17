import sys
from rl.TrackingController import TrackingController
import argparse
import os
from IPython import embed

import tensorflow as tf
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help='configuration file path')
	args = parser.parse_args()

	if args.config is None:
		print("Configuration file path required!")
		exit()
		
	with tf.device("/cpu:0"):
		tracking_controller = TrackingController()
		tracking_controller.initialize(
			configuration_filepath=args.config,
			session_name="follow",
			trajectory_length=300,
			origin=True,
			origin_offset=0
		)
		tracking_controller.followReference()