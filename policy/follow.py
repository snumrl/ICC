import sys
from rl.TrackingController import TrackingController
import argparse
import os
from IPython import embed

import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = ''
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help='configuration file path')
	args = parser.parse_args()

	if args.config is None:
		print("Configuration file path required!")
		exit()
	with tf.device("/cpu:0"):
		tracking_controller = TrackingController()
		tracking_controller.initialize(configuration_filepath=args.config)
		tracking_controller.followReference()