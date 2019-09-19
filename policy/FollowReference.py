import sys
from TrackingController import TrackingController
import argparse
import os
from IPython import embed

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help='configuration file path')
	args = parser.parse_args()

	if args.config is None:
		print("Configuration file path required!")
		exit()
	tracking_controller = TrackingController()
	tracking_controller.initialize(configuration_filepath=args.config)
	tracking_controller.followReference()