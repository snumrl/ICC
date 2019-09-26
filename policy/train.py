import sys
from rl.TrackingController import TrackingController
import argparse
import os
from IPython import embed

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--session_name', help='session name')
	parser.add_argument('-c', '--config', help='configuration file path')
	parser.add_argument('-ns', '--num_slaves', help='the number of slaves', default=8)
	parser.add_argument('-n', '--network', help='network file path')
	parser.add_argument('-t', '--network_type', help='type of the network, None : final network, rmax : maximum reward, smax : maximum step ', choices={'rmax', 'smax'})
	args = parser.parse_args()
	
	if args.session_name is None:
		print("Session name required!")
		exit()
	if args.config is None:
		print("Configuration file path required!")
		exit()

	tracking_controller = TrackingController()
	tracking_controller.initialize(
		configuration_filepath=args.config,
		session_name=args.session_name,
		num_slaves=args.num_slaves,
		trajectory_length=2000, 
		origin=True, 
		origin_offset=0,
		use_evaluation=False
	)
	if args.network is not None:
		tracking_controller.loadNetworks(args.network, args.network_type)
	tracking_controller.runTraining(10)