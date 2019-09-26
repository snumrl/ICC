import sys
from rl.TrackingController import TrackingController
import argparse
from argparse import RawTextHelpFormatter
import os
from IPython import embed

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)
	parser.add_argument('-n', '--network', help='network file path')
	parser.add_argument('-t', '--network_type', help='type of the network\n  None : final network\n  rmax : maximum reward\n  smax : maximum step ', choices={'rmax', 'smax'})
	args = parser.parse_args()

	tracking_controller = TrackingController()
	tracking_controller.initialize(
		configuration_filepath='{}/configuration.xml'.format(args.network),
		trajectory_length=2000,
		origin=False,
		origin_offset=0
	)
	if args.network is not None:
		tracking_controller.loadNetworks(args.network, args.network_type)
	tracking_controller.play()