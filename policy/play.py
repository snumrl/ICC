import sys
from rl.TrackingController import TrackingController
import argparse
from argparse import RawTextHelpFormatter
import os
from IPython import embed

if __name__=="__main__":
	parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
	parser.add_argument('-ns', '--num_slaves', help='the number of slaves', default=8)
	parser.add_argument('-n', '--network', help='network file path')
	parser.add_argument('-t', '--network_type', help='type of the network\n  None : final network\n  rmax : maximum reward\n  smax : maximum step ', choices={'rmax', 'smax'})
	args = parser.parse_args()

	if args.network is None:
		print("Network path required!")
		exit()

	tracking_controller = TrackingController()
	tracking_controller.initialize(
		configuration_filepath='{}/configuration.xml'.format(args.network),
		session_name="play",
		num_slaves=args.num_slaves,
		trajectory_length=2000,
		origin=False,
		origin_offset=0
	)
	tracking_controller.loadNetworks(args.network, args.network_type)
	tracking_controller.play()