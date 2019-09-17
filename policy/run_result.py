import sys
from PPO import PPO
import argparse
import os
from IPython import embed
import CharacterConfigurations


server_list = {}
server_list["loco1"] = "soohwan@loco1.snu.ac.kr:~/workspace/DeepPhysics/output"
server_list["loco2"] = "soohwan@loco2.snu.ac.kr:~/workspace/DeepPhysics/output"
server_list["rhs1"] = "-P 28227 root@106.10.41.197:~/DeepPhysics/output"
server_list["soohwan1"] = "-P 28227 root@101.101.160.205:~/DeepPhysics/output"
server_list["soohwan2"] = "-P 28228 root@101.101.160.205:~/DeepPhysics/output"
server_list["lsm1"] = "-P 1044 root@106.10.52.137:~/DeepPhysics/output"
server_list["lsm3"] = "-P 1045 root@106.10.52.137:~/DeepPhysics/output"
server_list["aws1"] = "ubuntu@ec2-34-214-161-91.us-west-2.compute.amazonaws.com:~/DeepPhysics/output"
server_list["aws2"] = "ubuntu@ec2-34-211-48-201.us-west-2.compute.amazonaws.com:~/DeepPhysics/output"
server_list["aws3"] = "ubuntu@ec2-18-237-69-153.us-west-2.compute.amazonaws.com:~/DeepPhysics/output"

os.environ['CUDA_VISIBLE_DEVICES'] = ''
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-s','--server',help='server name')
	parser.add_argument('-l','--local',help='local path')
	parser.add_argument('-t', '--test_number', help='test number')
	parser.add_argument('-i', '--iter_number', help='iteration number')
	args =parser.parse_args()
	if args.server is not None:
		if args.server in server_list:
			cmd = "scp {}/test{}/parameters ./server_network/".format(server_list[args.server], args.test_number)
			os.system(cmd)
			cmd = "scp {}/test{}/network-{}\* ./server_network/".format(server_list[args.server], args.test_number, args.iter_number)
			os.system(cmd)
			cmd = "scp {}/test{}/rms/mean_{}.npy ./server_network/mean.npy".format(server_list[args.server], args.test_number, args.iter_number)
			os.system(cmd)
			cmd = "scp {}/test{}/rms/var_{}.npy ./server_network/var.npy".format(server_list[args.server], args.test_number, args.iter_number)
			os.system(cmd)

		path = "./server_network/network-{}".format(args.iter_number)

		parameter_path = "./server_network/parameters"
		parametre_file = open(parameter_path, "r")
		lines = parametre_file.readlines()
		if(lines[4][:9] != "num state"):
			print("parameter format is not matched")
			exit()
		num_state = int(lines[4][12:-1])
		parametre_file.close()

		# generate rms folder
		directory = "../rms/{}/".format(num_state)
		if not os.path.exists(directory):
			os.mkdir(directory)

		os.system("cp ./server_network/mean.npy ../rms/{}/mean.npy".format(num_state))
		os.system("cp ./server_network/var.npy ../rms/{}/var.npy".format(num_state))

		ppo = PPO()
		ppo.InitializeForTraining(env_name='humanoid',num_slaves=4, use_trajectory=True, 
					num_trajectories=1,	
					origin=True, frame=20000, origin_offset=0,
					motion=CharacterConfigurations.motion,
					pretrain=path,
					use_adaptive_initial_state=False,
					detail="Run") # do not change detail for run result
		ppo.RunWithSavedData(rsi=False)

	elif args.test_number is not None and args.iter_number is not None:
		# read num_state
		parameter_path = "../output/test{}/parameters".format(args.test_number)

		parametre_file = open(parameter_path, "r")
		lines = parametre_file.readlines()
		if(lines[4][:9] != "num state"):
			print("parameter format is not matched")
			exit()
		num_state = int(lines[4][12:-1])
		parametre_file.close()

		# generate rms folder
		directory = "../rms/{}/".format(num_state)
		if not os.path.exists(directory):
			os.mkdir(directory)

		# copy rms files
		rms_mean = "../output/test{}/rms/mean_{}.npy".format(args.test_number, args.iter_number)
		rms_var  = "../output/test{}/rms/var_{}.npy".format(args.test_number, args.iter_number)

		os.system("cp {} ../rms/{}/mean.npy".format(rms_mean, num_state))
		os.system("cp {} ../rms/{}/var.npy".format(rms_var, num_state))

		# network path
		path = "../output/test{}/network-{}".format(args.test_number, args.iter_number)

		ppo = PPO()
		ppo.InitializeForTraining(env_name='humanoid',num_slaves=8	, use_trajectory=True,
					num_trajectories=1,	
					origin=True, frame=2000, origin_offset=10000,
					motion=CharacterConfigurations.motion,
					pretrain=path,
					use_adaptive_initial_state=False,
					detail="Run") # do not change detail for run result
		ppo.RunWithSavedData(rsi=False)

