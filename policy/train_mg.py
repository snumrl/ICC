from rnn.RNNTraining import train
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
	motion_name = "walkrunfall"
	train(motion_name)