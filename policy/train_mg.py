from rnn.RNNTraining import train
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
	parser.add_argument('-m', '--motion', help='motion name(folder name)')
	args = parser.parse_args()
	train(args.motion)