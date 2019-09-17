import tensorflow as tf

class Configurations:
	_instance = None

	@classmethod
	def __getInstance(cls):
		return cls.__instance

	@classmethod
	def instance(cls, *args, **kargs):
		print("Configurations instance is created")
		cls.__instance = cls(*args, **kargs)
		cls.instance = cls.__getInstance
		return cls.__instance

	def __init__(self):
		self._policyLayerSize = 512
		self._policyLayerNumber = 4

		self._activationFunction = tf.nn.relu
		self._kernelInitializationFunction = tf.keras.initializers.GlorotUniform()

		self._valueLayerSize = 256
		self._valueLayerNumber = 2

		self._numSlaves = 8
		self._motion = "walkrunfall"

		self._gamma = 0.95
		self._lambd = 0.95
		self._clipRange = 0.2

		self._learningRatePolicy = 2e-4
		self._learningRatePolicyDecay = 0.9993
		self._learningRateValueFunction = 1e-3

		self._batchSize = 1024
		self._transitionsPerIteration = 20000

		self._trajectoryLength = 2000
		self._useOrigin = True
		self._originOffset = 0

		self._adaptiveSamplingSize = 1000

		self._useEvaluation = False
		self._sessionName = "test_session"

	def loadData(self, filename):
		print("TODO : Configurations.py loadData")
		# TODO
		return

	@property
	def policyLayerSize(self):
		return self._policyLayerSize

	@property
	def policyLayerNumber(self):
		return self._policyLayerNumber

	@property
	def activationFunction(self):
		return self._activationFunction
	
	@property
	def kernelInitializationFunction(self):
		return self._kernelInitializationFunction

	@property
	def valueLayerSize(self):
		return self._valueLayerSize

	@property
	def valueLayerNumber(self):
		return self._valueLayerNumber

	@property
	def numSlaves(self):
		return self._numSlaves
	
	@property
	def motion(self):
		return self._motion

	@property
	def gamma(self):
		return self._gamma

	@property
	def lambd(self):
		return self._lambd

	@property
	def clipRange(self):
		return self._clipRange

	@property
	def learningRatePolicy(self):
		return self._learningRatePolicy

	@property
	def learningRatePolicyDecay(self):
		return self._learningRatePolicyDecay
	

	@property
	def learningRateValueFunction(self):
		return self._learningRateValueFunction

	@property
	def batchSize(self):
		return self._batchSize
	
	@property
	def transitionsPerIteration(self):
		return self._transitionsPerIteration
	
	@property
	def trajectoryLength(self):
		return self._trajectoryLength

	@property
	def useOrigin(self):
		return self._useOrigin

	@property
	def originOffset(self):
		return self._originOffset
	
	@property
	def adaptiveSamplingSize(self):
		return self._adaptiveSamplingSize
	

	@property
	def useEvaluation(self):
		return self._useEvaluation
	
	@property
	def sessionName(self):
		return self._sessionName
		
	
	
	
	
	
	
	
	

