import tensorflow as tf
import numpy as np
from xml.dom.minidom import parse

from IPython import embed

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
		self._sessionName = "test_session"

	def loadData(self, filename):
		print("TODO : Configurations.py loadData")
		# TODO
		with parse(filename) as doc:
			config = doc.getElementsByTagName("Configuration")[0]

			self._numSlaves = int(config.getAttribute("numSlaves"))
			self._sessionName = config.getAttribute("name")

			learn = config.getElementsByTagName("Learning")[0]

			self._motion = learn.getElementsByTagName("Motion")[0].firstChild.nodeValue

			self._gamma = float(learn.getElementsByTagName("Gamma")[0].firstChild.nodeValue)
			self._lambd = float(learn.getElementsByTagName("Lambd")[0].firstChild.nodeValue)
			self._clipRange = float(learn.getElementsByTagName("ClipRange")[0].firstChild.nodeValue)

			valueLayer = learn.getElementsByTagName("ValueLayer")[0]
			self._valueLayerSize = int(valueLayer.getAttribute("size"))
			self._valueLayerNumber = int(valueLayer.getAttribute("number"))

			policyLayer = learn.getElementsByTagName("PolicyLayer")[0]
			self._policyLayerSize = int(policyLayer.getAttribute("size"))
			self._policyLayerNumber = int(policyLayer.getAttribute("number"))

			self._activationFunction = learn.getElementsByTagName("ActivationFunction")[0].firstChild.nodeValue

			lr = learn.getElementsByTagName("LearningRate")[0]
			self._learningRatePolicy = float(lr.getAttribute("policy"))
			self._learningRatePolicyDecay = float(lr.getAttribute("decay"))
			self._learningRateValueFunction = float(lr.getAttribute("value"))

			self._batchSize = int(learn.getElementsByTagName("BatchSize")[0].firstChild.nodeValue)
			self._transitionsPerIteration = int(learn.getElementsByTagName("TransitionsPerIteration")[0].firstChild.nodeValue)

			trajectory = learn.getElementsByTagName("Trajectory")[0]
			self._trajectoryLength = int(trajectory.getAttribute("length"))

			origin = trajectory.getAttribute("origin")
			if origin == "True":
				self._useOrigin = True
			elif origin == "False":
				self._useOrigin = False
			else:
				print("Configurations.py : loadData : Unavailable origin type")
				exit()

			self._originOffset = int(trajectory.getAttribute("offset"))

			evaluation = learn.getElementsByTagName("Evaluation")[0].firstChild.nodeValue
			if evaluation == "True":
				self._useEvaluation = True
			elif evaluation == "False":
				self._useEvaluation = False
			else:
				print("Configurations.py : loadData : Unavailable evaluation type")
				exit()

			sim = config.getElementsByTagName("Simulation")[0]

			self._TCMotionSize = int(sim.getElementsByTagName("TCMotionSize")[0].firstChild.nodeValue)
			self._MGMotionSize = int(sim.getElementsByTagName("MGMotionSize")[0].firstChild.nodeValue)

		self._adaptiveSamplingSize = int(min(self._trajectoryLength/10, 1000))

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
		
	@property
	def TCMotionSize(self):
		return self._TCMotionSize
		
	@property
	def MGMotionSize(self):
		return self._MGMotionSize
	
	
	
	
	
	
	
	

