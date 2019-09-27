#pragma once

#include <cmath>
#include <vector>
#include <tinyxml.h>

#include "Utils.h"

namespace ICC
{


enum class ReferenceType
{
	FIXED,
	INTERACTIVE,
};

enum class RewardType
{
	SUMMATION,
	MULTIPLICATION,
};

class Configurations
{
public:
	static Configurations& instance(){
		static Configurations* instance = new Configurations();
		return *instance;
	}

	double getJointDamping(){
		return this->mJointDamping;
	}

	int getControlHz(){
		return this->mControlHz;
	}

	int getMotionHz(){
		return this->mMotionHz;
	}

	int getSimulationHz(){
		return this->mSimulationHz;
	}

	ReferenceType getReferenceType(){
		return this->mReferenceType;
	}

	bool useEarlyTermination(){
		return this->mUseEarlyTermination;
	}

	int getFutureCount(){
		return this->mFutureCount;
	}

	double getRootHeightOffset(){
		return this->mRootHeightOffset;
	}

	int getNumActors(){
		return this->mNumActors;
	}

	std::vector<int>& getPredictionsInState(){
		return this->mPredictionsInState;
	}

	RewardType getRewardType(){
		return this->mRewardType;
	}

	double getRootHeightLowerLimit(){
		return this->mRootHeightLowerLimit;
	}

	double getRootHeightUpperLimit(){
		return this->mRootHeightUpperLimit;
	}

	double getRootDiffThreshold(){
		return this->mRootDiffThreshold;
	}

	double getRootAngleDiffThreshold(){
		return this->mRootAngleDiffThreshold;
	}

	int getTCMotionSize(){
		return this->mTCMotionSize;
	}

	int getMGMotionSize(){
		return this->mMGMotionSize;
	}

	void setReferenceType(ReferenceType rt){
		this->mReferenceType = rt;
	}

	void setEarlyTermination(bool et){
		this->mUseEarlyTermination = et;
	}


	void LoadConfigurations(std::string filename){
		TiXmlDocument doc;
		if(!doc.LoadFile(filename)){
			std::cout << "Can't open file : " << filename << std::endl;
			return;
		}

		TiXmlElement *config = doc.FirstChildElement("Configuration");

		TiXmlElement *sim = config->FirstChildElement("Simulation");

		// joint damping
		this->mJointDamping = atof(sim->FirstChildElement("JointDamping")->GetText());

		// control, motion, simulation hz
		this->mControlHz = atoi(sim->FirstChildElement("ControlHz")->GetText());
		this->mMotionHz = atoi(sim->FirstChildElement("MotionHz")->GetText());
		this->mSimulationHz = atoi(sim->FirstChildElement("SimulationHz")->GetText());

		// root height offset
		this->mRootHeightOffset = atof(sim->FirstChildElement("RootHeightOffset")->GetText());

		// reward type
		std::string reward_type = sim->FirstChildElement("RewardType")->GetText();
		if(reward_type == "Mul"){
			this->mRewardType = RewardType::MULTIPLICATION;
		}
		else if(reward_type == "Sum"){
			this->mRewardType = RewardType::SUMMATION;
		}
		else{
			std::cout << "environment::Configurations.h : Unspecified reward type!" << std::endl;
			exit(0);
		}

		// early termination
		std::string early_termination = sim->FirstChildElement("EarlyTermination")->GetText();
		if(early_termination=="True"){
			this->mUseEarlyTermination = true;
		}
		else if(early_termination=="False"){
			this->mUseEarlyTermination = false;	
		}
		else{
			std::cout << "environment::Configurations.h : Unspecified early termination!" << std::endl;
			exit(0);
		}
		
		// motion size of TC and MG. It depends on the character configuration and motion generator design
		this->mTCMotionSize = atoi(sim->FirstChildElement("TCMotionSize")->GetText());
		this->mMGMotionSize = atoi(sim->FirstChildElement("MGMotionSize")->GetText());

		// predictions included in stae
		// 0 : very next prediction
		this->mPredictionsInState = ICC::Utils::splitToInt(sim->FirstChildElement("Predictions")->GetText());

		// required futures for interactive mode
		this->mFutureCount = 1 + *std::max_element(this->mPredictionsInState.begin(), this->mPredictionsInState.end());

		// terminal condition
		TiXmlElement *terminal = sim->FirstChildElement("TerminalCondition");
		this->mRootDiffThreshold = atof(terminal->FirstChildElement("RootDiff")->GetText());
		this->mRootAngleDiffThreshold = atof(terminal->FirstChildElement("RootAngleDiff")->GetText());
		this->mRootHeightLowerLimit = atof(terminal->FirstChildElement("RootHeight")->Attribute("lower"));
		this->mRootHeightUpperLimit = atof(terminal->FirstChildElement("RootHeight")->Attribute("upper"));
	}
private:
	Configurations(){
		// Reference type is not set by configuration file.
		// It depends on it is interactive mode or not
		this->mReferenceType = ReferenceType::FIXED;
	}


	double mJointDamping;

	int mMotionHz, mControlHz, mSimulationHz;

	ReferenceType mReferenceType;
	bool mUseEarlyTermination;
	int mNumActors;

	int mFutureCount;

	double mRootHeightOffset;

	std::vector<int> mPredictionsInState;

	RewardType mRewardType;

	double mRootHeightLowerLimit, mRootHeightUpperLimit;
	double mRootDiffThreshold, mRootAngleDiffThreshold;

	int mMGMotionSize, mTCMotionSize;
};

}