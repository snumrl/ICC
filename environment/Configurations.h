#pragma once

#include <cmath>
#include <vector>

// #define MOTION_WALK_EXTENSION
// #define MOTION_BASKETBALL
// #define MOTION_ZOMBIE
// #define MOTION_GORILLA
// #define MOTION_WALKRUN
// #define MOTION_WALK_NEW
// #define MOTION_WALKFALL
// #define MOTION_JOG_ROLL
#define MOTION_WALKRUNFALL

#define FUTURE_TIME (0.33)
#define FUTURE_COUNT (0)
#define FUTURE_DISPLAY_COUNT (1)
#define JOINT_DAMPING (0.05)

#define FORCE_MAGNITUDE (20)
#define FORCE_APPLYING_FRAME (60)
#define FORCE_APPLYING_BODYNODE ("Neck")


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

	int getNumSlaves(){
		return this->mNumSlaves;
	}

	int getTCMotionSize(){
		return this->mTCMotionSize;
	}

	int getMGMotionSize(){
		return this->mMGMotionSize;
	}
private:
	Configurations(){
		this->mJointDamping = 0.05;

		this->mMotionHz = 30;
		this->mControlHz = 30;
		this->mSimulationHz = 600;

		this->mReferenceType = ReferenceType::FIXED;
		this->mUseEarlyTermination = true;
		this->mFutureCount = 1;

		this->mRootHeightOffset = 0.0;

		// predictions included in stae
		// 0 : very next prediction
		this->mPredictionsInState.clear();
		this->mPredictionsInState.emplace_back(0);

		this->mRewardType = RewardType::MULTIPLICATION;

		this->mRootHeightLowerLimit = 0.5;
		this->mRootHeightUpperLimit = 2.0;

		this->mRootDiffThreshold = 1.0;
		this->mRootAngleDiffThreshold = 0.7*M_PI;
		
		this->mNumSlaves = 8;

		this->mTCMotionSize = 54;
		this->mMGMotionSize = 111;
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

	int mNumSlaves;

	int mMGMotionSize, mTCMotionSize;
};

}