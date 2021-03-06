#pragma once

#include "dart/dart.hpp"
#include "Character.h"
#include "ReferenceManager.h"
#include "ThrowingBall.h"

#include <string>

namespace ICC
{

enum class TerminationReason{
	NOT_TERMINATED=1,
	ROOT_HEIGHT=2,
	ROOT_DIFF=3,
	ROOT_ANGLE_DIFF=4,
	OUT_OF_AREA=5,
	NAN_P=6,
	NAN_V=7,
	END_OF_TRAJECTORY=8,
};

/**
*
* @brief Environment class
* @details execute forward dynamics simulations
* 
*/
class Environment
{
public:
	/// Constructor
	Environment();

	/// Step environment
	void step(bool record);

	/// just follow reference motion
	void followReference();

	/// record status
	void record();

	/// write records
	void writeRecords(std::string filename);

	/// Reset environment
	void reset(double reset_time = 0);

	/// Update reference
	void updateReference();	

	/// Get state
	Eigen::VectorXd getState();

	/// Get end effector state from joint p, v
	Eigen::VectorXd getEndEffectorStatePV(const dart::dynamics::SkeletonPtr skel, const Eigen::VectorXd& pv);

	/// Set action
	void setAction(const Eigen::VectorXd& action);

	/// Get reward
	std::vector<double> getReward();

	/// Check whether the episode is terminated or not
	bool isTerminal();
	/// Check whether the episode is temrinated by nan or not
	bool isNanAtTerminal(){
		return this->mIsNanAtTerminal;
	}
	/// Check whether the episode is temrinated by end of trajectory or not
	bool isEndOfTrajectory(){
		// if(this->mIsTerminal)
		// 	std::cout << (int)this->mTerminationReason << std::endl;
		return this->mTerminationReason == TerminationReason::END_OF_TRAJECTORY;
	}

	/// Get the size of the state
	int getStateSize(){ return this->mStateSize; }
	/// Get the size of the action
	int getActionSize(){ return this->mActionSize; }

	/// Get dart world pointer
	const dart::simulation::WorldPtr& getWorld(){ return this->mWorld; }

	/// Set reference trajectory
	void setReferenceTrajectory(const std::vector<Eigen::VectorXd>& trajectory);
 
	/// Set reference target trajectory
	void setReferenceTargetTrajectory(const std::vector<Eigen::Vector3d>& traj);

	/// clear reference manager
	void clearReferenceManager(){ this->mReferenceManager->clear(); }

	/// add reference to reference manager
	void addReference(Eigen::VectorXd ref){ this->mReferenceManager->addReference(ref); }

	/// add reference target to manager
	void addReferenceTarget(Eigen::Vector3d ref_target){ this->mReferenceManager->addTarget(ref_target); }

	/// get reference
	Eigen::VectorXd getReference(int time=0){ return this->mReferenceManager->getReference(time); }

	std::vector<Eigen::VectorXd> getPositionsForMG();

	Character* getActor(){
		return this->mActor;
	}

	ThrowingBall* getThrowingBall(){ return this->mThrowingBall; }
	void createNewBall(bool randomFlag){ this->mThrowingBall->createNewBall(randomFlag); }

protected:
	dart::simulation::WorldPtr mWorld;

	Character* mActor;
	Character* mGround;
	ReferenceManager* mReferenceManager;

	int mStateSize, mActionSize;

	int mControlHz, mSimulationHz;

	/// End-effector list
	std::vector<std::string> mEndEffectors;

	// Joints used in reward evaluation
	std::vector<std::string> mRewardJoints;

	Eigen::VectorXd mAction;

	bool mIsTerminal;
	bool mIsNanAtTerminal;
	TerminationReason mTerminationReason;

	Eigen::VectorXd mTargetPositions, mTargetVelocities;
	Eigen::VectorXd mModifiedTargetPositions, mModifiedTargetVelocities;
	Eigen::Vector3d mTarget;

	std::vector<Eigen::VectorXd> mRecords, mReferenceRecords;
	std::vector<Eigen::Vector3d> mTargetRecords;


	ThrowingBall* mThrowingBall;


};

}