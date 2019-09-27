#pragma once

#include <Eigen/Core>

#include "Character.h"

namespace ICC
{ 
/**
*
* @brief Reference manager
* @details Managing reference motions
*
*/
class ReferenceManager
{
public:

	/// Constructor
	ReferenceManager(Character* character);

	/// Get positions and velocities at time
	Eigen::VectorXd getPositionsAndVelocities(int time=0);

	/// Get current target
	Eigen::Vector3d getTarget(int time=0){
		if(this->mTotalFrame == 0) return Eigen::Vector3d::Zero();
		return this->mReferenceTargetTrajectory[this->mCurFrame + time];
	}

	/// Set reference motion from motion generator
	void setReferenceTrajectory(const std::vector<Eigen::VectorXd>& trajectory);

	/// Set reference motion from motion generator
	void setReferenceTargetTrajectory(const std::vector<Eigen::Vector3d>& target_traj);


	/// clear reference manger
	void clear(){
		this->mCurFrame = 0;
		this->mTotalFrame = 0;
		this->mReferenceTrajectory.clear();
		this->mReferenceTargetTrajectory.clear();
	}
	/// add reference
	void addReference(Eigen::VectorXd ref);

	/// add target
	void addTarget(Eigen::Vector3d target);

	/// Set current frame
	void setCurrentFrame(int frame){
		this->mCurFrame = frame;
	}

	/// Get total frame
	int getTotalFrame(){
		return this->mTotalFrame;
	}

	/// Increase current frame
	void increaseCurrentFrame(){
		this->mCurFrame++;
	}

	bool isEndOfTrajectory(){
		return (this->mCurFrame >= (this->mTotalFrame - 1));
	}



protected:
	int mCurFrame;
	int mTotalFrame;

	Character* mCharacter;

	std::vector<Eigen::VectorXd> mReferenceTrajectory;
	std::vector<Eigen::Vector3d> mReferenceTargetTrajectory;
};


}