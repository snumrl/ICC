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

	/// Get motion : current positions, next positions, current velocities
	std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> getMotion(int time);
	
	/// Get positions and velocities at time
	Eigen::VectorXd getPositionsAndVelocities(int time=0);

	/// Set reference motion from motion generator
	void setReferenceTrajectory(Eigen::MatrixXd& trajectory);

	/// Convert motion format from motion generator
	Eigen::VectorXd convertFromMG(const Eigen::VectorXd& input);

	/// Set current frame
	void setCurrentFrame(int frame){
		this->mCurFrame = frame;
	}

	/// Increase current frame
	void increaseCurrentFrame(){
		this->mCurFrame++;
	}



protected:
	int mCurFrame;
	int mTotalFrame;

	Character* mCharacter;

	std::vector<Eigen::VectorXd> mReferenceTrajectory;
};


}