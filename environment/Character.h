#pragma once

#include "dart/dart.hpp"

namespace ICC
{
/**
*
* @brief Character class
* @details -
* 
*/
class Character
{

public:
	/// Construction from file
	Character(const std::string& path="");

	/// Get skeleton pointer
	const dart::dynamics::SkeletonPtr& getSkeleton(){ return this->mSkeleton; }

	/// Get skeleton dof
	int getNumDofs(){ return this->mSkeletonDof; }

	/// Get character file path
	const std::string& getCharacterFilePath(){ return this->mCharacterFilePath; }

	/// Set pd parameters
	void setPDParameters(double kp, double kv);

	/// Set pd parameters
	void setPDParameters(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv);

	/// Get SPD forces for desired positions and velocities(Stable PD controller)
	Eigen::VectorXd getSPDForces(const Eigen::VectorXd& p_desired, const Eigen::VectorXd& v_desired);

	/// Apply torques
	void applyForces(const Eigen::VectorXd& forces);

protected:
	dart::dynamics::SkeletonPtr mSkeleton;
	int mSkeletonDof;
	std::string mCharacterFilePath;
	Eigen::VectorXd mKp, mKv;

};

}