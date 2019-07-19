#pragma once

#include <Eigen/Core>


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
	ReferenceManager();

	/// Get motion : current positions, next positions, current velocities
	std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> getMotion(int time);
	/// get positions and velocities at time
	Eigen::VectorXd getPositionsAndVelocities(int time);

	/// set reference motion from motion generator
	void setReferenceMotion();

};


}