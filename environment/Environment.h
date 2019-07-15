#pragma once

#include "dart/dart.hpp"

namespace ICC
{

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
	void step();

	/// Reset environment
	void Reset();

	/// Get state
	Eigen::VectorXd getState();

	/// Set action
	void setAction(Eigen::VectorXd action);

	/// Get reward
	double getReward();

	/// Check whether the episode is terminated or not
	int isTerminal();
	/// Check whether the episode is temrinated by nan or not
	bool isNan();

	/// Get the size of the state
	int getStateSize();
	/// Get the size of the action
	int getActionSize();

	/// Get dart world pointer
	const dart::simulation::WorldPtr& getWorld(){ return this->mWorld; }

protected:
	dart::simulation::WorldPtr mWrold;
	std::vector<Character*> mActors;
	Character* mGround;
};

}