#pragma once

#include "dart/dart.hpp"
#include "Character.h"

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
	Environment(int num_actors=1);

	/// Step environment
	void step();

	/// Reset environment
	void reset();

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
	int getNumStates(){ return this->mNumStates; }
	/// Get the size of the action
	int getNumActions(){ return this->mNumActions; }

	/// Get dart world pointer
	const dart::simulation::WorldPtr& getWorld(){ return this->mWorld; }

protected:
	dart::simulation::WorldPtr mWorld;
	std::vector<Character*> mActors;
	Character* mGround;

	int mNumActors, mNumObstacles;
	int mNumStates, mNumActions;

	int mControlHz, mSimulationHz;
};

}