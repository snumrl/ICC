#include <iostream>

#include "Environment.h"

namespace ICC
{

Environment::
Environment(int num_actors)
{
	std::cout << "Initializing Environment" << std::endl;

	// Create actors
	this->mNumActors = num_actors;
	for(int i = 0; i < this->mNumActors; i++){
		this->mActors.emplace_back(new Character(std::string(DPHY_DIR)+std::string("/characters/humanoid.xml")));
	}

	// Create gound
	this->mGround = new Character(std::string(DPHY_DIR)+std::string("/character/ground.xml"));

	// Create motion references
	//  - fixed trajectory
	//  - interactive control

	// Set initial configurations
	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
}

void
Environment::
step()
{
	std::cout << "Environment::step()" << std::endl;
	return;
}

Eigen::VectorXd 
Environment::
getState()
{
	std::cout << "Environment::getState()" << std::endl;
	return Eigen::Vector3d::Zero();
}

double
Environment::
getReward()
{
	std::cout << "Environment::getReward()" << std::endl;
	return 0;
}

void 
Environment::
setAction(Eigen::VectorXd action)
{
	std::cout << "Environment::setAction()" << std::endl;
	return;
}


}