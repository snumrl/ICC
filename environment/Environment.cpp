#include <iostream>

#include "Environment.h"



namespace ICC
{

Environment::
Environment()
{
	std::cout << "Initializing Environment" << std::endl;
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