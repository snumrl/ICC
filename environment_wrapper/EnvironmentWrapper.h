#pragma once
#include "Environment.h"
#include <vector>
#include <string>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
namespace DPhy
{
	class Controller;
}
namespace p = boost::python;
namespace np = boost::python::numpy;
class EnvironmentWrapper
{
public:
	EnvironmentWrapper(std::string configuration_filepath, int num_slaves);

	// For general properties
	int getStateSize();
	int getActionSize();

	// For each slave
	void step(int id, bool record);
	void followReference(int id);
	void reset(int id,double time);
	bool isTerminal(int id);
	p::tuple isNanAtTerminal(int id);

	np::ndarray getState(int id);
	void setAction(int id, np::ndarray np_array);
	np::ndarray getReward(int id);

	// For all slaves
	void steps(bool record);
	void followReferences();
	void resets();
	np::ndarray isTerminals();

	np::ndarray getStates();
	void setActions(np::ndarray np_array);
	np::ndarray getRewards();

	// Set reference
	void setReferenceTrajectory(int id, int frame, np::ndarray ref_trajectory);
    void setReferenceTrajectories(int frame, np::ndarray ref_trajectory);

	void setReferenceTargetTrajectory(int id, int frame, np::ndarray ref_trajectory);
    void setReferenceTargetTrajectories(int frame, np::ndarray ref_trajectory);

	void writeRecords(std::string path);

private:
	std::vector<ICC::Environment*> mSlaves;
	int mNumSlaves;

	int mStateSize;
	int mActionSize;
};


