#include "EnvironmentWrapper.h"

#include <omp.h>
#include <iostream>

#include "dart/dart.hpp"
#include "Utils.h"
#include "Configurations.h"

EnvironmentWrapper::
EnvironmentWrapper(std::string configuration_filepath, int num_slaves)
{
	ICC::Configurations::instance().LoadConfigurations(configuration_filepath);

	this->mNumSlaves = num_slaves;
	dart::math::seedRand();
	omp_set_num_threads(this->mNumSlaves);
	for(int id = 0; id < this->mNumSlaves; id++){
		this->mSlaves.emplace_back(new ICC::Environment());
	}

	this->mStateSize = this->mSlaves[0]->getStateSize();
	this->mActionSize = this->mSlaves[0]->getActionSize();
}

//For general properties
int
EnvironmentWrapper::
getStateSize()
{
	return this->mStateSize;
}

int
EnvironmentWrapper::
getActionSize()
{
	return this->mActionSize;
}

//For each slave
void 
EnvironmentWrapper::
step(int id, bool record)
{
	this->mSlaves[id]->step(record);
}

void 
EnvironmentWrapper::
reset(int id, double time)
{
	this->mSlaves[id]->reset(time);
}

bool 
EnvironmentWrapper::
isTerminal(int id)
{
	return this->mSlaves[id]->isTerminal();
}

p::tuple 
EnvironmentWrapper::
isNanAtTerminal(int id)
{
	bool t = mSlaves[id]->isTerminal();
	bool n = mSlaves[id]->isNanAtTerminal();
	bool e = mSlaves[id]->isEndOfTrajectory();
	return p::make_tuple(t, n, e);
}

np::ndarray
EnvironmentWrapper::
getState(int id)
{
	return ICC::Utils::toNumPyArray(this->mSlaves[id]->getState());
}

void 
EnvironmentWrapper::
setAction(int id, np::ndarray np_array)
{
	this->mSlaves[id]->setAction(ICC::Utils::toEigenVector(np_array, this->mActionSize));
}

np::ndarray
EnvironmentWrapper::
getReward(int id)
{
	return ICC::Utils::toNumPyArray(this->mSlaves[id]->getReward());
}

//For all slaves
void
EnvironmentWrapper::
steps(bool record)
{
	if( this->mNumSlaves == 1){
		this->step(0, record);
	}
	else{
#pragma omp parallel for
		for (int id = 0; id < this->mNumSlaves; id++){
			this->step(id, record);
		}
	}
}

void
EnvironmentWrapper::
resets()
{
	for (int id = 0; id < this->mNumSlaves; id++){
		this->reset(id, 0);
	}
}

np::ndarray
EnvironmentWrapper::
isTerminals()
{
	std::vector<bool> is_terminate_vector(this->mNumSlaves);

	for(int id = 0; id < this->mNumSlaves; id++){
		is_terminate_vector[id] = this->isTerminal(id);
	}

	return ICC::Utils::toNumPyArray(is_terminate_vector);
}

np::ndarray
EnvironmentWrapper::
getStates()
{
	std::vector<Eigen::VectorXd> states(this->mNumSlaves);

	for (int id = 0; id < this->mNumSlaves; id++){
		states[id] = this->mSlaves[id]->getState();
		
	}
	return ICC::Utils::toNumPyArray(states);
}

void
EnvironmentWrapper::
setActions(np::ndarray np_array)
{
	Eigen::MatrixXd action = ICC::Utils::toEigenMatrix(np_array, this->mNumSlaves, this->mActionSize);

	for (int id = 0; id < this->mNumSlaves; id++){
		this->mSlaves[id]->setAction(action.row(id));
	}
}

np::ndarray
EnvironmentWrapper::
getRewards()
{
	std::vector<std::vector<double>> rewards(this->mNumSlaves);
	for (int id = 0; id < this->mNumSlaves; id++){
		rewards[id] = this->mSlaves[id]->getReward();
	}

	return ICC::Utils::toNumPyArray(rewards);
}

void
EnvironmentWrapper::
setReferenceTrajectory(int id, int frame, np::ndarray ref_trajectory)
{
	Eigen::MatrixXd mat = ICC::Utils::toEigenMatrix(ref_trajectory, frame, ICC::Configurations::instance().getTCMotionSize());
	std::vector<Eigen::VectorXd> converted_traj;
	converted_traj.resize(mat.rows());
	for(int i = 0; i < mat.rows(); i++){
		converted_traj[i] = ICC::Utils::convertMGToTC(mat.row(i), this->mSlaves[id]->getActor()->getSkeleton());
	}
	this->mSlaves[id]->setReferenceTrajectory(converted_traj);
}

void
EnvironmentWrapper::
setReferenceTrajectories(int frame, np::ndarray ref_trajectory)
{
	Eigen::MatrixXd mat = ICC::Utils::toEigenMatrix(ref_trajectory, frame, ICC::Configurations::instance().getTCMotionSize());
	std::vector<Eigen::VectorXd> converted_traj;
	converted_traj.resize(mat.rows());
	for(int i = 0; i < mat.rows(); i++){
		converted_traj[i] = ICC::Utils::convertMGToTC(mat.row(i), this->mSlaves[0]->getActor()->getSkeleton());
	}
#pragma omp parallel for
	for(int id = 0; id < this->mNumSlaves; id++){
		this->mSlaves[id]->setReferenceTrajectory(converted_traj);
	}
}

void
EnvironmentWrapper::
setReferenceTargetTrajectory(int id, int frame, np::ndarray target_trajectory)
{
	Eigen::MatrixXd mat = ICC::Utils::toEigenMatrix(target_trajectory, frame, 2);
	std::vector<Eigen::Vector3d> converted_traj;
	converted_traj.resize(mat.rows());
	for(int i = 0; i < mat.rows(); i++){
		converted_traj[i] << mat(i,0), 0, mat(i,1);
	}
	this->mSlaves[id]->setReferenceTargetTrajectory(converted_traj);
}

void
EnvironmentWrapper::
setReferenceTargetTrajectories(int frame, np::ndarray ref_trajectory)
{
	Eigen::MatrixXd mat = ICC::Utils::toEigenMatrix(ref_trajectory, frame, 2);
	std::vector<Eigen::Vector3d> converted_traj;
	converted_traj.resize(mat.rows());
	for(int i = 0; i < mat.rows(); i++){
		converted_traj[i] << mat(i,0), 0, mat(i,1);
	}
#pragma omp parallel for
	for(int id = 0; id < this->mNumSlaves; id++){
		this->mSlaves[id]->setReferenceTargetTrajectory(converted_traj);
	}
}


void
EnvironmentWrapper::
followReference(int id)
{
	this->mSlaves[id]->followReference();
}

void
EnvironmentWrapper::
followReferences()
{
	for (int id = 0; id < this->mNumSlaves; id++)
		this->mSlaves[id]->followReference();
}

void
EnvironmentWrapper::
writeRecords(std::string path)
{
	for (int id = 0; id < this->mNumSlaves; id++){
		mSlaves[id]->writeRecords(path + std::to_string(id) + ".rec");	
		// mSlaves[id]->writeFullRecords(path + std::to_string(id) + "_full.rec");	
	}
}

using namespace boost::python;

BOOST_PYTHON_MODULE(environment_wrapper)
{
	Py_Initialize();
	np::initialize();

	class_<EnvironmentWrapper>("environment",init<std::string, int>())
		.def("getStateSize",&EnvironmentWrapper::getStateSize)
		.def("getActionSize",&EnvironmentWrapper::getActionSize)
		.def("step",&EnvironmentWrapper::step)
		.def("reset",&EnvironmentWrapper::reset)
		.def("isTerminal",&EnvironmentWrapper::isTerminal)
		.def("isNanAtTerminal",&EnvironmentWrapper::isNanAtTerminal)
		.def("getState",&EnvironmentWrapper::getState)
		.def("setAction",&EnvironmentWrapper::setAction)
		.def("getReward",&EnvironmentWrapper::getReward)
		.def("steps",&EnvironmentWrapper::steps)
		.def("resets",&EnvironmentWrapper::resets)
		.def("isTerminals",&EnvironmentWrapper::isTerminals)
		.def("getStates",&EnvironmentWrapper::getStates)
		.def("setActions",&EnvironmentWrapper::setActions)
		.def("getRewards",&EnvironmentWrapper::getRewards)
		.def("setReferenceTrajectory",&EnvironmentWrapper::setReferenceTrajectory)
		.def("setReferenceTrajectories",&EnvironmentWrapper::setReferenceTrajectories)
		.def("setReferenceTargetTrajectory",&EnvironmentWrapper::setReferenceTargetTrajectory)
		.def("setReferenceTargetTrajectories",&EnvironmentWrapper::setReferenceTargetTrajectories)
		.def("followReference",&EnvironmentWrapper::followReference)
		.def("followReferences",&EnvironmentWrapper::followReferences)
		.def("writeRecords",&EnvironmentWrapper::writeRecords);
}