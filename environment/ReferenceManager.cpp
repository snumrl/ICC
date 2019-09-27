#include <iostream>

#include "Configurations.h"
#include "ReferenceManager.h"
#include "Utils.h"


namespace ICC
{

ReferenceManager::
ReferenceManager(Character* character) 
: mCharacter(character), mTotalFrame(0)
{
	if(this->mCharacter == nullptr){
		std::cout << "Character is null" << std::endl;
	}
}

Eigen::VectorXd 
ReferenceManager::
getPositionsAndVelocities(int time)
{
	if(this->mTotalFrame == 0){
		return Eigen::VectorXd::Zero(this->mCharacter->getSkeleton()->getNumDofs()*2);
	}
	int t = this->mCurFrame + time;
	if(t >= this->mTotalFrame - 1){
		std::cout << "ReferenceManager::getPositionsAndVelocities()" << std::endl;
		std::cout << "requested : " << t << ", total : " << this->mTotalFrame << std::endl;
		exit(0);
	}

	Eigen::VectorXd cur_pos = this->mReferenceTrajectory[t];
	Eigen::VectorXd next_pos = this->mReferenceTrajectory[t+1];
	Eigen::VectorXd cur_vel = this->mCharacter->getSkeleton()->getPositionDifferences(next_pos, cur_pos) * Configurations::instance().getMotionHz();

	Eigen::VectorXd res = Eigen::VectorXd::Zero(cur_pos.rows() * 2);
	res << cur_pos, cur_vel;
	return res;
}

void 
ReferenceManager::
setReferenceTrajectory(const std::vector<Eigen::VectorXd>& trajectory)
{
	this->mTotalFrame = trajectory.size();
	this->mReferenceTrajectory.clear();
	this->mReferenceTrajectory.resize(this->mTotalFrame);

	for(int i = 0; i < this->mTotalFrame; i++){
		this->mReferenceTrajectory[i] = trajectory[i];
	}
}

void 
ReferenceManager::
setReferenceTargetTrajectory(const std::vector<Eigen::Vector3d>& target_trajectory)
{
	this->mReferenceTargetTrajectory.clear();
	this->mReferenceTargetTrajectory.resize(this->mTotalFrame);

	for(int i = 0; i < this->mTotalFrame; i++){
		this->mReferenceTargetTrajectory[i] = target_trajectory[i];
	}
}

/// add reference
void
ReferenceManager::
addReference(Eigen::VectorXd ref){
	this->mReferenceTrajectory.emplace_back(ref);
	this->mTotalFrame++;
}

/// add target
void
ReferenceManager::
addTarget(Eigen::Vector3d target){
	this->mReferenceTargetTrajectory.emplace_back(target);
}

}