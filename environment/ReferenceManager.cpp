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
setReferenceTrajectory(Eigen::MatrixXd& trajectory)
{
	this->mTotalFrame = trajectory.rows();
	this->mReferenceTrajectory.clear();
	this->mReferenceTrajectory.resize(this->mTotalFrame);

	for(int i = 0; i < this->mTotalFrame; i++){
		this->mReferenceTrajectory[i] = this->convertFromMG(trajectory.row(i));
	}
}

void 
ReferenceManager::
setReferenceTargetTrajectory(Eigen::MatrixXd& trajectory)
{
	this->mReferenceTargetTrajectory.clear();
	this->mReferenceTargetTrajectory.resize(this->mTotalFrame);

	for(int i = 0; i < this->mTotalFrame; i++){
		Eigen::Vector3d goal;
		goal << trajectory.row(i)[1]*0.01, 0, trajectory.row(i)[0]*0.01;
		this->mReferenceTargetTrajectory[i] = goal;
	}
}


Eigen::VectorXd 
ReferenceManager::
convertFromMG(const Eigen::VectorXd& input)
{
	Eigen::VectorXd converted_motion = this->mCharacter->getSkeleton()->getPositions();
	converted_motion.setZero();

	// root position
	converted_motion[3] = -input[2]*0.01;
	converted_motion[4] = input[1]*0.01 + Configurations::instance().getRootHeightOffset();
	converted_motion[5] = input[0]*0.01;
	// root orientation
	Eigen::Quaterniond root_y_ori(Eigen::AngleAxisd(input[3], Eigen::Vector3d::UnitY()));
	Eigen::Quaterniond hip_ori = Utils::dartToQuat(input.segment<3>(4));
	root_y_ori = root_y_ori * hip_ori;
	converted_motion.segment<3>(0) = Utils::quatToDart(root_y_ori);

	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("Spine"   )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(7);
	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("Neck"    )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(10);
	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("Head"    )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(13);

	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("ArmL"    )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(16);
	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("ForeArmL")->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(19);
	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("HandL"   )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(22);

	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("ArmR"    )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(25);
	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("ForeArmR")->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(28);
	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("HandR"   )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(31);

	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("FemurL"  )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(34);
	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("TibiaL"  )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(37);
	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("FootL"   )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(40);

	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("FemurR"  )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(43);
	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("TibiaR"  )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(46);
	converted_motion.segment<3>(this->mCharacter->getSkeleton()->getBodyNode("FootR"   )->getParentJoint()->getIndexInSkeleton(0)) = input.segment<3>(49);

	return converted_motion;

}

}