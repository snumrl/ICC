#include <iostream>

#include "Character.h"
#include "SkeletonBuilder.h"
#include "Utils.h"


namespace ICC
{

Character::
Character(const std::string& path)
{
	if(path == ""){
		std::cout << "Character::Character : character file path is required!!!" << std::endl;
		exit(0);
	}
	this->mSkeleton = ICC::SkeletonBuilder::buildFromFile(path);
	this->mSkeletonDof = this->mSkeleton->getNumDofs();
	this->mCharacterFilePath = path;
}

void
Character::
setPDParameters(double kp, double kv)
{
	this->setPDParameters(
		Eigen::VectorXd::Constant(this->mSkeletonDof, kp), 
		Eigen::VectorXd::Constant(this->mSkeletonDof, kv)
	);
}

void 
Character::
setPDParameters(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv)
{
	this->mKp = kp;
	this->mKv = kv;
	this->mKp.block<6,1>(0,0).setZero();
	this->mKv.block<6,1>(0,0).setZero();
}

Eigen::VectorXd 
Character::
getSPDForces(const Eigen::VectorXd& p_desired, const Eigen::VectorXd& v_desired)
{
	auto& skel = this->mSkeleton;
	Eigen::VectorXd q = skel->getPositions();
	Eigen::VectorXd dq = skel->getVelocities();
	double dt = skel->getTimeStep();

	// This line requires half the time of the simulation, it need to be improved.
	Eigen::MatrixXd M_inv = (skel->getMassMatrix() + Eigen::MatrixXd(dt*mKv.asDiagonal())).inverse();

	// Eigen::VectorXd p_d = q + dq*dt - p_desired;
	Eigen::VectorXd p_d(q.rows());
	// clamping radians to [-pi, pi], only for ball joints
	// TODO : make it for all type joints
	p_d.head<6>().setZero();
	for(int i = 6; i < skel->getNumDofs(); i+=3){
		Eigen::Quaterniond q_s = Utils::dartToQuat(q.segment<3>(i));
		Eigen::Quaterniond dq_s = Utils::dartToQuat(dt*(dq.segment<3>(i)));
		Eigen::Quaterniond q_d_s = Utils::dartToQuat(p_desired.segment<3>(i));

		Eigen::Quaterniond p_d_s = q_d_s.inverse()*q_s*dq_s;

		Eigen::Vector3d v = Utils::quatToDart(p_d_s);
		double angle = v.norm();
		if(angle > 1e-8){
			Eigen::Vector3d axis = v.normalized();

			angle = Utils::radianClamp(angle);	
			p_d.segment<3>(i) = angle * axis;
		}
		else
			p_d.segment<3>(i) = v;
	}

	Eigen::VectorXd p_diff = -mKp.cwiseProduct(p_d);
	Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq-v_desired);
	Eigen::VectorXd qddot = M_inv*(-skel->getCoriolisAndGravityForces()+
							p_diff+v_diff+skel->getConstraintForces());

	Eigen::VectorXd tau = p_diff + v_diff - dt*mKv.cwiseProduct(qddot);
	tau.head<6>().setZero();

	return tau;
}

void
Character::
applyForces(const Eigen::VectorXd& forces)
{
	this->mSkeleton->setForces(forces);
}


}