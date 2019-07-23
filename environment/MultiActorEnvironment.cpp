#include <iostream>

#include "MultiActorEnvironment.h"
#include "Configurations.h"
#include "Utils.h"

namespace ICC
{

MultiActorEnvironment::
MultiActorEnvironment()
{
	std::cout << "Initializing MultiActorEnvironment" << std::endl;

	// Create world
	this->mWorld = std::make_shared<dart::simulation::World>();
	this->mWorld->setGravity(Eigen::Vector3d(0,-9.81,0));
	
	/// Set initial configurations
	this->mWorld->setTimeStep(1.0/(double)Configurations::instance().getSimulationHz());
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());

	/// Create ground
	this->mGround = new Character(std::string(DPHY_DIR)+std::string("/characters/ground.xml"));
	this->mWorld->addSkeleton(this->mGround->getSkeleton());

	/// Create actors
	if(Configurations::instance().getReferenceType() == ReferenceType::FIXED){
		this->mNumActors = 1;
		this->mNumReferences = 1;
	}
	else if(Configurations::instance().getReferenceType() == ReferenceType::INTERACTIVE){
		this->mNumActors = Configurations::instance().getNumActors();
		this->mNumReferences = this->mNumActors;
	}

	this->mActors.clear();
	this->mReferenceManagers.clear();

	for(int i = 0; i < this->mNumActors; i++){
		this->mActors.emplace_back(new Character(std::string(DPHY_DIR)+std::string("/characters/humanoid.xml")));
		this->mWorld->addSkeleton(this->mActors[i]->getSkeleton());
		this->mReferenceManagers.emplace_back(new ReferenceManager(this->mActors[i]));
	}

	this->mTargetPositions.resize(this->mNumActors);
	this->mTargetVelocities.resize(this->mNumActors);

	this->mModifiedTargetPositions.resize(this->mNumActors);
	this->mModifiedTargetVelocities.resize(this->mNumActors);

	this->mIsTerminal = false;
	this->mIsNanAtTerminal = false;
	this->mTerminationReason = TerminationReason::NOT_TERMINATED;

	/// Compute state and action sizes
	this->mStateSize = this->getState()[0].rows();
	this->mActionSize = this->mActors[0]->getNumDofs()-6;

	this->mActions.resize(this->mNumActors);

	// Define reward bodynodes
	this->mRewardJoints.clear();
	this->mRewardJoints.emplace_back("Torso");
	this->mRewardJoints.emplace_back("Spine");
	this->mRewardJoints.emplace_back("Neck");
	this->mRewardJoints.emplace_back("Head");

	this->mRewardJoints.emplace_back("FemurR");
	this->mRewardJoints.emplace_back("TibiaR");
	this->mRewardJoints.emplace_back("FootR");
	// mRewardJoints.emplace_back("FootEndR");

	this->mRewardJoints.emplace_back("FemurL");
	this->mRewardJoints.emplace_back("TibiaL");
	this->mRewardJoints.emplace_back("FootL");
	// mRewardJoints.emplace_back("FootEndL");

	this->mRewardJoints.emplace_back("ForeArmL");
	this->mRewardJoints.emplace_back("ArmL");
	this->mRewardJoints.emplace_back("HandL");

	this->mRewardJoints.emplace_back("ForeArmR");
	this->mRewardJoints.emplace_back("ArmR");
	this->mRewardJoints.emplace_back("HandR");

	// Define end-effectors
	this->mEndEffectors.clear();
	this->mEndEffectors.emplace_back("FootR");
	this->mEndEffectors.emplace_back("FootL");
	this->mEndEffectors.emplace_back("HandL");
	this->mEndEffectors.emplace_back("HandR");
	this->mEndEffectors.emplace_back("Head");



	// set pd gain
	Eigen::VectorXd p_gain, v_gain;
	// default 500
	p_gain = Eigen::VectorXd::Constant(this->mActors[0]->getNumDofs(), 500);
	// root 0
	p_gain.head<6>().setZero();
	v_gain = p_gain*0.1;

	for(int i = 0; i < this->mNumActors; i++){
		this->mActors[i]->setPDParameters(p_gain, v_gain);
	}

	this->reset();
}

void
MultiActorEnvironment::
reset(int reset_time)
{
	std::cout << "MultiActorEnvironment::reset()" << std::endl;

	for(int i = 0; i < this->mNumReferences; i++){
		// reset reference manager
		this->mReferenceManagers[i]->setCurrentFrame(reset_time);

		// set new character positions and velocities
		Eigen::VectorXd pv = this->mReferenceManagers[i]->getPositionsAndVelocities();
		int dof = this->mActors[i]->getNumDofs();
		this->mActors[i]->getSkeleton()->setPositions(pv.head(dof));
		this->mActors[i]->getSkeleton()->setVelocities(pv.tail(dof));
	}



	// time stepping
	if(Configurations::instance().getReferenceType() == ReferenceType::FIXED){
		for(int i = 0; i < this->mNumReferences; i++){
			this->mReferenceManagers[i]->increaseCurrentFrame();		
		}
	}

	// get target positions and velocities
	for(int i = 0; i < this->mNumReferences; i++){
		Eigen::VectorXd pv = this->mReferenceManagers[i]->getPositionsAndVelocities();
		int dof = this->mActors[i]->getNumDofs();
		this->mTargetPositions[i] = pv.head(dof);
		this->mTargetVelocities[i] = pv.tail(dof);
	}



	this->mIsTerminal = false;
	this->mIsNanAtTerminal = false;
	this->mTerminationReason = TerminationReason::NOT_TERMINATED;
}

void
MultiActorEnvironment::
step()
{
	std::cout << "MultiActorEnvironment::step()" << std::endl;

	// check terminal
	if(this->isTerminal()){
		return;
	}

	for(int i = 0; i < this->mNumReferences; i++){
		// apply actions
		this->mModifiedTargetPositions[i] = this->mTargetPositions[i];
		this->mModifiedTargetVelocities[i] = this->mTargetVelocities[i];

		Eigen::VectorXd action = this->mActions[i];
		double action_multiplier = 0.2;
		for(int j = 0; j < this->mActionSize; j++){
			action[j] = dart::math::clip(action[j]*action_multiplier, -0.7*M_PI, 0.7*M_PI);
		}

		this->mModifiedTargetPositions[i].tail(this->mActionSize) += action;
	}


	// apply forces multiple times per one spd calculation
	// regularization?
	int per = Configurations::instance().getSimulationHz()/Configurations::instance().getControlHz();
	for(int i=0;i<per;i+=2){
		std::vector<Eigen::VectorXd> torques(this->mNumActors);
		for(int k = 0; k < this->mNumActors; k++){
		 	torques[k] = this->mActors[k]->getSPDForces(this->mModifiedTargetPositions[k], this->mModifiedTargetVelocities[k]);
		}

		for(int j=0;j<2;j++)
		{
			// apply forces for all characters
			for(int k = 0; k < this->mNumActors; k++){
			 	this->mActors[k]->applyForces(torques[k]);
			}
			// forward dynamics simulation
			this->mWorld->step();
		}
	}

	// time stepping
	if(Configurations::instance().getReferenceType() == ReferenceType::FIXED){
		for(int i = 0; i < this->mNumReferences; i++){
			this->mReferenceManagers[i]->increaseCurrentFrame();		
		}
	}
	// get target positions and velocities
	for(int i = 0; i < this->mNumReferences; i++){
		Eigen::VectorXd pv = this->mReferenceManagers[i]->getPositionsAndVelocities();
		int dof = this->mActors[i]->getNumDofs();
		this->mTargetPositions[i] = pv.head(dof);
		this->mTargetVelocities[i] = pv.tail(dof);
	}

}


Eigen::VectorXd 
MultiActorEnvironment::
getEndEffectorStatePV(const dart::dynamics::SkeletonPtr skel, const Eigen::VectorXd& pv)
{
	Eigen::VectorXd ret;

	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();

	// int num_body_nodes = mInterestedBodies.size();
	int num_ee = this->mEndEffectors.size();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	int size = pv.size();
	Eigen::VectorXd pos = pv.head(size/2);
	Eigen::VectorXd vel = pv.tail(size/2);

	skel->setPositions(pos);
	skel->setVelocities(vel);

	ret.resize((num_ee)*9+12);
	for(int i=0;i<num_ee;i++)
	{
		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(this->mEndEffectors[i])->getWorldTransform();
		Eigen::Vector3d rot = Utils::quatToDart(Eigen::Quaterniond(transform.linear()));
	    int idx = skel->getBodyNode(this->mEndEffectors[i])->getParentJoint()->getIndexInSkeleton(0);

		ret.segment<9>(9*i) << rot, transform.translation(), vel.segment<3>(idx);
	}

	// root diff with target com
	Eigen::Isometry3d transform = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	Eigen::Vector3d rot = Utils::quatToDart(Eigen::Quaterniond(transform.linear()));
	Eigen::Vector3d root_angular_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getAngularVelocity();
	Eigen::Vector3d root_linear_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getCOMLinearVelocity();
	ret.tail<12>() << rot, transform.translation(), root_angular_vel_relative, root_linear_vel_relative;


	// restore
	skel->setPositions(p_save);
	skel->setVelocities(v_save);

	return ret;
}

std::vector<Eigen::VectorXd>
MultiActorEnvironment::
getState()
{
	std::cout << "MultiActorEnvironment::getState()" << std::endl;

	if(this->mIsTerminal)
		return std::vector<Eigen::VectorXd>(1, Eigen::VectorXd::Zero(this->mStateSize));


	std::vector<Eigen::VectorXd> states;
	states.resize(this->mNumActors);

	for(int i = 0; i < this->mNumActors; i++){
		auto& skel = this->mActors[i]->getSkeleton();
		dart::dynamics::BodyNode* root = skel->getRootBodyNode();

		Eigen::VectorXd p_save = skel->getPositions();
		Eigen::VectorXd v_save = skel->getVelocities();

		// predictions
		std::vector<Eigen::VectorXd> predictions;
		predictions.clear();
		for(auto pis : Configurations::instance().getPredictionsInState()){
			predictions.emplace_back(this->getEndEffectorStatePV(skel, this->mReferenceManagers[i]->getPositionsAndVelocities(pis)));
		}

		Eigen::VectorXd predictions_concatenated;
		predictions_concatenated.resize(predictions.size()*predictions[0].rows());
		for(int j = 0; j < predictions.size(); j++){
			predictions_concatenated.segment(j*predictions[0].rows(), predictions[0].rows()) = predictions[j];
		}

		// current character configurations
		// current joint positions and velocities
		Eigen::VectorXd p_cur, v_cur, ee_cur;
		// remove global transform of the root
		p_cur.resize(p_save.rows()-6);
		p_cur = p_save.tail(p_save.rows()-6);
		v_cur = v_save/10.0;

		// current end effector positions and velocities
		Eigen::VectorXd pv;
		pv.resize(p_save.rows()*2);
		pv << p_save, v_save;
		pv = this->getEndEffectorStatePV(skel, pv);
		// remove root transform because it is always zero
		ee_cur.resize(pv.rows()-6);
		ee_cur = pv.head(pv.rows()-6);
		ee_cur.tail<6>() = pv.tail<6>();


		// up vector angle
		Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
		double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);

/*
		// foot corner height
		const dart::dynamics::BodyNode *bnL, *bnEL, *bnR, *bnER;
		bnL = skel->getBodyNode("FootL");
		bnEL = skel->getBodyNode("FootEndL");
		bnR = skel->getBodyNode("FootR");
		bnER = skel->getBodyNode("FootEndR");

		// local positions of foot corner
		Eigen::Vector3d p0 = Eigen::Vector3d(0.04, -0.025, -0.065);
		Eigen::Vector3d p1 = Eigen::Vector3d(-0.04, -0.025, -0.065);
		Eigen::Vector3d p2 = Eigen::Vector3d(0.04, -0.025, 0.035);
		Eigen::Vector3d p3 = Eigen::Vector3d(-0.04, -0.025, 0.035);

		Eigen::Vector3d p0_l = bnL->getWorldTransform()*p0;
		Eigen::Vector3d p1_l = bnL->getWorldTransform()*p1;
		Eigen::Vector3d p2_l = bnEL->getWorldTransform()*p2;
		Eigen::Vector3d p3_l = bnEL->getWorldTransform()*p3;

		Eigen::Vector3d p0_r = bnR->getWorldTransform()*p0;
		Eigen::Vector3d p1_r = bnR->getWorldTransform()*p1;
		Eigen::Vector3d p2_r = bnER->getWorldTransform()*p2;
		Eigen::Vector3d p3_r = bnER->getWorldTransform()*p3;

		Eigen::VectorXd foot_corner_heights;
		foot_corner_heights.resize(8);
		foot_corner_heights << p0_l[1], p1_l[1], p2_l[1], p3_l[1], 
								p0_r[1], p1_r[1], p2_r[1], p3_r[1];
		foot_corner_heights *= 10.0;


		// root height
		double root_height = skel->getRootBodyNode()->getCOM()[1];
*/
		Eigen::VectorXd state;
		state.resize(p_cur.rows()+v_cur.rows()+ee_cur.rows()+predictions_concatenated.rows()+1);
		state<<p_cur, v_cur, ee_cur, predictions_concatenated, up_vec_angle;
		states[i] = state;
	}

	return states;
}

std::vector<std::vector<double>>
MultiActorEnvironment::
getReward()
{
	std::cout << "MultiActorEnvironment::getReward()" << std::endl;

	std::vector<std::vector<double>> ret;
	ret.resize(this->mNumActors);
	for(int i = 0; i < this->mNumActors; i++){

		auto& skel = this->mActors[i]->getSkeleton();

		// Position and velocities differences
		Eigen::VectorXd p_diff = skel->getPositionDifferences(this->mTargetPositions[i], skel->getPositions());
		Eigen::VectorXd v_diff = skel->getVelocityDifferences(this->mTargetVelocities[i], skel->getVelocities());

		int num_reward_bodynodes = this->mRewardJoints.size();
		Eigen::VectorXd p_diff_reward(num_reward_bodynodes*3), v_diff_reward(num_reward_bodynodes*3);

		for(int j = 0; j < num_reward_bodynodes; j++){
			int idx = this->mActors[i]->getSkeleton()->getJoint(this->mRewardJoints[j])->getIndexInSkeleton(0);
			p_diff_reward.segment<3>(3*j) = p_diff.segment<3>(idx);
			v_diff_reward.segment<3>(3*j) = v_diff.segment<3>(idx);
		}



		// End-effector position and COM differences
		dart::dynamics::BodyNode* root = skel->getRootBodyNode();
		Eigen::VectorXd p_save = skel->getPositions();
		Eigen::VectorXd v_save = skel->getVelocities();

		std::vector<Eigen::Isometry3d> ee_transforms;
		Eigen::VectorXd ee_diff(mEndEffectors.size()*3);
		Eigen::Vector3d com_diff;

		for(int i=0;i<mEndEffectors.size();i++){
			ee_transforms.push_back(skel->getBodyNode(this->mEndEffectors[i])->getWorldTransform());
		}
		com_diff = skel->getCOM();


		skel->setPositions(this->mTargetPositions[i]);
		skel->setVelocities(this->mTargetVelocities[i]);

		for(int i = 0; i < this->mEndEffectors.size(); i++){
			Eigen::Isometry3d diff = ee_transforms[i].inverse() * skel->getBodyNode(this->mEndEffectors[i])->getWorldTransform();
			ee_diff.segment<3>(3*i) = diff.translation();
		}
		com_diff -= skel->getCOM();

		skel->setPositions(p_save);
		skel->setVelocities(v_save);



		// Evaluate total reward
		double scale 	= 1.0;
		double sig_p 	= 0.1 * scale; 		// 2
		double sig_v 	= 1.0 * scale;		// 3
		double sig_com 	= 0.3 * scale;		// 4
		double sig_ee 	= 0.3 * scale;		// 8

		double w_p 		= 0.35;
		double w_v 		= 0.10;
		double w_com 	= 0.30;
		double w_ee 	= 0.25;

		double r_p 		= Utils::expOfSquared(p_diff_reward,sig_p);
		double r_v 		= Utils::expOfSquared(v_diff_reward,sig_v);
		double r_ee 	= Utils::expOfSquared(ee_diff,sig_ee);
		double r_com 	= Utils::expOfSquared(com_diff,sig_com);

		double rew;
		if( Configurations::instance().getRewardType() == RewardType::SUMMATION )
			rew = w_p*r_p + w_v*r_v + w_com*r_com + w_ee*r_ee;
		else if( Configurations::instance().getRewardType() == RewardType::MULTIPLICATION )
			rew = r_p * r_v * r_com * r_ee;
		else{
			std::cout << "MultiActorEnvironment::getReward() : Unspecified reward type!" << std::endl;
			exit(0);
		}

		std::vector<double> reward_list;
		reward_list.clear();
		reward_list.emplace_back(rew);
		reward_list.emplace_back(r_p);
		reward_list.emplace_back(r_v);
		reward_list.emplace_back(r_com);
		reward_list.emplace_back(r_ee);

		ret[i] = reward_list;
	}

	return ret;
}

void 
MultiActorEnvironment::
setAction(const Eigen::MatrixXd action)
{
	std::cout << "MultiActorEnvironment::setAction()" << std::endl;
	if(action.rows() != this->mNumActors){
		std::cout << "MultiActorEnvironment::setAction() : action size is not matched!" << std::endl;
	}

	for(int i = 0; i < this->mNumActors; i++){
		this->mActions[i] = action.row(i);
	}
}

bool
MultiActorEnvironment::
isTerminal()
{
	if(this->mIsTerminal)
		return true;

	this->mIsNanAtTerminal = false;

	auto& skel = this->mActors[0]->getSkeleton();

	Eigen::VectorXd p = skel->getPositions();
	Eigen::VectorXd v = skel->getVelocities();
	Eigen::Vector3d root_pos = skel->getPositions().segment<3>(3);
	Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();

	double root_y = skel->getBodyNode(0)->getTransform().translation()[1];
	Eigen::Vector3d root_v = skel->getBodyNode(0)->getCOMLinearVelocity();
	double root_v_norm = root_v.norm();
	Eigen::Vector3d root_pos_diff = this->mTargetPositions[0].segment<3>(3) - root_pos;

	skel->setPositions(this->mTargetPositions[0]);
	Eigen::Isometry3d root_diff = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	skel->setPositions(p);

	Eigen::AngleAxisd root_diff_aa(root_diff.linear());
	double angle = Utils::radianClamp(root_diff_aa.angle());

	// Nan check
	if(dart::math::isNan(p)){
		this->mIsNanAtTerminal = true;
		this->mIsTerminal = true;
		this->mTerminationReason = TerminationReason::NAN_P;
		return this->mIsTerminal;
	}
	if(dart::math::isNan(v)){
		this->mIsNanAtTerminal = true;
		this->mIsTerminal = true;
		this->mTerminationReason = TerminationReason::NAN_V;
		return this->mIsTerminal;
	}

	// Early termination
	if(Configurations::instance().useEarlyTermination()){
		if(root_y<Configurations::instance().getRootHeightLowerLimit() 
			|| root_y > Configurations::instance().getRootHeightUpperLimit()){
			this->mIsNanAtTerminal = false;
			this->mIsTerminal = true;
			this->mTerminationReason = TerminationReason::ROOT_HEIGHT;
		}
		if(std::abs(root_pos[0]) > 4990){
			this->mIsNanAtTerminal = false;
			this->mIsTerminal = true;
			this->mTerminationReason = TerminationReason::OUT_OF_AREA;
		}
		if(std::abs(root_pos[2]) > 4990){
			this->mIsNanAtTerminal = false;
			this->mIsTerminal = true;
			this->mTerminationReason = TerminationReason::OUT_OF_AREA;
		}
		if(root_pos_diff.norm() > Configurations::instance().getRootDiffThreshold()){
		this->mIsNanAtTerminal = false;
		this->mIsTerminal = true;
		this->mTerminationReason = TerminationReason::ROOT_DIFF;
		}
		if(std::abs(angle) > Configurations::instance().getRootAngleDiffThreshold()){
			this->mIsNanAtTerminal = false;
			this->mIsTerminal = true;
			this->mTerminationReason = TerminationReason::ROOT_ANGLE_DIFF;
		}
	}

	if(Configurations::instance().getReferenceType() == ReferenceType::FIXED){
		if(this->mReferenceManagers[0]->isEndOfTrajectory()){
			this->mIsNanAtTerminal = false;
			this->mIsTerminal = true;
			this->mTerminationReason = TerminationReason::END_OF_TRAJECTORY;
		}
	}

	return this->mIsTerminal;
}

void 
MultiActorEnvironment::
setReferenceTrajectory(int idx, Eigen::MatrixXd trajectory)
{
	this->mReferenceManagers[idx]->setReferenceTrajectory(trajectory);
}

void 
MultiActorEnvironment::
setReferenceTrajectoryAll(Eigen::MatrixXd trajectory)
{
	for(int i = 0; i < this->mNumActors; i++)
		this->mReferenceManagers[i]->setReferenceTrajectory(trajectory);
}

}