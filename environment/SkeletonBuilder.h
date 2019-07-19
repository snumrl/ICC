#pragma once
#include "dart/dart.hpp"

namespace ICC
{
class SkeletonBuilder
{
public:
	static dart::dynamics::SkeletonPtr buildFromFile(const std::string& filename);
	//static void WriteSkeleton(std::string filename, dart::dynamics::SkeletonPtr& skel);

	//SM) added for drawing ball
    static dart::dynamics::BodyNode* makeFreeJointBall(
        const std::string& body_name,
        const dart::dynamics::SkeletonPtr& target_skel,
        dart::dynamics::BodyNode* const parent,
        const Eigen::Vector3d& size,
        const Eigen::Isometry3d& joint_position,
        const Eigen::Isometry3d& body_position,
        double mass,
        bool contact);


    static dart::dynamics::BodyNode* makeFreeJointBody(
		const std::string& body_name,
		const dart::dynamics::SkeletonPtr& target_skel,
		dart::dynamics::BodyNode* const parent,
		const Eigen::Vector3d& size,
		const Eigen::Isometry3d& joint_position,
		const Eigen::Isometry3d& body_position,
		double mass,
		bool contact,
		int shape_type,
		double shape_radius, 
		double shape_height, 
		Eigen::Vector3d shape_direction,
		Eigen::Vector3d shape_offset,
		Eigen::Vector3d shape_size
	);

	static dart::dynamics::BodyNode* makeBallJointBody(
		const std::string& body_name,
		const dart::dynamics::SkeletonPtr& target_skel,
		dart::dynamics::BodyNode* const parent,
		const Eigen::Vector3d& size,
		const Eigen::Isometry3d& joint_position,
		const Eigen::Isometry3d& body_position,
		bool isLimitEnforced,
		const Eigen::Vector3d& upper_limit,
		const Eigen::Vector3d& lower_limit,
		double mass,
		bool contact,
		int shape_type,
		double shape_radius, 
		double shape_height, 
		Eigen::Vector3d shape_direction,
		Eigen::Vector3d shape_offset,
		Eigen::Vector3d shape_size
	);
	
	static dart::dynamics::BodyNode* makeRevoluteJointBody(
		const std::string& body_name,
		const dart::dynamics::SkeletonPtr& target_skel,
		dart::dynamics::BodyNode* const parent,
		const Eigen::Vector3d& size,
		const Eigen::Isometry3d& joint_position,
		const Eigen::Isometry3d& body_position,
		bool isLimitEnforced,
		double upper_limit,
		double lower_limit,
		double mass,
		const Eigen::Vector3d& axis,
		bool contact);

	static dart::dynamics::BodyNode* makePrismaticJointBody(
		const std::string& body_name,
		const dart::dynamics::SkeletonPtr& target_skel,
		dart::dynamics::BodyNode* const parent,
		const Eigen::Vector3d& size,
		const Eigen::Isometry3d& joint_position,
		const Eigen::Isometry3d& body_position,
		bool isLimitEnforced,
		double upper_limit,
		double lower_limit,
		double mass,
		const Eigen::Vector3d& axis,
		bool contact);

	static dart::dynamics::BodyNode* makeWeldJointBody(
		const std::string& body_name,
		const dart::dynamics::SkeletonPtr& target_skel,
		dart::dynamics::BodyNode* const parent,
		const Eigen::Vector3d& size,
		const Eigen::Isometry3d& joint_position,
		const Eigen::Isometry3d& body_position,
		double mass,
		bool contact);
};
}
