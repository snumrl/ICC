#include <tinyxml.h>
#include "SkeletonBuilder.h"
#include "Utils.h"
#include "Configurations.h"

using namespace dart::dynamics;

double _default_damping_coefficient = JOINT_DAMPING;

namespace ICC
{
Eigen::Vector3d proj(const Eigen::Vector3d& u,const Eigen::Vector3d& v)
{
	Eigen::Vector3d proj;
	proj = u.dot(v)/u.dot(u)*u;
	return proj;	
}
Eigen::Isometry3d orthoNormalize(const Eigen::Isometry3d& T_old)
{
	Eigen::Isometry3d T;
	T.translation() = T_old.translation();
	Eigen::Vector3d v0,v1,v2;
	Eigen::Vector3d u0,u1,u2;
	v0 = T_old.linear().col(0);
	v1 = T_old.linear().col(1);
	v2 = T_old.linear().col(2);

	u0 = v0;
	u1 = v1 - proj(u0,v1);
	u2 = v2 - proj(u0,v2) - proj(u1,v2);

	u0.normalize();
	u1.normalize();
	u2.normalize();

	T.linear().col(0) = u0;
	T.linear().col(1) = u1;
	T.linear().col(2) = u2;
	return T;
}


SkeletonPtr 
SkeletonBuilder::
buildFromFile(const std::string& filename){
	TiXmlDocument doc;
	if(!doc.LoadFile(filename)){
		std::cout << "Can't open file : " << filename << std::endl;
		return nullptr;
	}

	TiXmlElement *skeldoc = doc.FirstChildElement("Skeleton");
	
	std::string skelname = skeldoc->Attribute("name");
	SkeletonPtr skel = Skeleton::create(skelname);
	// std::cout << skelname << std::endl;

	for(TiXmlElement *body = skeldoc->FirstChildElement("Joint"); body != nullptr; body = body->NextSiblingElement("Joint")){
		// type
		std::string jointType = body->Attribute("type");
		// name
		std::string name = body->Attribute("name");
		// parent name
		std::string parentName = body->Attribute("parent_name");
		BodyNode *parent;
		if(!parentName.compare("None"))
			parent = nullptr;
		else
			parent = skel->getBodyNode(parentName);
		// size
		Eigen::Vector3d size = ICC::Utils::stringToVector3d(std::string(body->Attribute("size")));
		// body position
		TiXmlElement *bodyPosElem = body->FirstChildElement("BodyPosition");
		Eigen::Isometry3d bodyPosition;
		bodyPosition.setIdentity();
		bodyPosition.linear() = ICC::Utils::stringToMatrix3d(bodyPosElem->Attribute("linear"));
		bodyPosition.translation() = ICC::Utils::stringToVector3d(bodyPosElem->Attribute("translation"));
		bodyPosition = orthoNormalize(bodyPosition);
		// joint position
		TiXmlElement *jointPosElem = body->FirstChildElement("JointPosition");
		Eigen::Isometry3d jointPosition;
		jointPosition.setIdentity();
		if(jointPosElem->Attribute("linear")!=nullptr)
			jointPosition.linear() = ICC::Utils::stringToMatrix3d(jointPosElem->Attribute("linear"));
		jointPosition.translation() = ICC::Utils::stringToVector3d(jointPosElem->Attribute("translation"));
		jointPosition = orthoNormalize(jointPosition);

		// shape : capsule, sphere, none, cylinder, box
		double shape_radius = 0;
		double shape_height = 0;
		int shape_type = 0;
		Eigen::Vector3d shape_direction, shape_offset, shape_size;
		shape_direction.setZero();
		shape_offset.setZero();
		shape_size.setZero();

		// capsule
		TiXmlElement *shapeElem = body->FirstChildElement("Capsule");
		if(shapeElem != nullptr){
			shape_direction = ICC::Utils::stringToVector3d(shapeElem->Attribute("direction"));
			shape_radius = atof(shapeElem->Attribute("radius"));
			shape_height = atof(shapeElem->Attribute("height"));
			if(shapeElem->Attribute("offset")!=nullptr)
				shape_offset = ICC::Utils::stringToVector3d(shapeElem->Attribute("offset"));;
			shape_type = 1;
		}

		// sphere
		shapeElem = body->FirstChildElement("Sphere");
		if(shapeElem != nullptr){
			shape_radius = atof(shapeElem->Attribute("radius"));
			if(shapeElem->Attribute("offset")!=nullptr)
				shape_offset = ICC::Utils::stringToVector3d(shapeElem->Attribute("offset"));;
			shape_type = 2;
		}

		// cylinder
		shapeElem = body->FirstChildElement("Cylinder");
		if(shapeElem != nullptr){
			shape_direction = ICC::Utils::stringToVector3d(shapeElem->Attribute("direction"));
			shape_radius = atof(shapeElem->Attribute("radius"));
			shape_height = atof(shapeElem->Attribute("height"));
			if(shapeElem->Attribute("offset")!=nullptr)
				shape_offset = ICC::Utils::stringToVector3d(shapeElem->Attribute("offset"));;
			shape_type = 3;
		}

		// box
		shapeElem = body->FirstChildElement("Box");
		if(shapeElem != nullptr){
			shape_size = ICC::Utils::stringToVector3d(shapeElem->Attribute("size"));
			if(shapeElem->Attribute("offset")!=nullptr)
				shape_offset = ICC::Utils::stringToVector3d(shapeElem->Attribute("offset"));;
			shape_type = 4;
		}

		// mass
		double mass = atof(body->Attribute("mass"));

		bool contact = true;
		
		// if(body->Attribute("contact")!=nullptr)
		// {
		// 	if(std::string(body->Attribute("contact"))=="On")
		// 		contact = true;
		// }
		if(!jointType.compare("FreeJoint") ){
			SkeletonBuilder::makeFreeJointBody(
				name,
				skel,
				parent,
				size,
				jointPosition,
				bodyPosition,
				mass,
				contact,
				shape_type,
				shape_radius, 
				shape_height, 
				shape_direction,
				shape_offset,
				shape_size
				);
		}

//        if(!jointType.compare("FreeJointBall") ){
//            SkeletonBuilder::MakeFreeJointBall(
//                    name,
//                    skel,
//                    parent,
//                    size,
//                    jointPosition,
//                    bodyPosition,
//                    mass,
//                    contact
//            );
//        }

		else if(!jointType.compare("BallJoint")){
			// joint limit
			bool isLimitEnforced = false;
			Eigen::Vector3d upperLimit(1E6,1E6,1E6), lowerLimit(-1E6,-1E6,-1E6);
			if(jointPosElem->Attribute("upper")!=nullptr)
			{
				isLimitEnforced = true;
				upperLimit = ICC::Utils::stringToVector3d(jointPosElem->Attribute("upper"));
				lowerLimit = ICC::Utils::stringToVector3d(jointPosElem->Attribute("lower"));
			}
			
			SkeletonBuilder::makeBallJointBody(
				name,
				skel,
				parent,
				size,
				jointPosition,
				bodyPosition,
				isLimitEnforced,
				upperLimit,
				lowerLimit,
				mass,
				contact,
				shape_type,
				shape_radius, 
				shape_height, 
				shape_direction,
				shape_offset,
				shape_size
				);
		}
		else if(!jointType.compare("RevoluteJoint")){
			// joint limit
			bool isLimitEnforced = false;
			double upperLimit(1E6), lowerLimit(-1E6);
			if(jointPosElem->Attribute("upper")!=nullptr)
			{
				isLimitEnforced = true;
				upperLimit = atof(jointPosElem->Attribute("upper"));
				lowerLimit = atof(jointPosElem->Attribute("lower"));
			}

			// axis
			Eigen::Vector3d axis = ICC::Utils::stringToVector3d(body->Attribute("axis"));

			SkeletonBuilder::makeRevoluteJointBody(
				name,
				skel,
				parent,
				size,
				jointPosition,
				bodyPosition,
				isLimitEnforced,
				upperLimit,
				lowerLimit,
				mass,
				axis,
				contact
				);			
		}
		else if(!jointType.compare("PrismaticJoint")){
			// joint limit
			TiXmlElement *jointLimitElem = body->FirstChildElement("Limit");
			bool isLimitEnforced = false;
			double upperLimit, lowerLimit;
			if( jointLimitElem != nullptr ){
				isLimitEnforced = true;
				upperLimit = atof(jointLimitElem->Attribute("upper"));
				lowerLimit = atof(jointLimitElem->Attribute("lower"));
			}
			// axis
			Eigen::Vector3d axis = ICC::Utils::stringToVector3d(body->Attribute("axis"));

			SkeletonBuilder::makePrismaticJointBody(
				name,
				skel,
				parent,
				size,
				jointPosition,
				bodyPosition,
				isLimitEnforced,
				upperLimit,
				lowerLimit,
				mass,
				axis,
				contact
				);	
		}
		else if(!jointType.compare("WeldJoint")){
			SkeletonBuilder::makeWeldJointBody(
				name,
				skel,
				parent,
				size,
				jointPosition,
				bodyPosition,
				mass,
				contact
				);			
		}

	}
	return skel;
}


/*
void 
SkeletonBuilder::
WriteSkeleton(std::string filename, dart::dynamics::SkeletonPtr& skel){

	std::cout << std::endl << std::endl << "Write skeleton" << std::endl;

	// get body node && name
	BodyNode* bn = skel->getBodyNode(0);
	std::string name = bn->getName();

	// get parent body node && name
	BodyNode* parent = bn->getParentBodyNode();
	std::string parent_name;
	if(parent == nullptr)
		parent_name = "None";
	else
		parent_name = parent->getName();

	// get mass
	double mass = bn->getMass();

	// get size
	ShapeNode* sn = bn->getShapeNodesWith<DynamicsAspect>()[0];
	ShapeNode::BasicProperties props = sn->getShapeNodeProperties();
	ShapePtr sp = props.mShape;
	std::string shape_type = sp->getType();
	Eigen::Vector3d size;
	if(!shape_type.compare("BoxShape")){
		std::shared_ptr<BoxShape> bs = std::dynamic_pointer_cast<BoxShape>(sp);
		size = bs->getSize();
	}
	else{
		std::cout << "undefined shpae type" << std::endl;
	}

	// get type of joint
	std::string type = bn->getParentJoint()->getType();
	if(!type.compare("PrismaticJoint")){
		PrismaticJoint *pj = dynamic_cast<PrismaticJoint*>(bn->getParentJoint());
		PrismaticJoint::Properties props = pj->getPrismaticJointProperties();
	
		Eigen::Vector3d axis = props.mAxis;

		std::cout << mass << std::endl;
		std::cout << name << std::endl;
		std::cout << parent_name << std::endl;
		std::cout << type << std::endl;
		std::cout << axis.transpose() << std::endl;
		std::cout << size.transpose() << std::endl;
	}


}
*/

BodyNode* 
SkeletonBuilder::
makeFreeJointBall(
        const std::string& body_name,
        const dart::dynamics::SkeletonPtr& target_skel,
        dart::dynamics::BodyNode* const parent,
        const Eigen::Vector3d& size,
        const Eigen::Isometry3d& joint_position,
        const Eigen::Isometry3d& body_position,
        double mass,
        bool contact)
{
    double radius;
    ShapePtr shape = std::shared_ptr<SphereShape>(new SphereShape(radius/*size*/));

    dart::dynamics::Inertia inertia;
    inertia.setMass(mass);
    inertia.setMoment(shape->computeInertia(mass));

    BodyNode* bn;
    FreeJoint::Properties props;
    props.mName = body_name;
    // props.mT_ChildBodyToJoint = joint_position;
    props.mT_ParentBodyToJoint = body_position;

    bn = target_skel->createJointAndBodyNodePair<FreeJoint>(
            parent,props,BodyNode::AspectProperties(body_name)).second;

    if(contact)
        bn->createShapeNodeWith<VisualAspect,CollisionAspect,DynamicsAspect>(shape);
    else
        bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);
    bn->setInertia(inertia);
    return bn;
}



BodyNode*
SkeletonBuilder::
makeFreeJointBody(
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
)
{
	ShapePtr shape = std::shared_ptr<BoxShape>(new BoxShape(size));

	double r = shape_radius;
	double h = shape_height;
	Eigen::Vector3d direction = shape_direction;
	ShapePtr shapeVisual;
	if(shape_type == 1)
		shapeVisual = std::shared_ptr<CapsuleShape>(new CapsuleShape(r, h));
	if(shape_type == 2)
		shapeVisual = std::shared_ptr<SphereShape>(new SphereShape(r));
	if(shape_type == 3)
		shapeVisual = std::shared_ptr<CylinderShape>(new CylinderShape(r, h));
	if(shape_type == 4)
		shapeVisual = std::shared_ptr<BoxShape>(new BoxShape(shape_size));

	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));

	BodyNode* bn;
	FreeJoint::Properties props;
	props.mName = body_name;
	// props.mT_ChildBodyToJoint = joint_position;
	props.mT_ParentBodyToJoint = body_position;

	bn = target_skel->createJointAndBodyNodePair<FreeJoint>(
		parent,props,BodyNode::AspectProperties(body_name)).second;

	if(contact){
		// bn->createShapeNodeWith<VisualAspect,CollisionAspect,DynamicsAspect>(shape);
		bn->createShapeNodeWith<CollisionAspect,DynamicsAspect>(shape);
		if(shape_type == 0){
			bn->createShapeNodeWith<VisualAspect>(shape);
		}
		else if(shape_type == 1 || shape_type == 3){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.linear() = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), direction).toRotationMatrix();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 2){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 4){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		bn->createShapeNodeWith<VisualAspect>(shape);
	}
	else{
		bn->createShapeNodeWith<DynamicsAspect>(shape);
		if(shape_type == 0){
			bn->createShapeNodeWith<VisualAspect>(shape);
		}
		else if(shape_type == 1 || shape_type == 3){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.linear() = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), direction).toRotationMatrix();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 2){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 4){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		bn->createShapeNodeWith<VisualAspect>(shape);
	}
	bn->setInertia(inertia);
	return bn;
}

BodyNode* 
SkeletonBuilder::
makeBallJointBody(
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
)
{
	ShapePtr shape = std::shared_ptr<BoxShape>(new BoxShape(size));

	double r = shape_radius;
	double h = shape_height;
	Eigen::Vector3d direction = shape_direction;
	ShapePtr shapeVisual;
	if(shape_type == 1)
		shapeVisual = std::shared_ptr<CapsuleShape>(new CapsuleShape(r, h));
	if(shape_type == 2)
		shapeVisual = std::shared_ptr<SphereShape>(new SphereShape(r));
	if(shape_type == 3)
		shapeVisual = std::shared_ptr<CylinderShape>(new CylinderShape(r, h));
	if(shape_type == 4)
		shapeVisual = std::shared_ptr<BoxShape>(new BoxShape(shape_size));

	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));

	BodyNode* bn;
	BallJoint::Properties props;
	props.mName = body_name;
	if(parent!=nullptr)
		props.mT_ParentBodyToJoint = parent->getTransform().inverse()*joint_position;
	props.mT_ChildBodyToJoint = body_position.inverse()*joint_position;

	// std::cout<<props.mT_ChildBodyToJoint.translation().transpose()<<std::endl;
	// std::cout<<props.mT_ChildBodyToJoint.linear()<<std::endl;
	// std::cout<<props.mT_ChildBodyToJoint.linear().determinant()<<std::endl;
	// std::cout<<dart::math::verifyTransform(props.mT_ChildBodyToJoint)<<std::endl;

	bn = target_skel->createJointAndBodyNodePair<BallJoint>(
		parent,props,BodyNode::AspectProperties(body_name)).second;

	JointPtr jn = bn->getParentJoint();
	for(int i = 0; i < jn->getNumDofs(); i++){
		jn->getDof(i)->setDampingCoefficient(_default_damping_coefficient);
	}

	if(contact){
		// bn->createShapeNodeWith<VisualAspect,CollisionAspect,DynamicsAspect>(shape);
		bn->createShapeNodeWith<CollisionAspect,DynamicsAspect>(shape);
		if(shape_type == 0){
			bn->createShapeNodeWith<VisualAspect>(shape);
		}
		else if(shape_type == 1 || shape_type == 3){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.linear() = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), direction).toRotationMatrix();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 2){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 4){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		bn->createShapeNodeWith<VisualAspect>(shape);
	}
	else{
		bn->createShapeNodeWith<DynamicsAspect>(shape);
		if(shape_type == 0){
			bn->createShapeNodeWith<VisualAspect>(shape);
		}
		else if(shape_type == 1 || shape_type == 3){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.linear() = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), direction).toRotationMatrix();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 2){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 4){
			bn->createShapeNodeWith<VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		bn->createShapeNodeWith<VisualAspect>(shape);
	}
	bn->setInertia(inertia);

	if(isLimitEnforced){
		JointPtr joint = bn->getParentJoint();
		joint->setPositionLimitEnforced(isLimitEnforced);
		for(int i = 0; i < 3; i++)
		{
			joint->setPositionUpperLimit(i, upper_limit[i]);
			joint->setPositionLowerLimit(i, lower_limit[i]);
		}
	}
	return bn;
}

BodyNode* 
SkeletonBuilder::
makeRevoluteJointBody(
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
	bool contact)
{
	ShapePtr shape = std::shared_ptr<BoxShape>(new BoxShape(size));

	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));

	BodyNode* bn;
	RevoluteJoint::Properties props;
	props.mName = body_name;
	props.mAxis = axis;

	if(parent!=nullptr)
		props.mT_ParentBodyToJoint = parent->getTransform().inverse()*joint_position;
	props.mT_ChildBodyToJoint = body_position.inverse()*joint_position;

	bn = target_skel->createJointAndBodyNodePair<RevoluteJoint>(
		parent,props,BodyNode::AspectProperties(body_name)).second;

	JointPtr jn = bn->getParentJoint();
	for(int i = 0; i < jn->getNumDofs(); i++){
		jn->getDof(i)->setDampingCoefficient(_default_damping_coefficient);
	}

	if(contact)
		bn->createShapeNodeWith<VisualAspect,CollisionAspect,DynamicsAspect>(shape);
	else
		bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);
	bn->setInertia(inertia);

	if(isLimitEnforced){
		JointPtr joint = bn->getParentJoint();
		joint->setPositionLimitEnforced(isLimitEnforced);
		joint->setPositionUpperLimit(0, upper_limit);
		joint->setPositionLowerLimit(0, lower_limit);
	}

	return bn;
}

BodyNode* 
SkeletonBuilder::
makePrismaticJointBody(
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
	bool contact)
{
	ShapePtr shape = std::shared_ptr<BoxShape>(new BoxShape(size));

	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));

	BodyNode* bn;
	PrismaticJoint::Properties props;
	props.mName = body_name;
	props.mAxis = axis;

	if(parent!=nullptr)
		props.mT_ParentBodyToJoint = parent->getTransform().inverse()*joint_position;
	props.mT_ChildBodyToJoint = body_position.inverse()*joint_position;

	bn = target_skel->createJointAndBodyNodePair<PrismaticJoint>(
		parent,props,BodyNode::AspectProperties(body_name)).second;
	
	if(contact)
		bn->createShapeNodeWith<VisualAspect,CollisionAspect,DynamicsAspect>(shape);
	else
		bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);
	bn->setInertia(inertia);

	if(isLimitEnforced){
		JointPtr joint = bn->getParentJoint();
		joint->setPositionLimitEnforced(isLimitEnforced);
		joint->setPositionUpperLimit(0, upper_limit);
		joint->setPositionLowerLimit(0, lower_limit);
	}

	return bn;
}

BodyNode* 
SkeletonBuilder::
makeWeldJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& joint_position,
	const Eigen::Isometry3d& body_position,
	double mass,
	bool contact)
{
	ShapePtr shape = std::shared_ptr<BoxShape>(new BoxShape(size));

	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));

	BodyNode* bn;
	WeldJoint::Properties props;
	props.mName = body_name;
	
	if(parent!=nullptr)
		props.mT_ParentBodyToJoint = parent->getTransform().inverse()*joint_position;
	props.mT_ChildBodyToJoint = body_position.inverse()*joint_position;

	bn = target_skel->createJointAndBodyNodePair<WeldJoint>(
		parent,props,BodyNode::AspectProperties(body_name)).second;
	
	if(contact)
		bn->createShapeNodeWith<VisualAspect,CollisionAspect,DynamicsAspect>(shape);
	else
		bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);
	bn->setInertia(inertia);

	return bn;
}

}