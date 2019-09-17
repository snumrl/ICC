#include "DART_interface.h"
using namespace dart::dynamics;
using namespace dart::simulation;

void
GUI::
DrawSkeleton(
	const dart::dynamics::SkeletonPtr& skel, int type)
{
	for(int i=0;i<skel->getNumBodyNodes();i++)
	{
		auto bn = skel->getBodyNode(i);
		auto shapeNodes = bn->getShapeNodesWith<VisualAspect>();
		auto jn = bn->getParentJoint();
		Eigen::Isometry3d jn_transform = bn->getTransform()*jn->getTransformFromChildBodyNode();
		Eigen::Vector3d jn_com = jn_transform.translation();

		std::string name = bn->getName();
		Eigen::Vector4d color = shapeNodes[type]->getVisualAspect()->getRGBA();
		color.head<3>() *= 0.5;
		if(name == "FemurL" || name == "FemurR" || name == "TibiaL" || name == "TibiaR"){
			glPushMatrix();
			glColor4f(color[0], color[1], color[2], color[3]);
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			glTranslatef(jn_com[0], jn_com[1], jn_com[2]);
			GUI::DrawSphere(0.04);
			glPopMatrix();
		}

		if(name == "ForeArmL" || name == "ForeArmR"){
			glPushMatrix();
			glColor4f(color[0], color[1], color[2], color[3]);
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			glTranslatef(jn_com[0], jn_com[1], jn_com[2]);
			GUI::DrawSphere(0.03);
			glPopMatrix();
		}

		if(name == "HandL" || name == "HandR"){
			glPushMatrix();
			glColor4f(color[0], color[1], color[2], color[3]);
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			glTranslatef(jn_com[0], jn_com[1], jn_com[2]);
			GUI::DrawSphere(0.025);
			glPopMatrix();
		}

		if(name == "FootL" || name == "FootR"){
			glPushMatrix();
			glColor4f(color[0], color[1], color[2], color[3]);
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			glTranslatef(jn_com[0], jn_com[1], jn_com[2]);
			GUI::DrawSphere(0.035);
			glPopMatrix();
		}
		


		auto T = shapeNodes[type]->getTransform();
		DrawShape(T,shapeNodes[type]->getShape().get(),shapeNodes[type]->getVisualAspect()->getRGBA(), name);
	}
}

void
GUI::
DrawSkeleton(
	const dart::dynamics::SkeletonPtr& skel, const Eigen::Vector3d& uniform_color, int type)
{
	for(int i=0;i<skel->getNumBodyNodes();i++)
	{
		auto bn = skel->getBodyNode(i);
		auto shapeNodes = bn->getShapeNodesWith<VisualAspect>();
		auto jn = bn->getParentJoint();
		Eigen::Isometry3d jn_transform = bn->getTransform()*jn->getTransformFromChildBodyNode();
		Eigen::Vector3d jn_com = jn_transform.translation();

		std::string name = bn->getName();
		Eigen::Vector4d color = shapeNodes[type]->getVisualAspect()->getRGBA();
		color.head<3>() *= 0.5;
		if(name == "FemurL" || name == "FemurR" || name == "TibiaL" || name == "TibiaR"){
			glPushMatrix();
			glColor4f(color[0], color[1], color[2], color[3]);
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			glTranslatef(jn_com[0], jn_com[1], jn_com[2]);
			GUI::DrawSphere(0.04);
			glPopMatrix();
		}

		if(name == "ForeArmL" || name == "ForeArmR"){
			glPushMatrix();
			glColor4f(color[0], color[1], color[2], color[3]);
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			glTranslatef(jn_com[0], jn_com[1], jn_com[2]);
			GUI::DrawSphere(0.03);
			glPopMatrix();
		}

		if(name == "HandL" || name == "HandR"){
			glPushMatrix();
			glColor4f(color[0], color[1], color[2], color[3]);
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			glTranslatef(jn_com[0], jn_com[1], jn_com[2]);
			GUI::DrawSphere(0.025);
			glPopMatrix();
		}

		if(name == "FootL" || name == "FootR"){
			glPushMatrix();
			glColor4f(color[0], color[1], color[2], color[3]);
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			glTranslatef(jn_com[0], jn_com[1], jn_com[2]);
			GUI::DrawSphere(0.035);
			glPopMatrix();
		}
		


		auto T = shapeNodes[type]->getTransform();
		DrawShape(T,shapeNodes[type]->getShape().get(),shapeNodes[type]->getVisualAspect()->getRGBA(),uniform_color, name);
	}
}

void
GUI::
DrawSkeleton(
	const dart::dynamics::SkeletonPtr& skel,
	const Eigen::Vector3d& color)
{
	for(int i=0;i<skel->getNumBodyNodes();i++)
	{
		auto bn = skel->getBodyNode(i);
		auto shapeNodes = bn->getShapeNodesWith<VisualAspect>();

		auto T = shapeNodes[0]->getTransform();
		DrawShape(T,shapeNodes[0]->getShape().get(),color);
	}
}


void
GUI::
DrawShape(const Eigen::Isometry3d& T,
	const dart::dynamics::Shape* shape,
	const Eigen::Vector3d& color)
{
	glEnable(GL_LIGHTING);
	// glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glColor3f(color[0],color[1],color[2]);
	glPushMatrix();
	// glMultMatrixd(T.data());
	Eigen::Vector3d translation = T.translation();
	Eigen::Matrix3d linear = T.linear();
	Eigen::AngleAxisd aa(linear);
	glTranslatef(translation[0], translation[1], translation[2]);

	if(shape->is<SphereShape>())
	{
		const auto* sphere = dynamic_cast<const SphereShape*>(shape);
		// std::cout<<"draw sphere: "<<translation.transpose()<<std::endl;
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
		GUI::DrawSphere(sphere->getRadius());
		// glColor3f(0,0,0);
		// glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
		// GUI::DrawSphere(sphere->getRadius());
	}
	glRotatef(aa.angle()/M_PI*180.0, aa.axis()[0], aa.axis()[1], aa.axis()[2]);

	if (shape->is<BoxShape>())
	{
		const auto* box = dynamic_cast<const BoxShape*>(shape);
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    	GUI::DrawRoundedBox(box->getSize(), 0.01);
    	// GUI::DrawCube(Eigen::Vector3d(0.01,0.01,0.01));
	}
	else if(shape->is<MeshShape>())
	{
		auto* mesh = dynamic_cast<const MeshShape*>(shape);

		// for(int i =0;i<16;i++)
			// std::cout<<(*mesh->getMesh()->mRootNode->mTransformation)[i]<<" ";
    	GUI::DrawMesh(mesh->getScale(),mesh->getMesh());

	}

	glPopMatrix();

	// glDisable(GL_COLOR_MATERIAL);
}

void
GUI::
DrawShape(const Eigen::Isometry3d& T,
	const dart::dynamics::Shape* shape,
	const Eigen::Vector4d& color,
	std::string name)
{
	glEnable(GL_LIGHTING);
	// glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
    float ground_mat_shininess[] = {128.0};
    float ground_mat_specular[]  = {0.01, 0.01, 0.01, 0.35};
    float ground_mat_diffuse[]   = {0.05, 0.05, 0.05, 0.35};
    float ground_mat_ambient[]  = {0.05, 0.05, 0.05, 0.35};
    for(int i =0 ; i < 4; i++){
    	ground_mat_specular[i] = color[i]*0.1;
    	ground_mat_diffuse[i] = color[i];
    	ground_mat_ambient[i] = color[i];
    }

    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, ground_mat_shininess);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  ground_mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   ground_mat_diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   ground_mat_ambient);

	// glColor4f(color[0],color[1],color[2],color[3]);
	glPushMatrix();
	// glMultMatrixd(T.data());
	Eigen::Vector3d translation = T.translation();
	Eigen::Matrix3d linear = T.linear();
	Eigen::AngleAxisd aa(linear);
	glTranslatef(translation[0], translation[1], translation[2]);

	if(shape->is<SphereShape>())
	{
		const auto* sphere = dynamic_cast<const SphereShape*>(shape);
		// std::cout<<"draw sphere: "<<translation.transpose()<<std::endl;
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
		GUI::DrawSphere(sphere->getRadius());
		// glColor3f(0,0,0);
		// glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
		// GUI::DrawSphere(sphere->getRadius());
	}
	glRotatef(aa.angle()/M_PI*180.0, aa.axis()[0], aa.axis()[1], aa.axis()[2]);

	if (shape->is<BoxShape>())
	{
		const auto* box = dynamic_cast<const BoxShape*>(shape);
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
		if( name == "FootL" || name == "FootEndL" || name == "FootR" || name == "FootEndR"  || name == "HandL" || name == "HandR" )
    		GUI::DrawRoundedBox(box->getSize(), 0.01);
    	else
    		GUI::DrawRoundedBox(box->getSize(), 0.02);
    	// GUI::DrawCube(Eigen::Vector3d(0.01,0.01,0.01));
	}
	else if(shape->is<MeshShape>())
	{
		auto* mesh = dynamic_cast<const MeshShape*>(shape);

		// for(int i =0;i<16;i++)
			// std::cout<<(*mesh->getMesh()->mRootNode->mTransformation)[i]<<" ";
    	GUI::DrawMesh(mesh->getScale(),mesh->getMesh());

	}
	else if(shape->is<CapsuleShape>())
	{
		auto* capsule = dynamic_cast<const CapsuleShape*>(shape);

		// for(int i =0;i<16;i++)
			// std::cout<<(*mesh->getMesh()->mRootNode->mTransformation)[i]<<" ";
    	GUI::DrawCapsule(capsule->getRadius(),capsule->getHeight());

	}
	else if(shape->is<CylinderShape>())
	{
		auto* cylinder = dynamic_cast<const CylinderShape*>(shape);

		// for(int i =0;i<16;i++)
			// std::cout<<(*mesh->getMesh()->mRootNode->mTransformation)[i]<<" ";
    	GUI::DrawCylinder(cylinder->getRadius(),cylinder->getHeight());

	}

	glPopMatrix();

	// glDisable(GL_COLOR_MATERIAL);
}
void
GUI::
DrawShape(const Eigen::Isometry3d& T,
	const dart::dynamics::Shape* shape,
	const Eigen::Vector4d& color,
	const Eigen::Vector3d& uniform_color,
	std::string name)
{
	glEnable(GL_LIGHTING);
	// glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
    float ground_mat_shininess[] = {128.0};
    float ground_mat_specular[]  = {0.01, 0.01, 0.01, 0.35};
    float ground_mat_diffuse[]   = {0.05, 0.05, 0.05, 0.35};
    float ground_mat_ambient[]  = {0.05, 0.05, 0.05, 0.35};
    for(int i =0 ; i < 4; i++){
    	ground_mat_specular[i] = color[i]*0.1;
    	ground_mat_diffuse[i] = color[i];
    	ground_mat_ambient[i] = color[i];
    }

    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, ground_mat_shininess);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  ground_mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   ground_mat_diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   ground_mat_ambient);

	// glColor4f(color[0],color[1],color[2],color[3]);
	glPushMatrix();
	// glMultMatrixd(T.data());
	Eigen::Vector3d translation = T.translation();
	Eigen::Matrix3d linear = T.linear();
	Eigen::AngleAxisd aa(linear);
	glTranslatef(translation[0], translation[1], translation[2]);

	if(shape->is<SphereShape>())
	{
		const auto* sphere = dynamic_cast<const SphereShape*>(shape);
		// std::cout<<"draw sphere: "<<translation.transpose()<<std::endl;
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
		GUI::DrawSphere(sphere->getRadius());
		// glColor3f(0,0,0);
		// glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
		// GUI::DrawSphere(sphere->getRadius());
	}
	glRotatef(aa.angle()/M_PI*180.0, aa.axis()[0], aa.axis()[1], aa.axis()[2]);

	if (shape->is<BoxShape>())
	{
		const auto* box = dynamic_cast<const BoxShape*>(shape);
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
		if( name == "FootL" || name == "FootEndL" || name == "FootR" || name == "FootEndR"  || name == "HandL" || name == "HandR" )
    		GUI::DrawRoundedBox(box->getSize(), 0.01, uniform_color);
    	else
    		GUI::DrawRoundedBox(box->getSize(), 0.02, uniform_color);
    	// GUI::DrawCube(Eigen::Vector3d(0.01,0.01,0.01));
	}
	else if(shape->is<MeshShape>())
	{
		auto* mesh = dynamic_cast<const MeshShape*>(shape);

		// for(int i =0;i<16;i++)
			// std::cout<<(*mesh->getMesh()->mRootNode->mTransformation)[i]<<" ";
    	GUI::DrawMesh(mesh->getScale(),mesh->getMesh());

	}
	else if(shape->is<CapsuleShape>())
	{
		auto* capsule = dynamic_cast<const CapsuleShape*>(shape);

		// for(int i =0;i<16;i++)
			// std::cout<<(*mesh->getMesh()->mRootNode->mTransformation)[i]<<" ";
    	GUI::DrawCapsule(capsule->getRadius(),capsule->getHeight());

	}
	else if(shape->is<CylinderShape>())
	{
		auto* cylinder = dynamic_cast<const CylinderShape*>(shape);

		// for(int i =0;i<16;i++)
			// std::cout<<(*mesh->getMesh()->mRootNode->mTransformation)[i]<<" ";
    	GUI::DrawCylinder(cylinder->getRadius(),cylinder->getHeight());

	}

	glPopMatrix();

	// glDisable(GL_COLOR_MATERIAL);
}