#include "Utils.h"
#include "Configurations.h"
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm> 
#include <cctype>
#include <locale>


namespace ICC
{

namespace Utils
{


//always return 1-dim array
np::ndarray toNumPyArray(const std::vector<float>& val)
{
	int n = val.size();
	p::tuple shape = p::make_tuple(n);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray array = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(array.get_data());
	for(int i=0;i<n;i++)
	{
		dest[i] = val[i];
	}

	return array;
}

//always return 1-dim array
np::ndarray toNumPyArray(const std::vector<double>& val)
{
	int n = val.size();
	p::tuple shape = p::make_tuple(n);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray array = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(array.get_data());
	for(int i=0;i<n;i++)
	{
		dest[i] = (float)val[i];
	}

	return array;
}

//always return 1-dim array
np::ndarray toNumPyArray(const std::vector<bool>& val)
{
	int n = val.size();
	p::tuple shape = p::make_tuple(n);
	np::dtype dtype = np::dtype::get_builtin<bool>();
	np::ndarray array = np::empty(shape,dtype);

	bool* dest = reinterpret_cast<bool*>(array.get_data());
	for(int i=0;i<n;i++)
	{
		dest[i] = val[i];
	}

	return array;
}

//always return 1-dim array
np::ndarray toNumPyArray(const Eigen::VectorXd& vec)
{
	int n = vec.rows();
	p::tuple shape = p::make_tuple(n);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray array = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(array.get_data());
	for(int i =0;i<n;i++)
	{
		dest[i] = vec[i];
	}

	return array;
}
//always return 2-dim array
np::ndarray toNumPyArray(const Eigen::MatrixXd& matrix)
{
	int n = matrix.rows();
	int m = matrix.cols();

	p::tuple shape = p::make_tuple(n,m);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray array = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(array.get_data());
	int index = 0;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			dest[index++] = matrix(i,j);
		}
	}

	return array;
}
//always return 2-dim array
np::ndarray toNumPyArray(const std::vector<Eigen::VectorXd>& matrix)
{
	int n = matrix.size();
	int m = matrix[0].rows();

	p::tuple shape = p::make_tuple(n,m);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray array = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(array.get_data());
	int index = 0;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			dest[index++] = matrix[i][j];
		}
	}

	return array;
}

//always return 2-dim array
np::ndarray toNumPyArray(const std::vector<std::vector<double>>& matrix)
{
	int n = matrix.size();
	int m = matrix[0].size();

	p::tuple shape = p::make_tuple(n,m);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray array = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(array.get_data());
	int index = 0;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			dest[index++] = matrix[i][j];
		}
	}

	return array;
}

Eigen::VectorXd toEigenVector(const np::ndarray& array,int n)
{
	Eigen::VectorXd vec(n);

	float* srcs = reinterpret_cast<float*>(array.get_data());

	for(int i=0;i<n;i++)
	{
		vec[i] = srcs[i];
	}
	return vec;
}

Eigen::VectorXd toEigenVector(const p::object& array,int n)
{
	Eigen::VectorXd vec(n);

	float* srcs = reinterpret_cast<float*>(array.ptr());

	for(int i=0;i<n;i++)
	{
		vec[i] = srcs[i];
	}
	return vec;
}
Eigen::MatrixXd toEigenMatrix(const np::ndarray& array,int n,int m)
{
	Eigen::MatrixXd mat(n,m);

	float* srcs = reinterpret_cast<float*>(array.get_data());

	int index = 0;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			mat(i,j) = srcs[index++];
		}
	}
	return mat;
}

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
static inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
static inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
static inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
       	*(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim=' ') {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

std::string join(const std::vector<std::string> &v, char delim=' '){
	std::stringstream ss;
	for(size_t i = 0; i < v.size(); ++i)
	{
		if(i != 0)
			ss << delim;
		ss << v[i];
	}

	return ss.str();
}

std::vector<int> splitToInt(const std::string& input, int num)
{
    std::vector<int> result;
    std::string::size_type sz = 0, nsz = 0;
    for(int i = 0; i < num; i++){
        result.push_back(std::stoi(input.substr(sz), &nsz));
        sz += nsz;
    }
    return result;
}

std::vector<int> splitToInt(const std::string& input)
{
    std::vector<int> result;
    std::string::size_type sz = 0, nsz = 0;
    while(sz< input.length()){
        result.push_back(std::stoi(input.substr(sz), &nsz));
        sz += nsz;
    }
    return result;
}

std::vector<double> splitToDouble(const std::string& input, int num)
{
    std::vector<double> result;
    std::string::size_type sz = 0, nsz = 0;
    for(int i = 0; i < num; i++){
        result.push_back(std::stold(input.substr(sz), &nsz));
        sz += nsz;
    }
    return result;
}

std::vector<double> splitToDouble(const std::string& input)
{
    std::vector<double> result;
    std::string::size_type sz = 0, nsz = 0;
    while(sz< input.length()){
        result.push_back(std::stold(input.substr(sz), &nsz));
        sz += nsz;
    }
    return result;
}

Eigen::Vector3d stringToVector3d(const std::string& input){
	std::vector<double> v = splitToDouble(input, 3);
	Eigen::Vector3d res;
	res << v[0], v[1], v[2];

	return res;
}

Eigen::VectorXd stringToVectorXd(const std::string& input, int n){
	std::vector<double> v = splitToDouble(input, n);
	Eigen::VectorXd res(n);
	for(int i = 0; i < n; i++){
		res[i] = v[i];
	}
	return res;
}

Eigen::VectorXd stringToVectorXd(const std::string& input){
    std::vector<double> v = splitToDouble(input);
    Eigen::VectorXd res(v.size());
    for(int i = 0; i < v.size(); i++){
        res[i] = v[i];
    }
    return res;
}

Eigen::Matrix3d stringToMatrix3d(const std::string& input){
	std::vector<double> v = splitToDouble(input, 9);
	Eigen::Matrix3d res;
	res << v[0], v[1], v[2],
			v[3], v[4], v[5],
			v[6], v[7], v[8];

	return res;
}

double radianClamp(double input){
	return std::fmod(input+M_PI, 2*M_PI)-M_PI;
}

double expOfSquared(const Eigen::VectorXd& vec,double sigma)
{
	return exp(-1.0*vec.dot(vec)/(sigma*sigma)/vec.rows());
}
double expOfSquared(const Eigen::Vector3d& vec,double sigma)
{
	return exp(-1.0*vec.dot(vec)/(sigma*sigma)/vec.rows());
}
double expOfSquared(const Eigen::MatrixXd& mat,double sigma)
{
	return exp(-1.0*mat.squaredNorm()/(sigma*sigma)/mat.size());
}


std::pair<int, double> maxCoeff(const Eigen::VectorXd& in){
	double m = 0;
	int idx = 0;
	for(int i = 0; i < in.rows(); i++){
		if( m < in[i]){
			m = in[i];
			idx = i;
		}
	}
	return std::make_pair(idx, m);
}

void setBodyNodeColors(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& color)
{
	auto visualShapeNodes = bn->getShapeNodesWith<dart::dynamics::VisualAspect>();
	for(auto visualShapeNode : visualShapeNodes)
		visualShapeNode->getVisualAspect()->setColor(color);
}

void setSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector3d& color)
{
	// Set the color of all the shapes in the object
	for(std::size_t i=0; i < object->getNumBodyNodes(); ++i)
	{
		Eigen::Vector3d c = color;
		dart::dynamics::BodyNode* bn = object->getBodyNode(i);
		if(bn->getName() == "Neck")
			c.head<3>() *= 0.5;
		auto visualShapeNodes = bn->getShapeNodesWith<dart::dynamics::VisualAspect>();
		for(auto visualShapeNode : visualShapeNodes)
			visualShapeNode->getVisualAspect()->setColor(c);
	}
}

void setSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector4d& color)
{
	// Set the color of all the shapes in the object
	for(std::size_t i=0; i < object->getNumBodyNodes(); ++i)
	{
		Eigen::Vector4d c = color;
		dart::dynamics::BodyNode* bn = object->getBodyNode(i);
		if(bn->getName() == "Neck")
			c.head<3>() *= 0.5;
		auto visualShapeNodes = bn->getShapeNodesWith<dart::dynamics::VisualAspect>();
		for(auto visualShapeNode : visualShapeNodes)
			visualShapeNode->getVisualAspect()->setRGBA(c);
	}
}


Eigen::Quaterniond dartToQuat(Eigen::Vector3d in){
	if( in.norm() < 1e-8 ){
		return Eigen::Quaterniond::Identity();
	}
	Eigen::AngleAxisd aa(in.norm(), in.normalized());
	Eigen::Quaterniond q(aa);
	quaternionNormalize(q);
	return q;
}

Eigen::Vector3d quatToDart(const Eigen::Quaterniond& in){
	Eigen::AngleAxisd aa(in);
	double angle = aa.angle();
	angle = std::fmod(angle+M_PI, 2*M_PI)-M_PI;
	return angle*aa.axis();
}

void quaternionNormalize(Eigen::Quaterniond& in){
	if(in.w() < 0){
		in.coeffs() *= -1;
	}
}

Eigen::Quaterniond getYRotation(Eigen::Quaterniond q){
	// from body joint vector
	Eigen::Vector3d rotated = q._transformVector(Eigen::Vector3d::UnitZ());
	double angle = atan2(rotated[0], rotated[2]);
	Eigen::Quaterniond ret(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()));

	return ret;
}


Eigen::Vector3d changeToRNNPos(Eigen::Vector3d pos){
	Eigen::Vector3d ret;
	ret[0] = pos[2]*100;
	ret[1] = (pos[1] - Configurations::instance().getRootHeightOffset())*100;
	ret[2] = -pos[0]*100;
	return ret;
}

Eigen::Isometry3d getJointTransform(dart::dynamics::SkeletonPtr skel, std::string bodyname){
	return skel->getBodyNode(bodyname)->getParentBodyNode()->getWorldTransform()
		*skel->getBodyNode(bodyname)->getParentJoint()->getTransformFromParentBodyNode();
}


Eigen::VectorXd solveIK(dart::dynamics::SkeletonPtr skel, const std::string& bodyname, const Eigen::Vector3d& delta, const Eigen::Vector3d& offset)
{
	auto bn = skel->getBodyNode(bodyname);
	int foot_l_idx = skel->getBodyNode("FootL")->getParentJoint()->getIndexInSkeleton(0);
	int foot_r_idx = skel->getBodyNode("FootR")->getParentJoint()->getIndexInSkeleton(0);
	int footend_l_idx = skel->getBodyNode("FootEndL")->getParentJoint()->getIndexInSkeleton(0);
	int footend_r_idx = skel->getBodyNode("FootEndR")->getParentJoint()->getIndexInSkeleton(0);
	int femur_l_idx = skel->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	int femur_r_idx = skel->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_l_idx = skel->getBodyNode("TibiaL")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_r_idx = skel->getBodyNode("TibiaR")->getParentJoint()->getIndexInSkeleton(0);
	Eigen::VectorXd newPose = skel->getPositions();
	Eigen::Vector3d tp = delta;
	for(std::size_t i = 0; i < 1000; ++i)
	{
		Eigen::Vector3d deviation = tp - bn->getTransform()*offset;
		if(deviation.norm() < 0.001)
			break;
		// Eigen::Vector3d localCOM = bn->getCOM(bn);
		dart::math::LinearJacobian jacobian = skel->getLinearJacobian(bn, offset);
		jacobian.block<3,6>(0,0).setZero();
		// jacobian.block<3,3>(0,foot_l_idx).setZero();
		// jacobian.block<3,3>(0,foot_r_idx).setZero();
		jacobian.block<3,3>(0,footend_l_idx).setZero();
		jacobian.block<3,3>(0,footend_r_idx).setZero();
		// jacobian.block<3,2>(0,femur_l_idx+1).setZero();
		// jacobian.block<3,2>(0,femur_r_idx+1).setZero();
		// jacobian.block<3,2>(0,tibia_l_idx+1).setZero();
		// jacobian.block<3,2>(0,tibia_r_idx+1).setZero();

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3d inv_singular_value;
		
		inv_singular_value.setZero();
		for(int k=0;k<3;k++)
		{
			if(svd.singularValues()[k]==0)
				inv_singular_value(k,k) = 0.0;
			else
				inv_singular_value(k,k) = 1.0/svd.singularValues()[k];
		}


		Eigen::MatrixXd jacobian_inv = svd.matrixV()*inv_singular_value*svd.matrixU().transpose();

		// Eigen::VectorXd gradient = jacobian.colPivHouseholderQr().solve(deviation);
		Eigen::VectorXd gradient = jacobian_inv * deviation;
		double prev_norm = deviation.norm();
		double gamma = 0.5;
		for(int j = 0; j < 24; j++){
			Eigen::VectorXd newDirection = gamma * gradient;
			Eigen::VectorXd np = newPose + newDirection;
			skel->setPositions(np);
			skel->computeForwardKinematics(true, false, false);
			double new_norm = (tp - bn->getTransform()*offset).norm();
			if(new_norm < prev_norm){
				newPose = np;
				break;
			}
			gamma *= 0.5;
		}
	}
	return newPose;
}

Eigen::VectorXd solveMCIK(dart::dynamics::SkeletonPtr skel, const std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>& constraints)
{
	int foot_l_idx = skel->getBodyNode("FootL")->getParentJoint()->getIndexInSkeleton(0);
	int foot_r_idx = skel->getBodyNode("FootR")->getParentJoint()->getIndexInSkeleton(0);
	int footend_l_idx = skel->getBodyNode("FootEndL")->getParentJoint()->getIndexInSkeleton(0);
	int footend_r_idx = skel->getBodyNode("FootEndR")->getParentJoint()->getIndexInSkeleton(0);
	int femur_l_idx = skel->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	int femur_r_idx = skel->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_l_idx = skel->getBodyNode("TibiaL")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_r_idx = skel->getBodyNode("TibiaR")->getParentJoint()->getIndexInSkeleton(0);

	Eigen::VectorXd newPose = skel->getPositions();
	int num_constraints = constraints.size();

	std::vector<dart::dynamics::BodyNode*> bodynodes(num_constraints);
	std::vector<Eigen::Vector3d> targetposes(num_constraints);
	std::vector<Eigen::Vector3d> offsets(num_constraints);

	for(int i = 0; i < num_constraints; i++){
		bodynodes[i] = skel->getBodyNode(std::get<0>(constraints[i]));
		targetposes[i] = std::get<1>(constraints[i]);
		offsets[i] = std::get<2>(constraints[i]);
	}

	int not_improved = 0;
	for(std::size_t i = 0; i < 100; i++)
	{

		// make deviation vector and jacobian matrix
		Eigen::VectorXd deviation(num_constraints*3);
		for(int j = 0; j < num_constraints; j++){
			deviation.segment<3>(j*3) = targetposes[j] - bodynodes[j]->getTransform()*offsets[j];
		}
		if(deviation.norm() < 0.001)
			break;

		int nDofs = skel->getNumDofs();
		Eigen::MatrixXd jacobian_concatenated(3*num_constraints, nDofs);
		for(int j = 0; j < num_constraints; j++){
			dart::math::LinearJacobian jacobian = skel->getLinearJacobian(bodynodes[j], offsets[j]);
			jacobian.block<3,6>(0,0).setZero();
			// jacobian.block<3,3>(0,foot_l_idx).setZero();
			// jacobian.block<3,3>(0,foot_r_idx).setZero();
			jacobian.block<3,3>(0,footend_l_idx).setZero();
			jacobian.block<3,3>(0,footend_r_idx).setZero();
			// jacobian.block<3,2>(0,femur_l_idx+1).setZero();
			// jacobian.block<3,2>(0,femur_r_idx+1).setZero();
			jacobian.block<3,2>(0,tibia_l_idx+1).setZero();
			jacobian.block<3,2>(0,tibia_r_idx+1).setZero();

			jacobian_concatenated.block(3*j, 0, 3, nDofs) = jacobian;
		}
		// std::cout << jacobian_concatenated << std::endl;

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian_concatenated, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXd inv_singular_value(3*num_constraints, 3*num_constraints);
		
		inv_singular_value.setZero();
		for(int k=0;k<3*num_constraints;k++)
		{
			if(svd.singularValues()[k]<1e-8)
				inv_singular_value(k,k) = 0.0;
			else
				inv_singular_value(k,k) = 1.0/svd.singularValues()[k];
		}


		Eigen::MatrixXd jacobian_inv = svd.matrixV()*inv_singular_value*svd.matrixU().transpose();
		// std::cout << svd.singularValues().transpose() << std::endl;
		// std::cout << svd.matrixV().size() << std::endl;

		// std::cout << jacobian_inv << std::endl;
		// exit(0);
		// Eigen::VectorXd gradient = jacobian.colPivHouseholderQr().solve(deviation);
		Eigen::VectorXd gradient = jacobian_inv * deviation;
		double prev_norm = deviation.norm();
		double gamma = 0.5;
		not_improved++;
		for(int j = 0; j < 24; j++){
			Eigen::VectorXd newDirection = gamma * gradient;
			Eigen::VectorXd np = newPose + newDirection;
			skel->setPositions(np);
			skel->computeForwardKinematics(true, false, false);

			Eigen::VectorXd new_deviation(num_constraints*3);
			for(int j = 0; j < num_constraints; j++){
				new_deviation.segment<3>(j*3) = targetposes[j] - bodynodes[j]->getTransform()*offsets[j];
			}
			double new_norm = new_deviation.norm();
			if(new_norm < prev_norm){
				newPose = np;
				not_improved = 0;
				break;
			}
			gamma *= 0.5;
		}
		if(not_improved > 1){
			break;
		}
	}
	return newPose;
}


Eigen::Vector4d rootDecomposition(dart::dynamics::SkeletonPtr skel, Eigen::VectorXd positions){
	// DEBUG : decomposition
	Eigen::VectorXd p_save = skel->getPositions();
	skel->setPositions(positions);
	int femur_l_idx = skel->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	int femur_r_idx = skel->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);

	Eigen::Isometry3d femur_l_transform = getJointTransform(skel, "FemurL");
	Eigen::Isometry3d femur_r_transform = getJointTransform(skel, "FemurR");

	Eigen::Vector3d up_vec = Eigen::Vector3d::UnitY();
	Eigen::Vector3d x_vec = femur_l_transform.translation() - femur_r_transform.translation();
	x_vec.normalize();
	Eigen::Vector3d z_vec = x_vec.cross(up_vec);
	z_vec[1] = 0;
	z_vec.normalize();
	double angle = std::atan2(z_vec[0], z_vec[2]);

	skel->setPositions(p_save);

	Eigen::AngleAxisd aa_root(angle, Eigen::Vector3d::UnitY());
	Eigen::AngleAxisd aa_hip(positions.segment<3>(0).norm(), positions.segment<3>(0).normalized());

	Eigen::Vector3d hip_dart = quatToDart(Eigen::Quaterniond(aa_root).inverse()*Eigen::Quaterniond(aa_hip));
	
	Eigen::Vector4d ret;
	ret << angle, hip_dart;

	return ret;
}

}

}