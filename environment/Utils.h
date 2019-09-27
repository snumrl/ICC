#pragma once

#include "dart/dart.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
namespace p = boost::python;
namespace np = boost::python::numpy;

namespace ICC
{

namespace Utils
{

//always return 1-dim array
np::ndarray toNumPyArray(const std::vector<float>& val);
//always return 1-dim array
np::ndarray toNumPyArray(const std::vector<double>& val);
//always return 1-dim array
np::ndarray toNumPyArray(const std::vector<bool>& val);
//always return 1-dim array
np::ndarray toNumPyArray(const Eigen::VectorXd& vec);
//always return 2-dim array
np::ndarray toNumPyArray(const Eigen::MatrixXd& matrix);
//always return 2-dim array
np::ndarray toNumPyArray(const std::vector<Eigen::VectorXd>& matrix);
//always return 2-dim array
np::ndarray toNumPyArray(const std::vector<std::vector<double>>& matrix);
Eigen::VectorXd toEigenVector(const np::ndarray& array,int n);
Eigen::VectorXd toEigenVector(const p::object& array,int n);
Eigen::MatrixXd toEigenMatrix(const np::ndarray& array,int n,int m);
// Utilities
std::vector<int> splitToInt(const std::string& input, int num);
std::vector<int> splitToInt(const std::string& input);
std::vector<double> splitToDouble(const std::string& input, int num);
std::vector<double> splitToDouble(const std::string& input);
Eigen::Vector3d stringToVector3d(const std::string& input);
Eigen::VectorXd stringToVectorXd(const std::string& input, int n);
Eigen::VectorXd stringToVectorXd(const std::string& input);
Eigen::Matrix3d stringToMatrix3d(const std::string& input);

double expOfSquared(const Eigen::VectorXd& vec,double sigma = 1.0);
double expOfSquared(const Eigen::Vector3d& vec,double sigma = 1.0);
double expOfSquared(const Eigen::MatrixXd& mat,double sigma = 1.0);
std::pair<int, double> maxCoeff(const Eigen::VectorXd& in);

double radianClamp(double input);

Eigen::Quaterniond dartToQuat(Eigen::Vector3d in);
Eigen::Vector3d quatToDart(const Eigen::Quaterniond& in);
void quaternionNormalize(Eigen::Quaterniond& in);

void setBodyNodeColors(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& color);
void setSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector3d& color);
void setSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector4d& color);

Eigen::Quaterniond getYRotation(Eigen::Quaterniond q);

Eigen::Vector3d changeToRNNPos(Eigen::Vector3d pos);
Eigen::Isometry3d getJointTransform(dart::dynamics::SkeletonPtr skel, std::string bodyname);
Eigen::Vector4d rootDecomposition(dart::dynamics::SkeletonPtr skel, Eigen::VectorXd positions);
Eigen::VectorXd solveIK(dart::dynamics::SkeletonPtr skel, const std::string& bodyname, const Eigen::Vector3d& delta,  const Eigen::Vector3d& offset);
Eigen::VectorXd solveMCIK(dart::dynamics::SkeletonPtr skel, const std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>& constraints);
Eigen::VectorXd convertMGToTC(const Eigen::VectorXd& input, dart::dynamics::SkeletonPtr skel);
}

}