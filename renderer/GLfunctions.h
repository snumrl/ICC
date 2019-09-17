#ifndef __GL_FUNCTIONS_H__
#define __GL_FUNCTIONS_H__
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <assimp/cimport.h>
#include <assimp/scene.h>
namespace GUI
{
	void DrawSphere(double r);
    void DrawOpenDome(double r, int slices, int stacks);
    void DrawOpenDomeQuater(double r, int slices, int stacks);
	void DrawCapsule(double radius, double height);
	void DrawCylinder(double radius, double height);
	void DrawCube(const Eigen::Vector3d& size);
	void DrawTetrahedron(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& p2,const Eigen::Vector3d& p3,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8));
	void DrawTriangle(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& p2,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8));
	void DrawLine(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8));
	void DrawPoint(const Eigen::Vector3d& p0,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8));
	void DrawArrow3D(const Eigen::Vector3d& _pt, const Eigen::Vector3d& _dir,
                 const double _length, const double _thickness,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8),
                 const double _arrowThickness = -1);

	// void drawCylinder(double _radius, double _height,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8), int slices = 16, int stacks = 16);

	void DrawRoundedBoxPlanes(const Eigen::Vector3d& size, double radius);
    void DrawRoundedBoxCylinder(const Eigen::Vector2d& size, double height, double radius);
    void DrawRoundedBoxSphere(const Eigen::Vector3d& size, double radius);
	void DrawRoundedBox(const Eigen::Vector3d& size, double radius, const Eigen::Vector3d& uniform_color);
	void DrawRoundedBox(const Eigen::Vector3d& size, double radius);
	void DrawBezierCurve(
		const Eigen::Vector3d& p0,
		const Eigen::Vector3d& p1,
		const Eigen::Vector3d& p2,
		const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8));
	void DrawMesh(const Eigen::Vector3d& _scale, const aiScene* _mesh,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8));

	void DrawStringOnScreen(float _x, float _y, const std::string& _s,bool _bigFont,const Eigen::Vector3d& color=Eigen::Vector3d(0.8,0.8,0.8));

	void DrawGround(int com_x, int com_z, double ground_height);

};

#endif