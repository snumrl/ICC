#pragma once

#include "Camera.h"
#include "GLUTWindow.h"
#include "GLfunctions.h"
#include "DART_interface.h"
#include "Character.h"
#include "SimWindow.h"
#include "Environment.h"
#include <string>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;
/**
/**
*
* @brief Modified InteractiveWindow class in dart.
* @details Interactive Window
*
*/
class InteractiveWindow : public GUI::GLUTWindow
{
public:
	/// Constructor.
	InteractiveWindow(std::string network_path="", std::string network_type="");
protected:
	/// Draw all the skeletons in mWorld. Lights and Camera are operated here.
	void drawSkeletons();
	void drawGround();
	void drawFlag();
	void display() override;

	/// The user interactions with keyboard.
	/// [ : Frame --
	/// ] : Frame ++
	/// r : Frame = 0
	/// C : Capture
	/// SPACE : Play
	/// ESC : exit
	void keyboard(unsigned char key,int x,int y) override;

	/// Stores the data for InteractiveWindow::Motion.
	void mouse(int button, int state, int x, int y) override;
 
	/// The user interactions with mouse. Camera view is set here.
	void motion(int x, int y) override;

	/// Reaction to window resizing.
	void reshape(int w, int h) override;

	/// 
	void timer(int value) override;

	/// Screenshot. The png file will be stored as ./frames/Capture/[number].png
	void screenshot();



	// TODO
	/// Set the skeleton positions in mWorld to the positions at n frame.
	void setFrame(int n);

	/// Set the skeleton positions in mWorld to the postions at the next frame.
	void nextFrame();

	/// Set the skeleton positions in mWorld to the postions at the previous frame.
	void prevFrame();

	void getPredictions();
	void step();


	/// Environment
	ICC::Environment *mEnvironment;

	int mStateSize, mActionSize;

	int mCurFrame;
	int mTotalFrame;

	Eigen::Vector3d mTarget;

	std::vector<Eigen::VectorXd> mRecords, mPredictionRecords;
	std::vector<Eigen::Vector3d> mTargetRecords;


	/// TrackingController
	p::object mTrackingController, mMotionGenerator;

	/// parameters for render
	bool mIsRotate;
	bool mIsAuto;
	bool mIsCapture;
	bool mShowCharacter;
	bool mShowPrediction;
	bool mTrackCamera;

	int mSkeletonDrawType;
};

