#ifndef __VMCON_SIM_WINDOW_H__
#define __VMCON_SIM_WINDOW_H__
#include "Camera.h"
#include "GLUTWindow.h"
#include "GLfunctions.h"
#include "DART_interface.h"
#include "Character.h"
#include <string>
/**
*
* @brief Modified SimWindow class in dart.
* @details Renders on window with the recorded data generated in sim.
*
*/
class SimWindow : public GUI::GLUTWindow
{
public:
	/// Constructor.
	SimWindow();

	/// Constructor.
	SimWindow(std::string filename);
 
	/// World object pointer
	dart::simulation::WorldPtr mWorld;
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

	/// Stores the data for SimWindow::Motion.
	void mouse(int button, int state, int x, int y) override;

	/// The user interactions with mouse. Camera view is set here.
	void motion(int x, int y) override;

	/// Reaction to window resizing.
	void reshape(int w, int h) override;

	/// 
	void timer(int value) override;

	/// Screenshot. The png file will be stored as ./frames/Capture/[number].png
	void screenshot();

	/// Set the skeleton positions in mWorld to the positions at n frame.
	void setFrame(int n);

	/// Set the skeleton positions in mWorld to the postions at the next frame.
	void nextFrame();

	/// set the skeleton positions in mWorld to the postions at 1/30 sec later.
	void nextFrameRealTime();

	/// Set the skeleton positions in mWorld to the postions at the previous frame.
	void prevFrame();

	
	bool mIsAuto;
	bool mIsCapture;
	bool mShowCharacter;
	bool mShowRef;
	bool mTrackCamera;
	int mCurFrame;
	int mTotalFrame;
	bool mIsRefExist;
	std::vector<Eigen::VectorXd> mRecords, mRefRecords;
	std::vector<Eigen::Vector3d> mTargetRecords;

	int mSkeletonDrawType;
};

#endif