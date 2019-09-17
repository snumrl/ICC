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
	void DrawSkeletons();
	void DrawGround();
	void Display() override;

	/// The user interactions with keyboard.
	/// [ : Frame --
	/// ] : Frame ++
	/// r : Frame = 0
	/// C : Capture
	/// SPACE : Play
	/// ESC : exit
	void Keyboard(unsigned char key,int x,int y) override;

	/// Stores the data for SimWindow::Motion.
	void Mouse(int button, int state, int x, int y) override;

	/// The user interactions with mouse. Camera view is set here.
	void Motion(int x, int y) override;

	/// Reaction to window resizing.
	void Reshape(int w, int h) override;

	/// 
	void Timer(int value) override;

	/// Screenshot. The png file will be stored as ./frames/Capture/[number].png
	void Screenshot();

	/// Set the skeleton positions in mWorld to the positions at n frame.
	void SetFrame(int n);

	/// Set the skeleton positions in mWorld to the postions at the next frame.
	void NextFrame();

	/// set the skeleton positions in mWorld to the postions at 1/30 sec later.
	void NextFrameRealTime();

	/// Set the skeleton positions in mWorld to the postions at the previous frame.
	void PrevFrame();
	bool mIsRotate;
	bool mIsAuto;
	bool mIsCapture;
	bool mShowCharacter;
	bool mShowRef;
	bool mShowMod;
	bool mShowRootTraj;
	bool mTrackCamera;
	double mTimeStep;
	int mCurFrame;
	int mTotalFrame;
	bool mIsVelExist;
	bool mIsRefExist;
	bool mIsModExist;
	bool mIsGoalExist;
	bool mIsFootExist;
	std::vector<Eigen::VectorXd> mRecords, mRefRecords, mModRecords;
	std::vector<Eigen::Vector3d> mVelRecords, mGoalRecords;
	std::vector<Eigen::Vector3d> mRootTrajectories, mRootTrajectoriesRef;
	std::vector<Eigen::Vector2d> mFootRecords, mRefFootRecords;
    std::vector<Eigen::Vector3d> mBasketBallRecords;

	ICC::Character* mCharacter;

	int mSkeletonDrawType;
};

#endif