#include <GL/glew.h>
#include "SimWindow.h"
#include "dart/external/lodepng/lodepng.h"
#include "SkeletonBuilder.h"
#include "Utils.h"
#include "Character.h"
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
using namespace GUI;
using namespace dart::simulation;
using namespace dart::dynamics;


SimWindow::
SimWindow()
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false),mIsCapture(false)
	,mShowRef(true),mShowMod(true),mIsVelExist(false),mShowCharacter(true),mShowRootTraj(false)
{
	mWorld = std::make_shared<dart::simulation::World>();
	mCurFrame = 0;
	mDisplayTimeout = 33;
}

SimWindow::
SimWindow(std::string filename)
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false),mIsCapture(false)
	,mShowRef(true),mShowMod(true),mIsVelExist(false),mShowCharacter(true),mShowRootTraj(false),mSkeletonDrawType(0)
{
	this->mWorld = std::make_shared<dart::simulation::World>();

	std::ifstream ifs(filename);
	if(!ifs.is_open()){
		std::cout << "File doesn't exist" << std::endl;
		exit(0);
	}

	// read a number of characters
	std::string line;

	// read skeleton file
	ICC::Character* character;
	std::string skelfilename;

	std::getline(ifs, skelfilename);
	SkeletonPtr skel = ICC::SkeletonBuilder::buildFromFile(skelfilename);
	int nDof = skel->getNumDofs();
	this->mWorld->addSkeleton(skel);

	// for reference motion
	SkeletonPtr refSkel = skel->cloneSkeleton("Ref");
	this->mWorld->addSkeleton(refSkel);
	// character = new ICC::Humanoid(skelfilename);

	// SkeletonPtr modSkel = skel->clone("Mod");
	// mWorld->addSkeleton(modSkel);

	ICC::Utils::setSkeletonColor(skel, Eigen::Vector4d(0.73, 0.73, 0.78, 1.0));
	ICC::Utils::setSkeletonColor(refSkel, Eigen::Vector4d(235./255., 87./255., 87./255., 0.9));
	// ICC::SetSkeletonColor(modSkel, Eigen::Vector4d(93./255., 176./255., 89./255., 0.9));

	// read frame number
	std::getline(ifs, line);
	this->mTotalFrame = atoi(line.c_str());
	std::cout << "total frame : " << this->mTotalFrame << std::endl;

	// read joint angles per frame
	for(int i = 0; i < this->mTotalFrame; i++){
		std::getline(ifs, line);
		Eigen::VectorXd record = ICC::Utils::stringToVectorXd(line, nDof);
		this->mRecords.push_back(record);
	}

	for(int i = 0; i < this->mTotalFrame; i++){
		std::getline(ifs, line);
		Eigen::VectorXd ref = ICC::Utils::stringToVectorXd(line, nDof);
		this->mRefRecords.push_back(ref);
	}

	for(int i = 0; i < this->mTotalFrame; i++){
		std::getline(ifs, line);
		Eigen::VectorXd ref = ICC::Utils::stringToVectorXd(line, 3);
		this->mTargetRecords.push_back(ref);
	}


	mDisplayTimeout = 33;
	mCurFrame = 0;

	this->SetFrame(this->mCurFrame);

	ifs.close();
}

void
SimWindow::
SetFrame(int n)
{
	if( n < 0 || n >= this->mTotalFrame )
	{
		std::cout << "Frame exceeds limits" << std::endl;
		return;
	}


	SkeletonPtr skel = this->mWorld->getSkeleton("Humanoid");
	Eigen::VectorXd pos = this->mRecords[n];
	// pos.setZero();
	// pos[4] = 1.0;
	skel->setPositions(pos);


	skel = this->mWorld->getSkeleton("Ref");
	pos = this->mRefRecords[n];
	skel->setPositions(pos);

	// if(this->mIsModExist){
	// 	skel = this->mWorld->getSkeleton("Mod");
	// 	pos = this->mModRecords[n];
	// 	skel->setPositions(pos);
	// 	skel->computeForwardKinematics(true, false, false);
	// }


	if(this->mTrackCamera){
		Eigen::Vector3d com = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getCOM();
		Eigen::Isometry3d transform = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getTransform();
		com[1] = 0.8;

		Eigen::Vector3d camera_pos;
		// Eigen::Quaterniond t(Eigen::AngleAxisd(transform.linear()));
		// transform.linear() = ICC::GetYRotation(t).toRotationMatrix();
		camera_pos << -3, 1, 1.5;
		// camera_pos = transform * camera_pos;
		camera_pos = camera_pos + com;
		camera_pos[1] = 2;

		// mCamera->SetCamera(com, camera_pos, Eigen::Vector3d::UnitY());
		mCamera->SetCenter(com);
	}


}
void
SimWindow::
NextFrame()
{ 
	this->mCurFrame+=1;
	this->mCurFrame %= this->mTotalFrame;
	this->SetFrame(this->mCurFrame);
}
void
SimWindow::
NextFrameRealTime()
{
	// int count = this->mDisplayTimeout/(this->mTimeStep*1000.);
	int count = 1;
	this->mCurFrame += count;
	this->mCurFrame %= this->mTotalFrame;
	this->SetFrame(this->mCurFrame);
}
void
SimWindow::
PrevFrame()
{
	this->mCurFrame-=1;
	if( this->mCurFrame < 0 ) this->mCurFrame = this->mTotalFrame -1;
	this->SetFrame(this->mCurFrame);
}
void
SimWindow::
DrawSkeletons()
{
	auto skel = this->mWorld->getSkeleton("Ref");
	if(mShowRef){
		GUI::DrawSkeleton(skel, this->mSkeletonDrawType);
	}
	// skel = this->mWorld->getSkeleton("Mod");
	// if(mShowMod){
	// 	GUI::DrawSkeleton(skel, this->mSkeletonDrawType);
	// }
	skel = this->mWorld->getSkeleton("Humanoid");
	if(mShowCharacter){
		GUI::DrawSkeleton(skel, this->mSkeletonDrawType);
	}
}

void
SimWindow::
DrawGround()
{
	Eigen::Vector3d com_root = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getCOM();
	double ground_height = 0.0;//this->mWorld->getSkeleton("Ground")->getRootBodyNode()->getCOM()[1]+0.5;
	GUI::DrawGround((int)com_root[0], (int)com_root[2], ground_height);
}

void 
SimWindow::
DrawFlag(){
    Eigen::Vector2d goal;
    goal[0] = this->mTargetRecords[mCurFrame][0];
    goal[1] = this->mTargetRecords[mCurFrame][2];

    // glDisable(GL_LIGHTING);
    glLineWidth(10.0);

    Eigen::Vector3d orange= dart::Color::Orange();
    glColor3f(orange[0],orange[1],orange[2]);
    Eigen::Vector3d A, B, C;
    A = Eigen::Vector3d(0, 1.6, 0);
    B = Eigen::Vector3d(0, 1.97, 0);
    C = Eigen::Vector3d(0.3, 1.85, 0.3);
//            glVertex3f(this->mGoalRecords[this->mCurFrame][0], 1.6, this->mGoalRecords[this->mCurFrame][2]);
//            glVertex3f(this->mGoalRecords[this->mCurFrame][0], 1.97, this->mGoalRecords[this->mCurFrame][2]);
//            glVertex3f(this->mGoalRecords[this->mCurFrame][0]+0.3, 1.85, this->mGoalRecords[this->mCurFrame][2]+0.3);
//
//

    {
        glPushMatrix();
        glTranslatef(goal[0], 0.05, goal[1]);
        glRotatef(90, 1, 0, 0);
//            std::cout<<orange.transpose()<<std::endl; 1, 0.63, 0
        glColor3f(0.9, 0.53, 0.1);
        GUI::DrawCylinder(0.04, 0.1);
        glPopMatrix();

        glPushMatrix();
        glTranslatef(goal[0], 1.0, goal[1]);
        glRotatef(90, 1, 0, 0);
//            std::cout<<orange.transpose()<<std::endl; 1, 0.63, 0
        glColor3f(0.9, 0.53, 0.1);
        GUI::DrawCylinder(0.02, 2);
        glPopMatrix();

        glPushMatrix();
        glTranslatef(goal[0], 2.039, goal[1]);
        glColor3f(0.9, 0.53, 0.1);
        GUI::DrawSphere(0.04);
        glPopMatrix();
    }
    {
        glPushMatrix();
        glTranslatef(goal[0], 0.0, goal[1]);


        double initTheta = 40.0*std::cos(240/120.0);
        glRotated(initTheta, 0, 1, 0);

        int slice = 100;
        for (int i = 0; i < slice; i++) {
            Eigen::Vector3d p[5];
            p[1] = A + (C - A) * i / slice;
            p[2] = A + (C - A) * (i + 1) / slice;
            p[3] = B + (C - B) * (i + 1) / slice;
            p[4] = B + (C - B) * i / slice;

            for (int j = 4; j >= 1; j--) p[j][0] -= p[1][0], p[j][2] -= p[1][2];

            glPushMatrix();
            glRotated(1.0*std::cos(initTheta + (double)i/slice*2*M_PI) * std::exp(-(double)(slice-i)/slice), 0, 1, 0);

            glBegin(GL_QUADS);
            for (int j = 1; j <= 4; j++) glVertex3d(p[j][0], p[j][1], p[j][2]);
            glEnd();

            glTranslatef(p[2][0], 0, p[2][2]);

        }
        for (int i = 0; i < slice; i++) {
            glPopMatrix();
        }

        glPopMatrix();
    }

}

void
SimWindow::
Display() 
{

	glClearColor(0.9, 0.9, 0.9, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	Eigen::Vector3d com_root = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getCOM();
	Eigen::Vector3d com_front = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getTransform()*Eigen::Vector3d(0.0, 0.0, 2.0);
	mCamera->Apply();
	// initLights(com_root[0], com_root[2]);
	
	glUseProgram(program);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPushMatrix();
    glScalef(1.0, -1.0, 1.0);
	initLights(com_root[0], com_root[2], com_front[0], com_front[2]);
	DrawSkeletons();
	DrawFlag();
	glPopMatrix();
	initLights(com_root[0], com_root[2], com_front[0], com_front[2]);
	// glColor4f(0.7, 0.0, 0.0, 0.40);  /* 40% dark red floor color */
	DrawGround();
	DrawSkeletons();
	DrawFlag();
	glDisable(GL_BLEND);



	// if(exist_humanoid)
	// {
	// 	glBegin(GL_LINES);
	// 	glLineWidth(3.0);
	// 	for(double z =-100.0;z<=100.0;z+=1.0){
	// 		glVertex3f(z,100,0);
	// 		glVertex3f(z,-100,0);
	// 	}
	// 	for(double y =-0.0;y<=3.0;y+=1.0){
	// 		glVertex3f(100,y,0);
	// 		glVertex3f(-100,y,0);
	// 	}
	// 	glEnd();
	// }


	glUseProgram(0);
	glutSwapBuffers();
	if(mIsCapture)
		Screenshot();
	// glutPostRedisplay();
}
void
SimWindow::
Keyboard(unsigned char key,int x,int y) 
{
	switch(key)
	{
		case '`' :mIsRotate= !mIsRotate;break;
		case '1' :mShowCharacter= !mShowCharacter;break;
		case '2' :mShowRef= !mShowRef;break;
		case '3' :mShowMod= !mShowMod;break;
		case '0' :mShowRootTraj= !mShowRootTraj;break;
		case '[': this->PrevFrame();break;
		case ']': this->NextFrame();break;
		case 'o': this->mCurFrame-=99; this->PrevFrame();break;
		case 'p': this->mCurFrame+=99; this->NextFrame();break;
		case 's': std::cout << this->mCurFrame << std::endl;break;
		case 'r': this->mCurFrame=0;this->SetFrame(this->mCurFrame);break;
		case 'C': mIsCapture = true; break;
		case 't': mTrackCamera = !mTrackCamera; this->SetFrame(this->mCurFrame); break;
		case 'T': this->mSkeletonDrawType++; this->mSkeletonDrawType %= 2; break;
		case ' ':
			mIsAuto = !mIsAuto;
			break;
		case 27: exit(0);break;
		default : break;
	}
	// this->SetFrame(this->mCurFrame);

	// glutPostRedisplay();
}
void
SimWindow::
Mouse(int button, int state, int x, int y) 
{
	if(button == 3 || button == 4){
		if (button == 3)
		{
			mCamera->Pan(0,-5,0,0);
		}
		else
		{
			mCamera->Pan(0,5,0,0);
		}
	}
	else{
		if (state == GLUT_DOWN)
		{
			mIsDrag = true;
			mMouseType = button;
			mPrevX = x;
			mPrevY = y;
		}
		else
		{
			mIsDrag = false;
			mMouseType = 0;
		}
	}

	// glutPostRedisplay();
}
void
SimWindow::
Motion(int x, int y) 
{
	if (!mIsDrag)
		return;

	int mod = glutGetModifiers();
	if (mMouseType == GLUT_LEFT_BUTTON)
	{
		// if(!mIsRotate)
		mCamera->Translate(x,y,mPrevX,mPrevY);
		// else
		// 	mCamera->Rotate(x,y,mPrevX,mPrevY);
	}
	else if (mMouseType == GLUT_RIGHT_BUTTON)
	{
		mCamera->Rotate(x,y,mPrevX,mPrevY);
		// switch (mod)
		// {
		// case GLUT_ACTIVE_SHIFT:
		// 	mCamera->Zoom(x,y,mPrevX,mPrevY); break;
		// default:
		// 	mCamera->Pan(x,y,mPrevX,mPrevY); break;		
		// }

	}
	mPrevX = x;
	mPrevY = y;
	// glutPostRedisplay();
}
void
SimWindow::
Reshape(int w, int h) 
{
	glViewport(0, 0, w, h);
	mCamera->Apply();
}
void
SimWindow::
Timer(int value) 
{
	if( mIsAuto )
		this->NextFrameRealTime();
	
	glutTimerFunc(mDisplayTimeout, TimerEvent,1);
	glutPostRedisplay();
}


void SimWindow::
Screenshot() {
  static int count = 0;
  const char directory[8] = "frames";
  const char fileBase[8] = "Capture";
  char fileName[32];

  boost::filesystem::create_directories(directory);
  std::snprintf(fileName, sizeof(fileName), "%s%s%s%.4d.png",
                directory, "/", fileBase, count++);
  int tw = glutGet(GLUT_WINDOW_WIDTH);
  int th = glutGet(GLUT_WINDOW_HEIGHT);

  glReadPixels(0, 0,  tw, th, GL_RGBA, GL_UNSIGNED_BYTE, &mScreenshotTemp[0]);

  // reverse temp2 temp1
  for (int row = 0; row < th; row++) {
    memcpy(&mScreenshotTemp2[row * tw * 4],
           &mScreenshotTemp[(th - row - 1) * tw * 4], tw * 4);
  }

  unsigned result = lodepng::encode(fileName, mScreenshotTemp2, tw, th);

  // if there's an error, display it
  if (result) {
    std::cout << "lodepng error " << result << ": "
              << lodepng_error_text(result) << std::endl;
    return ;
  } else {
    std::cout << "wrote screenshot " << fileName << "\n";
    return ;
  }
}
