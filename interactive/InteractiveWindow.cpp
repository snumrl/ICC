#include <GL/glew.h>
#include "InteractiveWindow.h"
#include "dart/external/lodepng/lodepng.h"
#include "Utils.h"
#include "Configurations.h"
#include "Character.h"
#include "ThrowingBall.h"
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
#include <chrono>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

using namespace GUI;
using namespace dart::simulation;
using namespace dart::dynamics;


InteractiveWindow::
InteractiveWindow(std::string network_path, std::string network_type)
	:GLUTWindow(),mTrackCamera(false),mIsAuto(false),mIsCapture(false)
	,mShowPrediction(true),mShowCharacter(true),mSkeletonDrawType(0)
{
	// load configurations
	std::string configuration_filepath = network_path + std::string("/configuration.xml");
	ICC::Configurations::instance().LoadConfigurations(configuration_filepath);
	// set configurations for interactive mode
	ICC::Configurations::instance().setEarlyTermination(false);
	ICC::Configurations::instance().setReferenceType(ICC::ReferenceType::INTERACTIVE);

	this->mEnvironment = new ICC::Environment();
	ICC::Utils::setSkeletonColor(this->mEnvironment->getActor()->getSkeleton(), Eigen::Vector4d(0.73, 0.73, 0.78, 1.0));

	this->mStateSize = this->mEnvironment->getStateSize();
	this->mActionSize = this->mEnvironment->getActionSize();

	this->mRefSkel = this->mEnvironment->getActor()->getSkeleton()->cloneSkeleton("Ref");
	for(int i = 0; i < this->mRefSkel->getNumBodyNodes(); i++){
		auto bn = this->mRefSkel->getBodyNode(i);
		bn->setCollidable(false);
	}
	ICC::Utils::setSkeletonColor(this->mRefSkel, Eigen::Vector4d(92/255.,145/255.,236/255., 1.0));

	this->mDisplayTimeout = 33;
	this->mCurFrame = 0;
	this->mTotalFrame = 0;

	// initial target
	this->mTarget << 0.0, 0.88, 30.0;

	// initialize python objects
	try{
	    Py_Initialize();
		np::initialize();

		this->mTrackingController = p::import("rl.TrackingController").attr("TrackingController")();
		this->mTrackingController.attr("initializeForInteractiveControl")(
			1, // num_slaves
			configuration_filepath, // configurations file path
			this->mStateSize,
			this->mActionSize
		);

		// create motion generator

		this->mMotionGenerator = this->mTrackingController.attr("_motionGenerator");

		// load network
		this->mTrackingController.attr("loadNetworks")(network_path, network_type);
		this->mMotionGenerator.attr("loadNetworks")(this->getPythonTarget());
	}
	catch(const  p::error_already_set&)
	{
		PyErr_Print();
	}

	// get initial pose(joint angles and vels)
	this->getPredictions();
	this->mEnvironment->reset();

	this->mTotalFrame = 0;

	this->record();

	this->setFrame(0);
}

p::list
InteractiveWindow::
getPythonTarget()
{
	// convert target
	p::list target;
	target.append(this->mTarget[2]*100);
	target.append(this->mTarget[0]*100);
	target.append(this->mTarget[1]*100);

	p::list target_wrapper;
	target_wrapper.append(target);

	return target_wrapper;
}

void 
InteractiveWindow::
getPredictions()
{
	try{
		// clear predictions
		this->mEnvironment->clearReferenceManager();

		p::list target = this->getPythonTarget();

		Eigen::VectorXd pos;
		// get prediction
		pos = ICC::Utils::toEigenVector(np::from_object(this->mMotionGenerator.attr("getReferences")(target)), ICC::Configurations::instance().getTCMotionSize());
		this->mEnvironment->addReference(ICC::Utils::convertMGToTC(pos, this->mEnvironment->getActor()->getSkeleton()));
		this->mEnvironment->addReferenceTarget(this->mTarget);


		this->mMotionGenerator.attr("saveState")();


		pos = ICC::Utils::toEigenVector(np::from_object(this->mMotionGenerator.attr("getReferences")(target)), ICC::Configurations::instance().getTCMotionSize());
		this->mEnvironment->addReference(ICC::Utils::convertMGToTC(pos, this->mEnvironment->getActor()->getSkeleton()));
		this->mEnvironment->addReferenceTarget(this->mTarget);


		this->mMotionGenerator.attr("loadState")();
	}
	catch(const  p::error_already_set&)
	{
		PyErr_Print();
	}
}

void
InteractiveWindow::
step()
{
	// set dynamic pose to rnn
	Eigen::VectorXd dynamic_pose = ICC::Utils::convertTCToMG(this->mEnvironment->getActor()->getSkeleton()->getPositions(), this->mEnvironment->getActor()->getSkeleton());
	np::ndarray converted_dynamic_pose = ICC::Utils::toNumPyArray(dynamic_pose);
	this->mMotionGenerator.attr("setDynamicPose")(converted_dynamic_pose);

	this->getPredictions();

	Eigen::VectorXd state = this->mEnvironment->getState();
	np::ndarray converted_state = ICC::Utils::toNumPyArray(state);
	converted_state = converted_state.reshape(p::make_tuple(1, converted_state.shape(0)));
	// Eigen::VectorXd action = Eigen::VectorXd::Zero(this->mActionSize);
	Eigen::VectorXd action = ICC::Utils::toEigenVector(np::from_object(this->mTrackingController.attr("getActionsForInteractiveControl")(converted_state)), this->mActionSize);
	this->mEnvironment->setAction(action);
	this->mEnvironment->step(false);

	this->record();
	
}

void
InteractiveWindow::
reset()
{
	// clear records
	this->mTotalFrame = 0;
	this->mRecords.clear();
	this->mTargetRecords.clear();
	this->mPredictionRecords.clear();
	this->mBallRecords.clear();

	// clear reference manager
	this->mEnvironment->clearReferenceManager();

	// clear motion generator
	this->mMotionGenerator.attr("resetAll")(this->getPythonTarget());

	// get first predictions
	this->getPredictions();

	// reset environment
	this->mEnvironment->reset();

	// get first frame
	this->mTotalFrame = 0;
	this->mCurFrame = 0;

	this->record();

	this->setFrame(0);
}

void 
InteractiveWindow::
record()
{
	this->mRecords.emplace_back(this->mEnvironment->getActor()->getSkeleton()->getPositions());
	this->mTargetRecords.emplace_back(this->mTarget);
	this->mPredictionRecords.emplace_back(this->mEnvironment->getReference(0));

	std::vector<Eigen::VectorXd> ballRecord;
	ballRecord.clear();

	for(auto& ball : this->mEnvironment->getThrowingBall()->mBalls){
		ballRecord.emplace_back(ball->getPositions());
	}
	this->mBallRecords.emplace_back(ballRecord);

	this->mTotalFrame++;
}

void
InteractiveWindow::
setFrame(int n)
{
	if( n < 0 || n >= this->mTotalFrame )
	{
		std::cout << "Frame exceeds limits" << std::endl;
		return;
	}


	this->mEnvironment->getActor()->getSkeleton()->setPositions(this->mRecords[n]);
	this->mRefSkel->setPositions(this->mPredictionRecords[n]);


	if(this->mTrackCamera){
		Eigen::Vector3d com = this->mEnvironment->getActor()->getSkeleton()->getRootBodyNode()->getCOM();
		Eigen::Isometry3d transform = this->mEnvironment->getActor()->getSkeleton()->getRootBodyNode()->getTransform();
		com[1] = 0.8;

		Eigen::Vector3d camera_pos;
		camera_pos << -3, 1, 1.5;
		camera_pos = camera_pos + com;
		camera_pos[1] = 2;

		mCamera->setCenter(com);
	}


}
void
InteractiveWindow::
nextFrame()
{
	if( this->mCurFrame == this->mTotalFrame - 1){
		this->step();
	}
	this->mCurFrame+=1;
	this->mCurFrame %= this->mTotalFrame;
	this->setFrame(this->mCurFrame);
}

void
InteractiveWindow::
prevFrame()
{
	this->mCurFrame-=1;
	if( this->mCurFrame < 0 ) this->mCurFrame = this->mTotalFrame -1;
	this->setFrame(this->mCurFrame);
}

void
InteractiveWindow::
drawSkeletons()
{
	if(mShowCharacter){
		GUI::DrawSkeleton(this->mEnvironment->getActor()->getSkeleton(), this->mSkeletonDrawType);
	}
	if(mShowPrediction){
		GUI::DrawSkeleton(this->mRefSkel, this->mSkeletonDrawType);
	}
}

void
InteractiveWindow::
drawGround()
{
	Eigen::Vector3d com_root = this->mEnvironment->getActor()->getSkeleton()->getRootBodyNode()->getCOM();
	double ground_height = 0.0;
	GUI::DrawGround((int)com_root[0], (int)com_root[2], ground_height);
}

void 
InteractiveWindow::
drawFlag(){
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

    {
        glPushMatrix();
        glTranslatef(goal[0], 0.05, goal[1]);
        glRotatef(90, 1, 0, 0);
        glColor3f(0.9, 0.53, 0.1);
        GUI::DrawCylinder(0.04, 0.1);
        glPopMatrix();

        glPushMatrix();
        glTranslatef(goal[0], 1.0, goal[1]);
        glRotatef(90, 1, 0, 0);
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
InteractiveWindow::
drawBalls()
{
    dart::dynamics::SkeletonPtr ball = this->mEnvironment->getThrowingBall()->mFirstBall;
    for (auto &pos:this->mBallRecords[this->mCurFrame]){
        ball->setPositions(pos);
        GUI::DrawSkeleton(ball);
    }
}

void
InteractiveWindow::
display() 
{

	glClearColor(0.9, 0.9, 0.9, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	Eigen::Vector3d com_root = this->mEnvironment->getActor()->getSkeleton()->getRootBodyNode()->getCOM();
	Eigen::Vector3d com_front = this->mEnvironment->getActor()->getSkeleton()->getRootBodyNode()->getTransform()*Eigen::Vector3d(0.0, 0.0, 2.0);
	mCamera->apply();
	
	glUseProgram(program);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPushMatrix();
    glScalef(1.0, -1.0, 1.0);
	initLights(com_root[0], com_root[2], com_front[0], com_front[2]);
	drawSkeletons();
	drawFlag();
	drawBalls();
	glPopMatrix();
	initLights(com_root[0], com_root[2], com_front[0], com_front[2]);
	drawGround();
	drawSkeletons();
	drawFlag();
	drawBalls();
	glDisable(GL_BLEND);




	glUseProgram(0);
	glutSwapBuffers();
	if(mIsCapture)
		this->screenshot();
}

void
InteractiveWindow::
toggleHeight(){
	if(this->mTarget[1] > 0.4){
		this->mTarget[1] = 0.0;
	}
	else{
		this->mTarget[1] = 0.88;
	}
	this->mMotionGenerator.attr("setTargetHeight")(this->mTarget[1]*100);
}

void
InteractiveWindow::
keyboard(unsigned char key,int x,int y) 
{
	switch(key)
	{
		case '1' :mShowCharacter= !mShowCharacter;break;
		case '2' :mShowPrediction= !mShowPrediction;break;
		case '[': this->prevFrame();break;
		case ']': this->nextFrame();break;
		case 'o': this->mCurFrame-=99; this->prevFrame();break;
		case 'p': this->mCurFrame+=99; this->nextFrame();break;
		case 's': std::cout << this->mCurFrame << std::endl;break;
		case 'r': this->reset();break;
		case 'C': mIsCapture = true; break;
		case 't': mTrackCamera = !mTrackCamera; this->setFrame(this->mCurFrame); break;
		case 'T': this->mSkeletonDrawType++; this->mSkeletonDrawType %= 2; break;
		case 'b': this->mEnvironment->createNewBall(false); break;
		case 'h': this->toggleHeight(); break;
		case ' ':
			mIsAuto = !mIsAuto;
			break;
		case 27: exit(0);break;
		default : break;
	}
}
void
InteractiveWindow::
mouse(int button, int state, int x, int y) 
{
	if(button == 3 || button == 4){
		if (button == 3)
		{
			mCamera->pan(0,-5,0,0);
		}
		else
		{
			mCamera->pan(0,5,0,0);
		}
	}
	else{
		if (state == GLUT_DOWN)
		{
			mIsDrag = true;
			mMouseType = button;
			mPrevX = x;
			mPrevY = y;
			if(mMouseType == GLUT_LEFT_BUTTON && glutGetModifiers()==GLUT_ACTIVE_SHIFT){
				GLdouble modelview[16], projection[16];
				GLint viewport[4];

				double height = glutGet(GLUT_WINDOW_HEIGHT);

				glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
				glGetDoublev(GL_PROJECTION_MATRIX, projection);
				glGetIntegerv(GL_VIEWPORT, viewport);

				double objx1, objy1, objz1;
				double objx2, objy2, objz2;

				int res1 = gluUnProject(x, height - y, 0, modelview, projection, viewport, &objx1, &objy1, &objz1);
				int res2 = gluUnProject(x, height - y, 10, modelview, projection, viewport, &objx2, &objy2, &objz2);

				this->mTarget[0] = objx1 + (objx2 - objx1)*(objy1)/(objy1-objy2);
				this->mTarget[2] = objz1 + (objz2 - objz1)*(objy1)/(objy1-objy2);
			}
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
InteractiveWindow::
motion(int x, int y) 
{
	if (!mIsDrag)
		return;

	int mod = glutGetModifiers();
	if (mMouseType == GLUT_LEFT_BUTTON)
	{
		// if(!mIsRotate)
		mCamera->translate(x,y,mPrevX,mPrevY);
		// else
		// 	mCamera->Rotate(x,y,mPrevX,mPrevY);
	}
	else if (mMouseType == GLUT_RIGHT_BUTTON)
	{
		mCamera->rotate(x,y,mPrevX,mPrevY);
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
InteractiveWindow::
reshape(int w, int h) 
{
	glViewport(0, 0, w, h);
	mCamera->apply();
}


void
InteractiveWindow::
timer(int value) 
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	if( mIsAuto )
		this->nextFrame();
	
	std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
	double elasped = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.;
	glutTimerFunc(std::max(0.0,mDisplayTimeout-elasped), timerEvent,1);
	glutPostRedisplay();
}


void InteractiveWindow::
screenshot() {
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
