#include "ThrowingBall.h"

//#include "SkeletonBuilder.h"
using namespace dart::dynamics;
namespace ICC
{

ThrowingBall::ThrowingBall(dart::simulation::WorldPtr world, dart::dynamics::SkeletonPtr aim, double ball_radius, double ball_mass )
:mWorld(world), mAim(aim), mRD(), mMT(mRD()), mDistribution(), ball_radius(ball_radius)
{
    this->mFirstBall = Skeleton::create("ball");

    // Create a body for the ball
    BodyNodePtr body = this->mFirstBall->createJointAndBodyNodePair<FreeJoint>(nullptr).second;

    // Box
    std::shared_ptr<BoxShape> box(new BoxShape(Eigen::Vector3d(ball_radius, ball_radius, ball_radius)));
    body->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(box);

    // Set up inertia for the ball
    dart::dynamics::Inertia inertia;
    inertia.setMass(ball_mass);
    inertia.setMoment(box->computeInertia(ball_mass));
    body->setInertia(inertia);

    reset();
}


void
ThrowingBall::
reset()
{
    this->mBalls.clear();
    this->mLastIndex=0;
}

Eigen::Vector6d
ThrowingBall::
randomPositionAndVelocity()
{
    // Compute the position for the new ball
    Eigen::Vector3d x = this->mAim->getRootBodyNode()->getCOM();

    Eigen::Vector3d head_pos= this->mAim->getBodyNode("Head")->getWorldTransform().translation();
    double rd_height = x[1];
    double rd_around_angle = this->mDistribution(mMT) * 2* M_PI;  //[0, 2PI)
    double rd_distance = distance_min + this->mDistribution(mMT)*(distance_max-distance_min); // [distance_min, distance_max)
    Eigen::Vector3d dx = rd_distance * Eigen::Vector3d( cos(rd_around_angle), 0, sin(rd_around_angle)) + Eigen::Vector3d(0,rd_height,0);

    x+= dx;

    double head_angle = 0; //atan2(head_pos[1]-x[1], dx.norm());
    double rd_throw_angle = this->mDistribution(mMT) * head_angle; //[0, head_angle)
    Eigen::Vector3d throw_dir = cos(rd_throw_angle)*(-dx)+ sin(rd_throw_angle)* Eigen::Vector3d(0,1,0);

    Eigen::Vector6d rd_pos_and_vel = Eigen::Vector6d::Zero();
    rd_pos_and_vel << x, 6*throw_dir;

    return rd_pos_and_vel;
}

dart::dynamics::SkeletonPtr 
ThrowingBall::
createNewBall(bool randomFlag)
{
    SkeletonPtr newBall = this->mFirstBall->cloneSkeleton();

    newBall->setName("ball_"+std::to_string(mLastIndex));

    Eigen::Vector6d newBall_pos = Eigen::Vector6d::Zero();
    Eigen::Vector6d newBall_vel = Eigen::Vector6d::Zero();

    Eigen::Vector6d newBall_pos_and_vel = this->randomPositionAndVelocity();
    if (!randomFlag){
        Eigen::Vector3d x = mAim->getRootBodyNode()->getCOM();

        Eigen::Vector3d head_pos = mAim->getBodyNode("Spine")->getWorldTransform().translation();
        double rd_height = x[1];//mDistribution(mMT)* (head_pos[1]-x[1]);
        double rd_around_angle = M_PI * 0.2;  //[0, 2PI)
        double rd_distance = distance_max; // [distance_min, distance_max)
        Eigen::Vector3d dx = rd_distance * Eigen::Vector3d( cos(rd_around_angle), 0, sin(rd_around_angle)) + Eigen::Vector3d(0,rd_height,0);
        x+= dx;

        double rd_throw_angle = 0; //[0, head_angle)
        Eigen::Vector3d throw_dir = cos(rd_throw_angle)*(-dx) + sin(rd_throw_angle)* Eigen::Vector3d(0,1,0);

        Eigen::Vector6d rd_pos_and_vel = Eigen::Vector6d::Zero();
        rd_pos_and_vel << x, 6*throw_dir;

        newBall_pos_and_vel = rd_pos_and_vel;
    }
    newBall_pos.tail<3>() = newBall_pos_and_vel.head<3>();
    newBall_vel.tail<3>() = newBall_pos_and_vel.tail<3>();
    newBall->setPositions(newBall_pos);

    newBall->setVelocities(newBall_vel);


    auto collisionEngine = this->mWorld->getConstraintSolver()->getCollisionDetector();
    auto collisionGroup = this->mWorld->getConstraintSolver()->getCollisionGroup();
    auto newGroup = collisionEngine->createCollisionGroup(newBall.get());

    dart::collision::CollisionOption option;
    dart::collision::CollisionResult result;
    bool collision = collisionGroup->collide(newGroup.get(), option, &result);


    // If the new object is not in collision
    this->mWorld->addSkeleton(newBall);
    this->mBalls.push_back(newBall);
    this->mLastIndex++;
    return newBall;
}


bool 
ThrowingBall::
checkReachGround(SkeletonPtr thisBall)
{
    Eigen::Vector3d x = thisBall->getBodyNode(0)->getWorldTransform().translation();
    if(x[1] <= ball_radius+0.01) return true;
    else return false;
}

bool 
ThrowingBall::
checkReachGround(int index)
{
    return checkReachGround(mBalls[index]);
}


std::vector<dart::dynamics::SkeletonPtr>::iterator 
ThrowingBall::
deleteBall(int index)
{
    std::vector<dart::dynamics::SkeletonPtr>::iterator it;
    if(mBalls.size() > index)
    {
        SkeletonPtr thisBall = this->mBalls.at(index);
        it = mBalls.erase(this->mBalls.begin()+index);
        this->mWorld->removeSkeleton(thisBall);
    }
    return it;
}

std::vector<dart::dynamics::SkeletonPtr>::iterator 
ThrowingBall::
deleteBall(SkeletonPtr thisBall)
{
    int index= -1;
    std::vector<dart::dynamics::SkeletonPtr>::iterator it;
    int i=0;
    for(auto&ball: mBalls)
    {
        if(ball == thisBall)
        {
            index= i;
            break;
        }
        i++;
    }

    if(index!= -1)
    {
        it = this->mBalls.erase(this->mBalls.begin()+index);
        this->mWorld->removeSkeleton(thisBall);
    }

    return it;
}

void 
ThrowingBall::
deleteBallAutomatically()
{
    for(std::vector<dart::dynamics::SkeletonPtr>::iterator it= mBalls.begin(); it!=mBalls.end(); )
    {
        dart::dynamics::SkeletonPtr ball= *it;
        if(checkReachGround(ball))
        {
            it=deleteBall(ball);
        }
        else
        {
            it++;
        }
    }
}

}