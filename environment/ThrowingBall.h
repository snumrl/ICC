#ifndef __DEEP_PHYSICS_THROWINGBALL_H__
#define __DEEP_PHYSICS_THROWINGBALL_H__
#include "Character.h"
#include <dart/dart.hpp>
#include <random>
#include <iterator>

#define distance_min 0.5
#define distance_max 1

namespace ICC
{
    class ThrowingBall
    {
    public:
        ThrowingBall(dart::simulation::WorldPtr world, dart::dynamics::SkeletonPtr aim, double ball_radius=0.13, double ball_mass=3);

        dart::dynamics::SkeletonPtr createNewBall(bool randomFlag = true);
        bool checkReachGround(dart::dynamics::SkeletonPtr thisBall);
        bool checkReachGround(int index);
        void deleteBallAutomatically();
        void reset();
        void setAim(dart::dynamics::SkeletonPtr aim);

        double ball_radius;

        int mLastIndex=0;
        dart::dynamics::SkeletonPtr mAim;
        dart::simulation::WorldPtr mWorld;
        dart::dynamics::SkeletonPtr mFirstBall;
        std::vector<dart::dynamics::SkeletonPtr> mBalls;

        // std library objects that allow us to generate high-quality random numbers
        std::random_device mRD;
        std::mt19937 mMT;
        std::uniform_real_distribution<double> mDistribution;

    private:
        double throw_time_interval= 1000;
        double time=0;
        Eigen::Vector6d randomPositionAndVelocity();
        std::vector<dart::dynamics::SkeletonPtr>::iterator deleteBall(int index);
        std::vector<dart::dynamics::SkeletonPtr>::iterator deleteBall(dart::dynamics::SkeletonPtr thisBall);

    };
}

#endif