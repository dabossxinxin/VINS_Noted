#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

/*!
*  @class ImageFrame 图像帧
*  @Description  图像帧类可由图像帧的特征点与时间戳构造，
*                此外还保存了位姿Rt，预积分对象pre_integration，是否是关键帧。
*/
class ImageFrame
{
public:
    ImageFrame(){};

    ImageFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):
        t{_t},is_key_frame{false} {
        points = _points;
    }

    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>> > > points;   // 当前帧的特征点
    double          t;                  // 当前真时间戳
    Eigen::Matrix3d R;                  // 当前帧姿态
    Eigen::Vector3d T;                  // 当前帧位置
    IntegrationBase *pre_integration;   // 当前帧预积分值
    bool            is_key_frame;       // 当前帧是否为关键帧
};

bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d* Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x);