#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

// 判断跟踪得到的特征点是否在图像边界内
bool inBorder(const cv::Point2f &pt);
// 去掉LK光流跟踪失败的特征点坐标
void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status);
// 去掉LK光流跟踪失败的特征点ID
void reduceVector(std::vector<int> &v, std::vector<uchar> status);

/**
* @class FeatureTracker
* @Description 视觉前端预处理：对每个相机进行角点LK光流跟踪
*/
class FeatureTracker
{
public:
	FeatureTracker();

	// 对图像使用光流法进行特征点跟踪
	void readImage(const cv::Mat &_img, double _cur_time);

	// 对跟踪得到的特征点进行排序并均匀化
	void setMask();

	// 添加当前帧中新检测到的特征点
	void addPoints();

	// 更新特征点的ID
	bool updateID(unsigned int i);

	// 读取相机的内参
	void readIntrinsicParameter(const string &calib_file);

	// 显示去畸变后的特征点，name为图像帧名称
	void showUndistortion(const string &name);

	// 通过fundamental矩阵去除跟踪的outlier
	void rejectWithF();

	// 去除特征点的畸变并计算每个特征点的速度
	void undistortedPoints();

	cv::Mat mask;			// 图像掩码
	cv::Mat fisheye_mask;	// 鱼眼镜头mask

	// prev_img是上一次发布的帧的图像数据
	// cur_img是光流跟踪的前一帧的图像数据
	// forw_img是光流跟踪的后一帧的图像数据
	cv::Mat prev_img, cur_img, forw_img;

	std::vector<cv::Point2f> n_pts;						// 每一帧中新提取的特征点
	std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
	std::vector<cv::Point2f> prev_un_pts, cur_un_pts;	// 归一化相机坐标系下的坐标
	std::vector<cv::Point2f> pts_velocity;				// 当前帧特征点x/y方向像素速度

	std::vector<int> ids;		//能够被跟踪到的特征点的id
	std::vector<int> track_cnt;	//当前帧forw_img中每个特征点被跟踪次数

	std::map<int, cv::Point2f> cur_un_pts_map;
	std::map<int, cv::Point2f> prev_un_pts_map;

	camodocal::CameraPtr m_camera;//相机模型

	double cur_time;	// 上一帧时间戳
	double prev_time;	// 当前帧时间戳

	static int n_id;	// 特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
};