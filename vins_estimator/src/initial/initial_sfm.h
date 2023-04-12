#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace Eigen;
using namespace std;

struct SFMFeature
{
	bool	state;			// 当前特征点是否被三角化
	int		id;				// 当前特征点的ID
	double	position[3];	// 当前特征点的坐标
	double	depth;			// 当前特征点的深度
	std::vector<std::pair<int, Eigen::Vector2d>> observation; //所有观测到该特征点的图像帧ID和图像坐标
};

struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v) {}

	template <typename T>
	bool operator()(const T* const camera_R, 
		const T* const camera_T, 
		const T* point, 
		T* residuals) const {
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];
		residuals[0] = xp - T(observed_u);
		residuals[1] = yp - T(observed_v);
		return true;
	}

	// 初始化SFM中采用自动求导方式优化初始地图
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y) {
		return (new ceres::AutoDiffCostFunction<
			ReprojectionError3D, 2, 4, 3, 3>(
				new ReprojectionError3D(observed_x, observed_y)));
	}

	double observed_u;	// 归一化相机坐标横坐标
	double observed_v;	// 归一化相机坐标纵坐标
};

class GlobalSFM
{
public:
	GlobalSFM() {}

	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
		const Matrix3d relative_R, const Vector3d relative_T,
		vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	// 通过PnP求解地图中某一帧(l->i)的位姿
	bool solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i, std::vector<SFMFeature> &sfm_f);
	
	// 三角化两帧间某个特征点的深度
	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
		Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);

	// 三角化两帧间所有特征点的深度
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
		int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
		std::vector<SFMFeature> &sfm_f);

	int feature_num;
};