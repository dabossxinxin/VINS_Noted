#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;

/*!
*  @class BriefExtractor
*  @Description 计算图像特征点的BRIEF描述子
*/
class BriefExtractor
{
public:
	virtual void operator()(
		const cv::Mat &im, 
		std::vector<cv::KeyPoint> &keys, 
		std::vector<BRIEF::bitset> &descriptors) const;

	BriefExtractor(const std::string &pattern_file);

	DVision::BRIEF m_brief;
};

/*!
*  @class KeyFrame
*  @Description 通过BRIEF描述子匹配关键帧和回环候选帧
*/
class KeyFrame
{
public:
	// 关键帧构造函数
	KeyFrame(double _time_stamp, int _index, Eigen::Vector3d &_vio_T_w_i, Eigen::Matrix3d &_vio_R_w_i,
		cv::Mat &_image, std::vector<cv::Point3f> &_point_3d, std::vector<cv::Point2f> &_point_2d_uv,
		std::vector<cv::Point2f> &_point_2d_normal, std::vector<double> &_point_id, int _sequence);
	KeyFrame(double _time_stamp, int _index, Eigen::Vector3d &_vio_T_w_i, Eigen::Matrix3d &_vio_R_w_i,
		Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i, cv::Mat &_image, int _loop_index,
		Eigen::Matrix<double, 8, 1 > &_loop_info, std::vector<cv::KeyPoint> &_keypoints,
		std::vector<cv::KeyPoint> &_keypoints_norm, std::vector<BRIEF::bitset> &_brief_descriptors);

	bool findConnection(KeyFrame* old_kf);
	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();

	//void extractBrief();
	// 计算两个描述子之间的汉明距离
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	bool searchInAera(const BRIEF::bitset window_descriptor,
		const std::vector<BRIEF::bitset> &descriptors_old,
		const std::vector<cv::KeyPoint> &keypoints_old,
		const std::vector<cv::KeyPoint> &keypoints_old_norm,
		cv::Point2f &best_match,
		cv::Point2f &best_match_norm);
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
		std::vector<cv::Point2f> &matched_2d_old_norm,
		std::vector<uchar> &status,
		const std::vector<BRIEF::bitset> &descriptors_old,
		const std::vector<cv::KeyPoint> &keypoints_old,
		const std::vector<cv::KeyPoint> &keypoints_old_norm);

	// 通过fundmantal矩阵筛选匹配点对
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
		const std::vector<cv::Point2f> &matched_2d_old_norm,
		vector<uchar> &status);

	// 通过PnP筛选匹配点对
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
		const std::vector<cv::Point3f> &matched_3d,
		std::vector<uchar> &status,
		Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);

	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);

	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();

	double					time_stamp;
	int						index;
	int						local_index;
	Eigen::Vector3d			vio_T_w_i;
	Eigen::Matrix3d			vio_R_w_i;
	Eigen::Vector3d			T_w_i;
	Eigen::Matrix3d			R_w_i;
	Eigen::Vector3d			origin_vio_T;
	Eigen::Matrix3d			origin_vio_R;
	cv::Mat					image;
	cv::Mat					thumbnail;
	vector<cv::Point3f>		point_3d;					// 前端传过来的图像特征点对应3D坐标
	vector<cv::Point2f>		point_2d_uv;				// 前端传过来的图像特征点图像坐标
	vector<cv::Point2f>		point_2d_norm;				// 前端传过来的图像特征点归一化图像坐标
	vector<double>			point_id;
	vector<cv::KeyPoint>	keypoints;					// 增强回环效果新提的图像特征图像坐标
	vector<cv::KeyPoint>	keypoints_norm;				// 增强回环效果新提的图像特征归一化坐标
	vector<cv::KeyPoint>	window_keypoints;			// 前端传过来的图像特征的图像坐标
	vector<BRIEF::bitset>	brief_descriptors;			// 新提取的图像特征点对应的描述子
	vector<BRIEF::bitset>	window_brief_descriptors;	// 前端传过来的图像特征对应描述子
	bool					has_fast_point;
	int						sequence;
	bool					has_loop;
	int						loop_index;
	Eigen::Matrix<double, 8, 1> loop_info;
};