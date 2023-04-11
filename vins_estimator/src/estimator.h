#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>

/**
* @class Estimator 状态估计器
* @Description IMU预积分，图像IMU融合的初始化和状态估计，重定位
* detailed 
*/
class Estimator
{
public:
	Estimator();

	void setParameter();

	// interface
	void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
	void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
	void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

	// internal
	void clearState();
	bool initialStructure();
	bool visualInitialAlign();
	bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
	void slideWindow();
	void solveOdometry();
	void slideWindowNew();
	void slideWindowOld();
	void optimization();
	void vector2double();
	void double2vector();
	bool failureDetection();

	// 求解器状态
	enum SolverFlag {
		INITIAL,
		NON_LINEAR
	};
	SolverFlag solver_flag;

	// 边缘化状态
	enum MarginalizationFlag {
		MARGIN_OLD = 0,
		MARGIN_SECOND_NEW = 1
	};
	MarginalizationFlag  marginalization_flag;

	Eigen::Vector3d g;
	Eigen::MatrixXd Ap[2], backup_A;
	Eigen::VectorXd bp[2], backup_b;

	Eigen::Matrix3d ric[NUM_OF_CAM];
	Eigen::Vector3d tic[NUM_OF_CAM];

	// 滑动窗口中的[P,V,R,Ba,Bg]
	Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];
	Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];
	Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];
	Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];
	Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];
	double td;

	Eigen::Matrix3d back_R0, last_R, last_R0;
	Eigen::Vector3d back_P0, last_P, last_P0;
	std_msgs::Header Headers[(WINDOW_SIZE + 1)];

	IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
	Eigen::Vector3d acc_0, gyr_0;

	// 滑动窗口中的dt,a,v
	std::vector<double> dt_buf[(WINDOW_SIZE + 1)];
	std::vector<Eigen::Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
	std::vector<Eigen::Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

	int frame_count;
	int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

	FeatureManager f_manager;
	MotionEstimator m_estimator;
	InitialEXRotation initial_ex_rotation;

	bool first_imu;
	bool is_valid, is_key;
	bool failure_occur;

	std::vector<Eigen::Vector3d> point_cloud;
	std::vector<Eigen::Vector3d> margin_cloud;
	std::vector<Eigen::Vector3d> key_poses;
	double initial_timestamp;

	double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
	double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
	double para_Feature[NUM_OF_F][SIZE_FEATURE];
	double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
	double para_Retrive_Pose[SIZE_POSE];
	double para_Td[1][1];
	double para_Tr[1][1];

	int loop_window_index;

	MarginalizationInfo *last_marginalization_info;
	std::vector<double*> last_marginalization_parameter_blocks;

	// key是时间戳，val是图像帧
	// 图像帧中保存了图像帧的特征点、时间戳、位姿Rt，预积分对象pre_integration，是否是关键帧。
	std::map<double, ImageFrame> all_image_frame;
	IntegrationBase *tmp_pre_integration;

	// 重定位所需的变量
	bool relocalization_info;
	double relo_frame_stamp;
	double relo_frame_index;
	int relo_frame_local_index;
	std::vector<Eigen::Vector3d> match_points;
	double relo_Pose[SIZE_POSE];
	Eigen::Matrix3d drift_correct_r;
	Eigen::Vector3d drift_correct_t;
	Eigen::Vector3d prev_relo_t;
	Eigen::Matrix3d prev_relo_r;
	Eigen::Vector3d relo_relative_t;
	Eigen::Quaterniond relo_relative_q;
	double relo_relative_yaw;
};
