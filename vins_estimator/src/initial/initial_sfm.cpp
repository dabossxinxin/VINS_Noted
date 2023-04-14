#include "initial_sfm.h"

void GlobalSFM::triangulatePoint(
	Eigen::Matrix<double, 3, 4> &Pose0,
	Eigen::Matrix<double, 3, 4> &Pose1,
	Eigen::Vector2d &point0,
	Eigen::Vector2d &point1,
	Eigen::Vector3d &point_3d) {
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool GlobalSFM::solveFrameByPnP(
	Eigen::Matrix3d &R_initial,
	Eigen::Vector3d &P_initial,
	int i,
	std::vector<SFMFeature> &sfm_f) {
	// 将第i帧的特征点以及对应的3D路标点取出来
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; ++j) {
		if (sfm_f[j].state != true) {
			continue;
		}

		Eigen::Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); ++k) {
			if (sfm_f[j].observation[k].first == i) {
				Eigen::Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}

	// 根据当前帧中2D-3D匹配点的数量确定当前匹配是否效果良好
	if (int(pts_2_vector.size()) < 15) {
		printf("unstable features tracki ng, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10) {
			return false;
		}
	}

	// 将R_initial与P_initial作为初始值参与PnP
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

	if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) {
		return false;
	}

	cv::Rodrigues(rvec, r);
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;
}

void GlobalSFM::triangulateTwoFrames(
	int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
	int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
	std::vector<SFMFeature> &sfm_f) {
	// 若给定的两帧图像是同一帧，不进行三角化
	if (frame0 == frame1) {
		return;
	}

	for (int j = 0; j < feature_num; ++j) {
		// 若当前点已经三角化了，不进行后续操作
		if (sfm_f[j].state == true) {
			continue;
		}
		
		// 取出当前特征点在两帧图像中的观测位置
		bool has_0 = false, has_1 = false;
		Eigen::Vector2d point0;
		Eigen::Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); ++k) {
			if (sfm_f[j].observation[k].first == frame0) {
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1) {
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}

		// 若特征点在两图像帧间都能被观测到，则进行三角化
		if (has_0 && has_1) {
			Eigen::Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
		}							  
	}
}

/*!
*  @brief 纯视觉sfm，求解窗口中的所有图像帧相对于l帧的位姿和特征点坐标
*  @param[in]	frame_num	窗口总帧数（frame_count + 1）
*  @param[out]	q			窗口内图像帧的旋转四元数q（相对于第l帧）
*  @param[out]	T			窗口内图像帧的平移向量T（相对于第l帧）
*  @param[in]	l			第l帧
*  @param[in]	relative_R	当前帧到第l帧的旋转矩阵
*  @param[in]	relative_T 	当前帧到第l帧的平移向量
*  @param[in]	sfm_f		所有特征点
*  @param[out]	sfm_tracked_points 所有在sfm中优化后的特征点ID和坐标
*  @return		bool		纯视觉SFM是否求解成功
*/
bool GlobalSFM::construct(
	int frame_num, 
	Eigen::Quaterniond* q, 
	Eigen::Vector3d* T, 
	int l,
	const Eigen::Matrix3d relative_R, 
	const Eigem::Vector3d relative_T,
	std::vector<SFMFeature> &sfm_f, 
	std::map<int, Eigen::Vector3d> &sfm_tracked_points) {
	feature_num = sfm_f.size();

	// 设第l帧为原点，根据当前帧到第l帧的相对平移和旋转，获取当前帧位姿
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	q[frame_num - 1] = q[l] * Eigen::Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;

	Eigen::Matrix3d c_Rotation[frame_num];			// 窗口中所有帧相对于相机系的旋转
	Eigen::Vector3d c_Translation[frame_num];		// 窗口中所有帧相对于相机系的平移
	Eigen::Quaterniond c_Quat[frame_num];			// 窗口中所有帧相对于相机系的旋转四元数
	double c_rotation[frame_num][4];				// 窗口中所有帧在相机系下的四元数，供ceres优化用
	double c_translation[frame_num][3];				// 窗口中所有帧在相机系下的平移，供ceres优化用
	Eigen::Matrix<double, 3, 4> Pose[frame_num];	// 第l帧相机系下所有帧的位置和姿态

	// 第l帧的位置姿态转换到第l帧的相机系下表示
	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	// 当前帧的位置和姿态转换到第l帧的相机系下表示
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

	// 求解第l帧与第frame_num帧之间帧的位姿以及路标点
	for (int i = l; i < frame_num - 1 ; ++i) {
		if (i > l) {
			// 使用i-1帧的姿态作为第i帧的初始值进行PnP求解第i帧的姿态
			Eigen::Matrix3d R_initial = c_Rotation[i - 1];
			Eigen::Vector3d P_initial = c_Translation[i - 1];
			if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) {
				return false;
			}
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}

	// 第l帧与l+1~frame_num-1帧再次进行三角化恢复更多的路标点
	for (int i = l + 1; i < frame_num - 1; ++i) {
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	}
		
	// 求解第l帧与第0帧之间帧的位姿以及路标点
	for (int i = l - 1; i >= 0; --i) {
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) {
			return false;
		}
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];

		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}

	// 三角化其他未在上述过程中被三角化的点
	for (int j = 0; j < feature_num; j++) {
		if (sfm_f[j].state == true) {
			continue;
		}

		if ((int)sfm_f[j].observation.size() >= 2) {
			Eigen::Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
		}		
	}

	// 使用ceres进行窗口中路标点以及相机姿态的优化
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization =
		new ceres::QuaternionParameterization();

	// 向优化问题中添加位姿优化量
	for (int i = 0; i < frame_num; ++i) {
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);

		// 单目问题具有姿态、位置和尺度七个自由度
		// 1、通过固定l帧姿态固定单目问题中的姿态不确定性
		// 2、通过固定l帧与当前帧的位置固定单目问题中位置与尺度不确定性
		if (i == l) {
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1) {
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

	// 向优化问题中添加重投影优化残差以及路标点优化量
	for (int i = 0; i < feature_num; ++i) {
		if (sfm_f[i].state != true) {
			continue;
		}

		problem.AddParameterBlock(sfm[i].position, 3);

		for (int j = 0; j < int(sfm_f[i].observation.size()); ++j) {
			int l = sfm_f[i].observation[j].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
				sfm_f[i].observation[j].second.x(),
				sfm_f[i].observation[j].second.y());

			problem.AddResidualBlock(cost_function, NULL, c_rotation[l],
				c_translation[l], sfm_f[i].position);
		}
	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || 
		summary.final_cost < 5e-03) {
		LOG(INFO) << "vision only BA converge";
	}
	else {
		LOG(INFO) << "vision only BA not converge ";
		return false;
	}
	
	// 获取窗口中所有帧相对于l帧的姿态
	for (int i = 0; i < frame_num; ++i) {
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
	}

	// 获取窗口中所有帧相对于l帧的位置
	for (int i = 0; i < frame_num; ++i) {
		T[i] = -1 * (q[i] * Eigen::Vector3d(
			c_translation[i][0],
			c_translation[i][1],
			c_translation[i][2]));
	}

	// 获取窗口中所有路标点优化后的位置
	for (int i = 0; i < (int)sfm_f.size(); ++i) {
		if (sfm_f[i].state) {
			sfm_tracked_points[sfm_f[i].id] = Eigen::Vector3d(
				sfm_f[i].position[0],
				sfm_f[i].position[1],
				sfm_f[i].position[2]);
		}
	}
	return true;
}

