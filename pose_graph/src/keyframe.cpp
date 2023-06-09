#include "keyframe.h"

// 剔除status为0的特征点
template <typename Derived>
static void reduceVector(
	std::vector<Derived> &v, 
	std::vector<uchar> status) {
    int j = 0;
	for (int i = 0; i < int(v.size()); i++) {
		if (status[i]) {
			v[j++] = v[i];
		}
	}
    v.resize(j);
}

// 构造新的关键帧
KeyFrame::KeyFrame(
	double _time_stamp, 
	int _index, 
	Eigen::Vector3d &_vio_T_w_i, 
	Eigen::Matrix3d &_vio_R_w_i, 
	cv::Mat &_image,
	std::vector<cv::Point3f> &_point_3d, 
	std::vector<cv::Point2f> &_point_2d_uv, 
	std::vector<cv::Point2f> &_point_2d_norm,
	std::vector<double> &_point_id, 
	int _sequence) {
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;		
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	computeWindowBRIEFPoint();
	computeBRIEFPoint();

	// 为了节省内存空间，将图像析构掉
	if (!DEBUG_IMAGE) {
		image.release();
	}
}

// 载入之前的关键帧
KeyFrame::KeyFrame(
	double _time_stamp,
	int _index,
	Eigen::Vector3d &_vio_T_w_i,
	Eigen::Matrix3d &_vio_R_w_i,
	Eigen::Vector3d &_T_w_i,
	Eigen::Matrix3d &_R_w_i,
	cv::Mat &_image,
	int _loop_index,
	Eigen::Matrix<double, 8, 1 > &_loop_info,
	std::vector<cv::KeyPoint> &_keypoints,
	std::vector<cv::KeyPoint> &_keypoints_norm,
	std::vector<BRIEF::bitset> &_brief_descriptors) {
	time_stamp = _time_stamp;
	index = _index;
	//vio_T_w_i = _vio_T_w_i;
	//vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;

	if (DEBUG_IMAGE) {
		image = _image.clone();
		cv::resize(image, thumbnail, cv::Size(80, 60));
	}

	if (_loop_index != -1) {
		has_loop = true;
	}
	else {
		has_loop = false;
	}

	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
	brief_descriptors = _brief_descriptors;
}

// 计算前端图像特征点的描述子
void KeyFrame::computeWindowBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for(int i = 0; i < (int)point_2d_uv.size(); i++) {
	    cv::KeyPoint key;
	    key.pt = point_2d_uv[i];
	    window_keypoints.push_back(key);
	}
	extractor(image, window_keypoints, window_brief_descriptors);
}

// 为了保证闭环检测效果，额外提取特征点并计算描述子
void KeyFrame::computeBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int fast_th = 20; // corner detector response threshold
	if (1)
		/*
		 *void cv::FAST	(
		 *	InputArray 				image,		关键点所在的灰度图像。
		 *	std::vector<KeyPoint>& 	keypoints,	在图像上检测到关键点
		 *	int 					threshold,	中心像素的强度与该像素周围圆的像素之间的差异的阈值
		 *	bool 	 nonmaxSuppression = true 	是否对检测到的角点（关键点）应用非最大抑制
		 *)
		*/
		cv::FAST(image, keypoints, fast_th, true);
	else
	{
		std::vector<cv::Point2f> tmp_pts;
		// 检测500个新的特征点并将其放入keypoints
		cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
		for(int i = 0; i < (int)tmp_pts.size(); i++)
		{
		    cv::KeyPoint key;
		    key.pt = tmp_pts[i];
		    keypoints.push_back(key);
		}
	}

	// 计算keypoints中所有特征点的描述子
	extractor(image, keypoints, brief_descriptors);

	// 将特征点去畸变矫正
	for (int i = 0; i < (int)keypoints.size(); i++) {
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}

// 计算图像特征点的BRIEF描述子
void BriefExtractor::operator()(
	const cv::Mat &im, 
	std::vector<cv::KeyPoint> &keys, 
	std::vector<BRIEF::bitset> &descriptors) const {
	m_brief.compute(im, keys, descriptors);
}

// 关键帧中某个特征点描述子与回环帧中所有点的描述子匹配得到最佳匹配点
bool KeyFrame::searchInAera(
	const BRIEF::bitset window_descriptor,
    const std::vector<BRIEF::bitset> &descriptors_old,
    const std::vector<cv::KeyPoint> &keypoints_old,
    const std::vector<cv::KeyPoint> &keypoints_old_norm,
    cv::Point2f &best_match,
    cv::Point2f &best_match_norm) {
	cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;

    for(int i = 0; i < (int)descriptors_old.size(); i++) {
        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if(dis < bestDist) {
            bestDist = dis;
            bestIndex = i;
        }
    }

    // 汉明距离小于80并且index存在正常值时所得特征为最佳匹配
    if (bestIndex != -1 && bestDist < 80) {
      best_match = keypoints_old[bestIndex].pt;
      best_match_norm = keypoints_old_norm[bestIndex].pt;
      return true;
    } 
	else {
		return false;
	}
}

/**
 * @brief   将关键帧与回环帧进行BRIEF描述子匹配
 * @param[out]  matched_2d_old  	回环帧匹配后的二维坐标
 * @param[out]  matched_2d_old_norm 回环帧匹配后的二维归一化坐标
 * @param[out]  status				匹配状态，成功为1
 * @param[in]   descriptors_old		回环帧的描述子
 * @param[in] 	keypoints_old 		回环帧的二维坐标
 * @param[in] 	keypoints_old_norm	回环帧的二维归一化坐标
 * @return      void
*/
void KeyFrame::searchByBRIEFDes(
	std::vector<cv::Point2f> &matched_2d_old,
	std::vector<cv::Point2f> &matched_2d_old_norm,
    std::vector<uchar> &status,
    const std::vector<BRIEF::bitset> &descriptors_old,
    const std::vector<cv::KeyPoint> &keypoints_old,
    const std::vector<cv::KeyPoint> &keypoints_old_norm) {
    for(int i = 0; i < (int)window_brief_descriptors.size(); ++i) {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
		if (searchInAera(window_brief_descriptors[i],
			descriptors_old, keypoints_old,
			keypoints_old_norm, pt, pt_norm)) {
			status.push_back(1);
		}
		else {
			status.push_back(0);
		}
		matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }
}

// 通过RANSAC的基本矩阵检验去除匹配异常的点
void KeyFrame::FundmantalMatrixRANSAC(
	const std::vector<cv::Point2f> &matched_2d_cur_norm,
    const std::vector<cv::Point2f> &matched_2d_old_norm,
    std::vector<uchar> &status) {
	int n = (int)matched_2d_cur_norm.size();
	for (int i = 0; i < n; i++) {
		status.push_back(0);
	}

    if (n >= 8) {
        std::vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++) {
			// 计算当前帧特征点图像坐标
			double FOCAL_LENGTH = 460.0;
			double tmp_x, tmp_y;
			tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
			tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
			tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

			// 计算回环帧特征点图像坐标
            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
		cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}

// 通过RANSAC的PNP检验去除匹配异常的点
void KeyFrame::PnPRANSAC(
	const vector<cv::Point2f> &matched_2d_old_norm,
    const std::vector<cv::Point3f> &matched_3d,
    std::vector<uchar> &status,
    Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old) {
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;

	// 将vio前端传过来的w->bk基准坐标系转换为w->ck
	Matrix3d R_w_c = origin_vio_R * qic;
	Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

	// 将坐标系从世界坐标基准转换为相机坐标基准
    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;

	// TODO：此处匹配用的是回环帧的2D点以及当前帧的3D点，是否有问题
	if (CV_MAJOR_VERSION < 3) {
		cv::solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
	}
	else {
		if (CV_MINOR_VERSION < 2) {
			cv::solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
		}
		else {
			cv::solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);
		}
    }

	// 统计匹配内点
	for (int i = 0; i < (int)matched_2d_old_norm.size(); i++) {
		status.push_back(0);
	}

    for( int i = 0; i < inliers.rows; i++) {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic.transpose();
	PnP_T_old = T_w_c_old - PnP_R_old * tic;
}

// 寻找与建立当前关键帧与回环帧间的匹配关系
bool KeyFrame::findConnection(
	KeyFrame* old_kf) {
	TicToc tmp_t;
	std::vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	std::vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	std::vector<cv::Point3f> matched_3d;
	std::vector<double> matched_id;
	std::vector<uchar> status;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	TicToc t_match;
	#if 0
		if (DEBUG_IMAGE)    
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif

	// 关键帧与回环帧进行BRIEF描述子匹配，剔除匹配失败的点
	searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);

	#if 0 
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);	        
	        */
	        
	    }
	#endif

	status.clear();
	/*
	FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	*/
	#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif

	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;

	// 达到回环最小匹配点数，使用Ransac PnP筛选内点
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
		status.clear();
		PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
	    reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
		reduceVector(matched_id, status);

	    #if 1
	    	if (DEBUG_IMAGE) {
	        	int gap = 10;
				cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
				cv::Mat gray_img, loop_match_img;
				cv::Mat old_img = old_kf->image;

	            cv::hconcat(image, gap_image, gap_image);
	            cv::hconcat(gap_image, old_img, gray_img);

	            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);

	            // 当前帧与闭环帧之间绘制匹配点对并连线
	            for(int i = 0; i< (int)matched_2d_cur.size(); i++) {
	                cv::Point2f cur_pt = matched_2d_cur[i];
	                cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for(int i = 0; i< (int)matched_2d_old.size(); i++) {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap);
	                cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for (int i = 0; i< (int)matched_2d_cur.size(); i++) {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap) ;
	                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
	            }

	            // 标注当前帧与回环帧的索引值以及序列号
				cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
	            putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
	            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
	            cv::vconcat(notation, loop_match_img, loop_match_img);

	            // 若达到最小匹配点数，图像缩小并发送出去
	            if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
	            	cv::Mat thumbimage;
	            	cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
	    	    	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
	                msg->header.stamp = ros::Time(time_stamp);
	    	    	pub_match_img.publish(msg);
	            }
	        }
	    #endif
	}

	// 若达到最小回环点数，计算相对位姿并发布匹配点信息
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
		// TODO：此处的相对位姿的计算方式是否有误
	    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
	    relative_q = PnP_R_old.transpose() * origin_vio_R;
	    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
	    
	    // 相对位姿检验
	    if (std::abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0) {
			// 更新闭环信息
	    	has_loop = true;
	    	loop_index = old_kf->index;
			loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
				relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
				relative_yaw;

	    	// 快速重定位
	    	if(FAST_RELOCALIZATION) {
			    sensor_msgs::PointCloud msg_match_points;
			    msg_match_points.header.stamp = ros::Time(time_stamp);
			    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++) {
		            geometry_msgs::Point32 p;
		            p.x = matched_2d_old_norm[i].x;
		            p.y = matched_2d_old_norm[i].y;
					p.z = matched_id[i];
		            msg_match_points.points.push_back(p);
			    }
			    Eigen::Vector3d T = old_kf->T_w_i; 
			    Eigen::Matrix3d R = old_kf->R_w_i;
			    Quaterniond Q(R);
			    sensor_msgs::ChannelFloat32 t_q_index;
			    t_q_index.values.push_back(T.x());
			    t_q_index.values.push_back(T.y());
			    t_q_index.values.push_back(T.z());
			    t_q_index.values.push_back(Q.w());
			    t_q_index.values.push_back(Q.x());
			    t_q_index.values.push_back(Q.y());
			    t_q_index.values.push_back(Q.z());
			    t_q_index.values.push_back(index);
			    msg_match_points.channels.push_back(t_q_index);
				pub_match_points.publish(msg_match_points);
	    	}
	        return true;
	    }
	}

	return false;
}

// 计算两个描述子之间的汉明距离
int KeyFrame::HammingDis(
	const BRIEF::bitset &a, 
	const BRIEF::bitset &b) {
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::getVioPose(
	Eigen::Vector3d &_T_w_i, 
	Eigen::Matrix3d &_R_w_i) {
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(
	Eigen::Vector3d &_T_w_i, 
	Eigen::Matrix3d &_R_w_i) {
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(
	const Eigen::Vector3d &_T_w_i, 
	const Eigen::Matrix3d &_R_w_i) {
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(
	const Eigen::Vector3d &_T_w_i, 
	const Eigen::Matrix3d &_R_w_i) {
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT() {
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ() {
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw() {
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info) {
	if (std::abs(_loop_info(7)) < 30.0 && 
		Eigen::Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
		loop_info = _loop_info;
	}
}

// 读取描述子构建所需的模板文件
BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
	// The DVision::BRIEF extractor computes a random pattern by default when
	// the object is created.
	// We load the pattern that we used to build the vocabulary, to make
	// the descriptors compatible with the predefined vocabulary

	cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
	if (!fs.isOpened()) throw string("Could not open file ") + pattern_file;

	std::vector<int> x1, y1, x2, y2;
	fs["x1"] >> x1;
	fs["x2"] >> x2;
	fs["y1"] >> y1;
	fs["y2"] >> y2;

	m_brief.importPairs(x1, y1, x2, y2);
}