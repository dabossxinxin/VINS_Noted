#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt) {
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
	return BORDER_SIZE <= img_x &&
		img_x < COL - BORDER_SIZE &&
		BORDER_SIZE <= img_y &&
		img_y < ROW - BORDER_SIZE;
}

void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(std::vector<int> &v, std::vector<uchar> status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker() {}

/**
 * @brief   对跟踪点进行排序并去除密集点
 * @Description 对跟踪到的特征点，按照被追踪到的次数排序并依次选点
 *              使用mask进行类似非极大抑制，半径为30，去掉密集点，使特征点分布均匀            
 * @return      void
*/
void FeatureTracker::setMask() {
	// 设置特征点提取mask
	if (FISHEYE)
		mask = fisheye_mask.clone();
	else
		mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    
    // 构造(cnt，pts，id)序列
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;
	for (unsigned int i = 0; i < forw_pts.size(); ++i)
		cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    // 对LK光流跟踪到的特征点按照其被跟踪的次数进行排序
    std::sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](
		const std::pair<int, pair<cv::Point2f, int>> &a, 
		const std::pair<int, pair<cv::Point2f, int>> &b) {
            return a.first > b.first;
	});

	// 选取跟踪次数多的、排布均匀的特征点作为当前帧的有效特征
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id) {
        if (mask.at<uchar>(it.second.first) == 255) {
			forw_pts.push_back(it.second.first);
			ids.push_back(it.second.second);
			track_cnt.push_back(it.first);

			cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

// 添将新检测到的特征点n_pts
void FeatureTracker::addPoints() {
    for (auto &p : n_pts) {
        forw_pts.push_back(p);
        ids.push_back(-1);		//新提取的特征点id初始化为-1
        track_cnt.push_back(1);	//新提取的特征点被跟踪的次数初始化为1
    }
}

/**
 * @brief   对图像使用光流法进行特征点跟踪
 * @Description createCLAHE() 对图像进行自适应直方图均衡化
 *              calcOpticalFlowPyrLK() LK金字塔光流法
 *              setMask() 对跟踪点进行排序，设置mask
 *              rejectWithF() 通过基本矩阵剔除outliers
 *              goodFeaturesToTrack() 添加特征点(shi-tomasi角点)，确保每帧都有足够的特征点
 *              addPoints()添加新的追踪点
 *              undistortedPoints() 对角点图像坐标去畸变矫正，并计算每个角点的速度
 * @param[in]   _img 输入图像
 * @param[in]   _cur_time 当前时间（图像时间戳）
 * @return      void
*/
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time) {
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    // 若图像太亮或者太暗，此时进行自适应直方图均衡化
    if (EQUALIZE) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
	else {
		img = _img;
	}

	// 判断是否为首次读入图像数据，forw_img表示当前帧图像
    if (forw_img.empty()) {
        prev_img = cur_img = forw_img = img;
    }
    else {
        forw_img = img;
    }
	
	forw_pts.clear();
    if (cur_pts.size() > 0) {
        TicToc t_o;
        std::vector<uchar> status;
        std::vector<float> err;

        // LK光流特征点跟踪，status标记跟踪状态，0为跟踪失败
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        // 将越界的跟踪点状态置为0
		for (int i = 0; i < int(forw_pts.size()); ++i) {
			if (status[i] && !inBorder(forw_pts[i])) {
				status[i] = 0;
			}
		}

        // 根据status,把跟踪失败的点剔除
        // 不仅要从当前帧数据forw_pts中剔除，而且还要从cur_un_pts、prev_pts和cur_pts中剔除
        // prev_pts和cur_pts中的特征点是一一对应的
        // 记录特征点id的ids，和记录特征点被跟踪次数的track_cnt也要剔除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    // LK光流跟踪成功，被跟踪次数加1
	for (auto &n : track_cnt) {
		n++;
	}

    // 需要发布当前帧时，对当前帧进行rejectWithF()操作
	// 该操作比较耗时，所以只有在需要发布时才进行
    if (PUB_THIS_FRAME) {
		rejectWithF();

        ROS_DEBUG("set mask begins");
        TicToc t_m;

        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;

        // 通过rejectWithF()以及均匀化去掉了一些特征
		// 为了保证后续跟踪的鲁棒性，此时要多跟踪一些特征出来
		int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0) {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            /** 
             *void cv::goodFeaturesToTrack(			 在mask中不为0的区域检测新的特征点
             *   InputArray  image,					 输入图像
             *   OutputArray corners,				 存放检测到的角点的vector
             *   int		 maxCorners,             返回的角点的数量的最大值
             *   double		 qualityLevel,           角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
             *   double		 minDistance,            返回角点之间欧式距离的最小值
             *   InputArray  mask = noArray(),		 和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
             *   int		 blockSize = 3,          计算协方差矩阵时的窗口大小
             *   bool    useHarrisDetector = false,  指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法
             *   double  k = 0.04					 Harris角点检测需要的k值
             *)   
             */
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
		else {
			n_pts.clear();
		}
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;

        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    // 当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;

    // 把当前帧的数据forw_img、forw_pts赋给上一帧cur_img、cur_pts
    cur_img = forw_img;
    cur_pts = forw_pts;

    // 根据不同的相机模型去畸变矫正和转换到归一化坐标系上，计算速度
	undistortedPoints();
	prev_time = cur_time;
}

void FeatureTracker::rejectWithF() {
    if (forw_pts.size() >= 8) {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;

		std::vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); ++i) {
            // 根据不同的相机模型将二维坐标转换到三维坐标
			Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);

            // 转换为归一化像素坐标
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

		//调用cv::findFundamentalMat对un_cur_pts和un_forw_pts计算F矩阵
        std::vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);

		reduceVector(prev_pts, status);
		reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);

		ROS_DEBUG("FM ransac: %d -> %lu: %f", cur_pts.size(), forw_pts.size(), 1.0 * forw_pts.size() / cur_pts.size());
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i) {
    if (i < ids.size()) {
		if (ids[i] == -1)
			ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file) {
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name) {
	cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    std::vector<Eigen::Vector2d> distortedp, undistortedp;
	for (int i = 0; i < COL; ++i) {
		for (int j = 0; j < ROW; ++j) {
			Eigen::Vector2d a(i, j);
			Eigen::Vector3d b;
			m_camera->liftProjective(a, b);
			distortedp.push_back(a);
			undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
		}
	}
    for (int i = 0; i < int(undistortedp.size()); ++i) {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        if (pp.at<float>(1, 0) + 300 >= 0 && 
			pp.at<float>(1, 0) + 300 < ROW + 600 && 
			pp.at<float>(0, 0) + 300 >= 0 && 
			pp.at<float>(0, 0) + 300 < COL + 600) {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
	cv::imshow(name, undistortedImg);
	cv::waitKey(0);
}
                     
void FeatureTracker::undistortedPoints() {
    cur_un_pts.clear();
    cur_un_pts_map.clear();

	// 计算当前特征点的去畸变归一化坐标
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
		Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
		Eigen::Vector3d b;
		m_camera->liftProjective(a, b);

		cv::Point2f lift_pt(b.x() / b.z(), b.y() / b.z());
		cur_un_pts.push_back(lift_pt);
		cur_un_pts_map.insert(make_pair(ids[i], lift_pt));
    }

    // 计算当前每个特征点的速度
    if (!prev_un_pts_map.empty()) {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++) {
            if (ids[i] != -1) {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end()) {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
					pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
				else {
					pts_velocity.push_back(cv::Point2f(0, 0));
				}
            }
            else {
				pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    } else {
        for (unsigned int i = 0; i < cur_pts.size(); i++) {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
	prev_un_pts_map = cur_un_pts_map;
}