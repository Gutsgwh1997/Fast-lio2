#include "lio_builder/loop_closure.h"

namespace fastlio {
ZaxisPriorFactor::ZaxisPriorFactor(gtsam::Key key, const gtsam::SharedNoiseModel &noise, double z)
    : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise, key), z_(z) {}

ZaxisPriorFactor::~ZaxisPriorFactor() {}

gtsam::Vector ZaxisPriorFactor::evaluateError(const gtsam::Pose3 &p, boost::optional<gtsam::Matrix &> H) const {
    auto z = p.translation()(2);
    if (H) {
        gtsam::Matrix Jac = gtsam::Matrix::Zero(1, 6);
        Jac << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
        (*H) = Jac;
    }
    return gtsam::Vector1(z - z_);
}

void LoopClosureThread::init() {
    gtsam::ISAM2Params isam2_params;
    isam2_params.relinearizeThreshold = 0.01;
    isam2_params.relinearizeSkip = 1;
    isam2_ = std::make_shared<gtsam::ISAM2>(isam2_params);
    kdtree_history_poses_.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>);
    cloud_history_poses_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    sub_map_downsize_filter_.reset(new pcl::VoxelGrid<fastlio::PointType>);
    sub_map_downsize_filter_->setLeafSize(loop_params_.submap_resolution, loop_params_.submap_resolution,
                                          loop_params_.submap_resolution);

    icp_.reset(new pcl::IterativeClosestPoint<fastlio::PointType, fastlio::PointType>);
    icp_->setMaxCorrespondenceDistance(100);
    icp_->setMaximumIterations(50);
    icp_->setTransformationEpsilon(1e-6);
    icp_->setEuclideanFitnessEpsilon(1e-6);
    icp_->setRANSACIterations(0);
}

void LoopClosureThread::setShared(const std::shared_ptr<SharedData> &share_data) { shared_data_ = share_data; }

void LoopClosureThread::setRate(double rate) { update_rate_ = 1.0 / (rate + 1e-5); }

LoopParams &LoopClosureThread::mutableParams() { return loop_params_; }

fastlio::PointCloudXYZI::Ptr LoopClosureThread::getSubMaps(std::vector<Pose6D> &pose_list,
                                                           std::vector<fastlio::PointCloudXYZI::Ptr> &cloud_list,
                                                           int index, int search_num) {
    fastlio::PointCloudXYZI::Ptr cloud(new fastlio::PointCloudXYZI);
    int max_size = pose_list.size();
    int min_index = std::max(0, index - search_num);
    int max_index = std::min(max_size - 1, index + search_num);
    for (int i = min_index; i <= max_index; i++) {
        const Pose6D &p = pose_list[i];
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = p.global_rot;
        T.block<3, 1>(0, 3) = p.global_pos;
        fastlio::PointCloudXYZI::Ptr temp_cloud(new fastlio::PointCloudXYZI);
        pcl::transformPointCloud(*cloud_list[p.index], *temp_cloud, T);
        *cloud += *temp_cloud;
    }
    sub_map_downsize_filter_->setInputCloud(cloud);
    sub_map_downsize_filter_->filter(*cloud);
    return cloud;
}

void LoopClosureThread::operator()() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(int(1000 * update_rate_)));
        if (terminate_flag) break;
        if (!loop_params_.activate) continue;
        if (shared_data_->key_poses.size() < loop_params_.loop_pose_index_thresh) continue;
        if (!shared_data_->key_pose_added) continue;

        shared_data_->key_pose_added = false;
        {
            std::lock_guard<std::mutex> lock(shared_data_->mutex);
            lastest_index_ = shared_data_->key_poses.size() - 1;
            temp_poses_.clear();
            temp_poses_.assign(shared_data_->key_poses.begin(), shared_data_->key_poses.end());
        }

        loopCheck();
        addOdomFactor();
        addLoopFactor();
        smoothAndUpdate();
    }
}

void LoopClosureThread::loopCheck() {
    if (temp_poses_.empty()) {
        return;
    }
    int pre_index = -1;
    int cur_index = temp_poses_.size() - 1;

    cloud_history_poses_->clear();
    for (const Pose6D &p : temp_poses_) {
        pcl::PointXYZ point;
        point.x = p.global_pos(0);
        point.y = p.global_pos(1);
        point.z = p.global_pos(2);
        cloud_history_poses_->push_back(point);
    }
    kdtree_history_poses_->setInputCloud(cloud_history_poses_);
    std::vector<int> ids;
    std::vector<float> sqdists;
    double loop_search_r = loop_params_.loop_pose_search_radius;
    kdtree_history_poses_->radiusSearch(cloud_history_poses_->back(), loop_search_r, ids, sqdists, 0);

    for (int i = 0; i < ids.size(); i++) {
        int id = ids[i];
        if (std::abs(temp_poses_[id].time - temp_poses_.back().time) > loop_params_.time_thresh) {
            pre_index = id;
            break;
        }
    }
    if (pre_index == -1 || pre_index == cur_index || cur_index - pre_index < loop_params_.loop_pose_index_thresh) {
        return;
    }

    auto &cloud_history = shared_data_->cloud_history;
    auto cur_cloud = getSubMaps(temp_poses_, cloud_history, cur_index, 0);
    auto sub_maps = getSubMaps(temp_poses_, cloud_history, pre_index, loop_params_.submap_search_num);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_cloud_xyz(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr sub_maps_xyz(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*cur_cloud, *cur_cloud_xyz);
    pcl::copyPointCloud(*sub_maps, *sub_maps_xyz);
    cur_cloud = addNorm(cur_cloud_xyz);
    sub_maps = addNorm(sub_maps_xyz);

    icp_->setInputSource(cur_cloud);
    icp_->setInputTarget(sub_maps);

    fastlio::PointCloudXYZI::Ptr aligned(new fastlio::PointCloudXYZI);

    icp_->align(*aligned, Eigen::Matrix4f::Identity());

    float score = icp_->getFitnessScore();
    if (!icp_->hasConverged() || score > loop_params_.loop_icp_thresh) {
        LOG_WARN("Pre_index: %d, cur_index: %d, icp_score: %f.", pre_index, cur_index, score);
        return;
    }

    loop_found_ = true;
    shared_data_->loop_history.emplace_back(pre_index, cur_index);
    LOG_INFO("Detected loop, pre_index: %d, cur_index: %d, icp_score: %f.", pre_index, cur_index, score);

    Eigen::Matrix4d T_pre_cur = icp_->getFinalTransformation().cast<double>();
    Eigen::Matrix3d R12 =
        temp_poses_[pre_index].global_rot.transpose() * T_pre_cur.block<3, 3>(0, 0) * temp_poses_[cur_index].global_rot;
    Eigen::Vector3d t12 = temp_poses_[pre_index].global_rot.transpose() *
                          (T_pre_cur.block<3, 3>(0, 0) * temp_poses_[cur_index].global_pos +
                           T_pre_cur.block<3, 1>(0, 3) - temp_poses_[pre_index].global_pos);
    shared_data_->loop_pairs.emplace_back(pre_index, cur_index, score, R12, t12);
}

void LoopClosureThread::addOdomFactor() {
    for (int i = previous_index_; i < lastest_index_; i++) {
        Pose6D &p1 = temp_poses_[i];
        Pose6D &p2 = temp_poses_[i + 1];

        if (i == 0) {
            initialized_estimate_.insert(i, gtsam::Pose3(gtsam::Rot3(p1.local_rot), gtsam::Point3(p1.local_pos)));
            gtsam::noiseModel::Diagonal::shared_ptr noise =
                gtsam::noiseModel::Diagonal::Variances(gtsam::Vector6::Ones() * 1e-12);
            gtsam_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
                i, gtsam::Pose3(gtsam::Rot3(p1.local_rot), gtsam::Point3(p1.local_pos)), noise));
        }
        initialized_estimate_.insert(i + 1, gtsam::Pose3(gtsam::Rot3(p2.local_rot), gtsam::Point3(p2.local_pos)));
        Eigen::Matrix3d R12 = p1.local_rot.transpose() * p2.local_rot;
        Eigen::Vector3d t12 = p1.local_rot.transpose() * (p2.local_pos - p1.local_pos);

        // ！！！！！！！！！！！carefully add this！！！！！！！！！！！
        // gtsam::noiseModel::Diagonal::shared_ptr noise_prior =
        // gtsam::noiseModel::Diagonal::Variances(gtsam::Vector1::Ones()); gtsam_graph_.add(ZaxisPriorFactor(i + 1,
        // noise_prior, p2.local_pos(2)));

        gtsam::noiseModel::Diagonal::shared_ptr noise =
            gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-6).finished());
        gtsam_graph_.add(
            gtsam::BetweenFactor<gtsam::Pose3>(i, i + 1, gtsam::Pose3(gtsam::Rot3(R12), gtsam::Point3(t12)), noise));
    }
    previous_index_ = lastest_index_;
}

void LoopClosureThread::addLoopFactor() {
    if (!loop_found_) return;
    if (shared_data_->loop_pairs.empty()) return;
    for (LoopPair &lp : shared_data_->loop_pairs) {
        gtsam::Pose3 pose_between(gtsam::Rot3(lp.diff_rot), gtsam::Point3(lp.diff_pos));
        gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            lp.pre_idx, lp.cur_idx, pose_between,
            gtsam::noiseModel::Diagonal::Variances(gtsam::Vector6::Ones() * lp.score)));
    }
    shared_data_->loop_pairs.clear();
}

void LoopClosureThread::smoothAndUpdate() {
    isam2_->update(gtsam_graph_, initialized_estimate_);
    isam2_->update();
    if (loop_found_) {
        isam2_->update();
        isam2_->update();
        isam2_->update();
        isam2_->update();
        isam2_->update();
        loop_found_ = false;
    }
    gtsam_graph_.resize(0);
    initialized_estimate_.clear();

    optimized_estimate_ = isam2_->calculateBestEstimate();
    gtsam::Pose3 latest_estimate = optimized_estimate_.at<gtsam::Pose3>(lastest_index_);
    temp_poses_[lastest_index_].global_rot = latest_estimate.rotation().matrix().cast<double>();
    temp_poses_[lastest_index_].global_pos = latest_estimate.translation().matrix().cast<double>();
    Eigen::Matrix3d offset_rot;
    Eigen::Vector3d offset_pos;
    temp_poses_[lastest_index_].getOffset(offset_rot, offset_pos);

    shared_data_->mutex.lock();
    int current_size = shared_data_->key_poses.size();
    shared_data_->offset_rot = offset_rot;
    shared_data_->offset_pos = offset_pos;
    shared_data_->mutex.unlock();

    for (int i = 0; i < lastest_index_; i++) {
        gtsam::Pose3 temp_pose = optimized_estimate_.at<gtsam::Pose3>(i);
        shared_data_->key_poses[i].global_rot = temp_pose.rotation().matrix().cast<double>();
        shared_data_->key_poses[i].global_pos = temp_pose.translation().matrix().cast<double>();
    }

    for (int i = lastest_index_; i < current_size; i++) {
        shared_data_->key_poses[i].addOffset(offset_rot, offset_pos);
    }
}
}  // namespace fastlio
