#include "localizer/icp_localizer.h"

#include <chrono>

namespace fastlio {
IcpLocalizer::IcpLocalizer()
    : refine_resolution_(0.2), rough_resolution_(0.5), rough_iter_(10), refine_iter_(5), thresh_(0.15) {
    voxel_rough_filter_.setLeafSize(rough_resolution_, rough_resolution_, rough_resolution_);
    voxel_refine_filter_.setLeafSize(refine_resolution_, refine_resolution_, refine_resolution_);
}

IcpLocalizer::IcpLocalizer(double refine_resolution, double rough_resolution, int refine_iter, int rough_iter,
                           double thresh)
    : refine_resolution_(refine_resolution),
      rough_resolution_(rough_resolution),
      refine_iter_(refine_iter),
      rough_iter_(rough_iter),
      thresh_(thresh) {
    voxel_rough_filter_.setLeafSize(rough_resolution_, rough_resolution_, rough_resolution_);
    voxel_refine_filter_.setLeafSize(refine_resolution_, refine_resolution_, refine_resolution_);
}

void IcpLocalizer::init(const std::string &pcd_path, bool with_norm) {
    if (!pcd_path_.empty() && pcd_path_ == pcd_path) {
        LOG_WARN("The pcd path is the same as the last one, no need to reinitialize!");
        return;
    }

    pcd_path_ = pcd_path;
    pcl::PCDReader reader;
    if (!with_norm) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        reader.read(pcd_path, *cloud);
        size_t ori_size = cloud->size();
        voxel_refine_filter_.setInputCloud(cloud);
        voxel_refine_filter_.filter(*cloud);
        refine_map_ = addNorm(cloud);
        LOG_INFO("Read %ld point from pcd file: %s.", ori_size, pcd_path.c_str());
    } else {
        refine_map_.reset(new PointCloudXYZI);
        reader.read(pcd_path, *refine_map_);
        LOG_INFO("Read %ld point from pcd file: %s.", refine_map_->size(), pcd_path.c_str());
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr point_rough(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr filterd_point_rough(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*refine_map_, *point_rough);
    voxel_rough_filter_.setInputCloud(point_rough);
    voxel_rough_filter_.filter(*filterd_point_rough);
    rough_map_ = addNorm(filterd_point_rough);

    icp_rough_.setMaximumIterations(rough_iter_);
    icp_rough_.setUseSymmetricObjective(true);
    icp_rough_.setUseReciprocalCorrespondences(true);
    icp_rough_.setMaxCorrespondenceDistance(3.0);
    icp_rough_.setRANSACIterations(rough_iter_ / 3.0);
    icp_rough_.setRANSACOutlierRejectionThreshold(0.05);
    icp_rough_.setTransformationEpsilon(1e-6);
    icp_rough_.setEuclideanFitnessEpsilon(1e-6);
    icp_rough_.setInputTarget(rough_map_);

    icp_refine_.setMaximumIterations(refine_iter_);
    icp_refine_.setUseReciprocalCorrespondences(true);
    icp_refine_.setMaxCorrespondenceDistance(0.4);
    icp_refine_.setRANSACIterations(rough_iter_ / 5.0);
    icp_refine_.setRANSACOutlierRejectionThreshold(0.04);
    icp_refine_.setTransformationEpsilon(1e-6);
    icp_refine_.setEuclideanFitnessEpsilon(1e-6);
    icp_refine_.setInputTarget(refine_map_);

    initialized_ = true;

    LOG_INFO("There are %ld points in rough_map, rough_resolution: %f.", rough_map_->size(), rough_resolution_);
    LOG_INFO("There are %ld points in refine_map, refine_resolution: %f.", refine_map_->size(), refine_resolution_);
}

Eigen::Matrix4d IcpLocalizer::align(pcl::PointCloud<pcl::PointXYZI>::Ptr source, const Eigen::Matrix4d &init_guess) {
    success_ = false;
    Eigen::Vector3d xyz = init_guess.block<3, 1>(0, 3);

    pcl::PointCloud<pcl::PointXYZI>::Ptr rough_source(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr refine_source(new pcl::PointCloud<pcl::PointXYZI>);

    voxel_rough_filter_.setInputCloud(source);
    voxel_rough_filter_.filter(*rough_source);
    voxel_refine_filter_.setInputCloud(source);
    voxel_refine_filter_.filter(*refine_source);

    PointCloudXYZI::Ptr rough_source_norm = addNorm(rough_source);
    PointCloudXYZI::Ptr refine_source_norm = addNorm(refine_source);
    PointCloudXYZI::Ptr align_point(new PointCloudXYZI);
    icp_rough_.setInputSource(rough_source_norm);
    icp_rough_.align(*align_point, init_guess.cast<float>());

    score_ = icp_rough_.getFitnessScore();
    if (!icp_rough_.hasConverged()) return Eigen::Matrix4d::Zero();

    icp_refine_.setInputSource(refine_source_norm);
    icp_refine_.align(*align_point, icp_rough_.getFinalTransformation());
    score_ = icp_refine_.getFitnessScore();

    if (!icp_refine_.hasConverged()) {
        return Eigen::Matrix4d::Zero();
    }
    if (score_ > thresh_) {
        return Eigen::Matrix4d::Zero();
    }
    success_ = true;
    return icp_refine_.getFinalTransformation().cast<double>();
}

void IcpLocalizer::writePCDToFile(const std::string &path, bool detail) {
    if (!initialized_) return;
    pcl::PCDWriter writer;
    writer.writeBinaryCompressed(path, detail ? *refine_map_ : *rough_map_);
}

void IcpLocalizer::setParams(double refine_resolution, double rough_resolution, int refine_iter, int rough_iter,
                             double thresh) {
    refine_resolution_ = refine_resolution;
    rough_resolution_ = rough_resolution;
    refine_iter_ = refine_iter;
    rough_iter_ = rough_iter;
    thresh_ = thresh;
}

void IcpLocalizer::setSearchParams(double xy_offset, int yaw_offset, double yaw_res) {
    xy_offset_ = xy_offset;
    yaw_offset_ = yaw_offset;
    yaw_resolution_ = yaw_res;
}

Eigen::Matrix4d IcpLocalizer::multi_align_sync(pcl::PointCloud<pcl::PointXYZI>::Ptr source,
                                               const Eigen::Matrix4d &init_guess) {
    success_ = false;
    Eigen::Vector3d xyz = init_guess.block<3, 1>(0, 3);
    Eigen::Matrix3d rotation = init_guess.block<3, 3>(0, 0);
    Eigen::Vector3d rpy = rotate2rpy(rotation);
    Eigen::AngleAxisf rollAngle(rpy(0), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(rpy(1), Eigen::Vector3f::UnitY());

    Eigen::Matrix4f temp_pose;
    std::vector<Eigen::Matrix4f> candidates;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            for (int k = -yaw_offset_; k <= yaw_offset_; k++) {
                Eigen::Vector3f pos(xyz(0) + i * xy_offset_, xyz(1) + j * xy_offset_, xyz(2));
                Eigen::AngleAxisf yawAngle(rpy(2) + k * yaw_resolution_, Eigen::Vector3f::UnitZ());
                temp_pose.setIdentity();
                temp_pose.block<3, 3>(0, 0) = (rollAngle * pitchAngle * yawAngle).toRotationMatrix();
                temp_pose.block<3, 1>(0, 3) = pos;
                candidates.push_back(temp_pose);
            }
        }
    }
    pcl::PointCloud<pcl::PointXYZI>::Ptr rough_source(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr refine_source(new pcl::PointCloud<pcl::PointXYZI>);

    voxel_rough_filter_.setInputCloud(source);
    voxel_rough_filter_.filter(*rough_source);
    voxel_refine_filter_.setInputCloud(source);
    voxel_refine_filter_.filter(*refine_source);

    PointCloudXYZI::Ptr rough_source_norm = addNorm(rough_source);
    PointCloudXYZI::Ptr refine_source_norm = addNorm(refine_source);

    PointCloudXYZI::Ptr align_point(new PointCloudXYZI);

    bool rough_converge = false;
    Eigen::Matrix4f best_rough_transform;
    double best_rough_score = min_best_rough_score_;
    for (const Eigen::Matrix4f &init_pose : candidates) {
        icp_rough_.setInputSource(rough_source_norm);
        icp_rough_.align(*align_point, init_pose);
        if (!icp_rough_.hasConverged()) {
            continue;
        }
        double rough_score = icp_rough_.getFitnessScore();
        if (rough_score > 2 * thresh_) {
            continue;
        }
        if (rough_score < best_rough_score) {
            rough_converge = true;
            best_rough_score = rough_score;
            best_rough_transform = icp_rough_.getFinalTransformation();
        }
    }
    LOG_INFO("Rough align score: %f, is rough_converge: %d.", best_rough_score, rough_converge);

    if (!rough_converge) {
        return Eigen::Matrix4d::Zero();
    }

    icp_refine_.setInputSource(refine_source_norm);
    icp_refine_.align(*align_point, best_rough_transform);
    score_ = icp_refine_.getFitnessScore();
    LOG_INFO("Refine align score: %f, good icp thresh: %f.", score_, thresh_);
    if (!icp_refine_.hasConverged() || score_ > thresh_) {
        return Eigen::Matrix4d::Zero();
    }
    success_ = true;
    return icp_refine_.getFinalTransformation().cast<double>();
}
}  // namespace fastlio